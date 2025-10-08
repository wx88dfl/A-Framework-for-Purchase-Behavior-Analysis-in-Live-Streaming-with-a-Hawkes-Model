import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

class ODELSTMCell(nn.Module):
    """
    ODE-LSTM单元，使用神经ODE定义LSTM状态之间的连续转换
    """
    def __init__(self, input_size, hidden_size):
        super(ODELSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM门控
        self.fc_i = nn.Linear(input_size + hidden_size, hidden_size)  # 输入门
        self.fc_f = nn.Linear(input_size + hidden_size, hidden_size)  # 遗忘门
        self.fc_o = nn.Linear(input_size + hidden_size, hidden_size)  # 输出门
        self.fc_g = nn.Linear(input_size + hidden_size, hidden_size)  # 细胞状态更新
        
        # ODE函数，定义隐藏状态的连续演化
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x, h, c, t):
        """
        x: 输入特征
        h: 隐藏状态
        c: 细胞状态
        t: 时间间隔
        """
        # LSTM门控计算
        combined = torch.cat([x, h], dim=1)
        i = self.sigmoid(self.fc_i(combined))
        f = self.sigmoid(self.fc_f(combined))
        o = self.sigmoid(self.fc_o(combined))
        g = self.tanh(self.fc_g(combined))
        
        # 更新细胞状态
        c = f * c + i * g
        
        # ODE部分：计算隐藏状态随时间的连续演化
        def odefunc(t, h):
            return self.ode_func(h)
        
        # 求解ODE得到新的隐藏状态
        h_new = odeint(odefunc, h, torch.tensor([0.0, t], device=x.device))[-1]
        
        # 应用输出门
        h_new = o * self.tanh(h_new)
        
        return h_new, c

class LHawkesModel(nn.Module):
    """
    Live Broadcast Neural Hawkes (LHawkes)模型
    包含ODE-LSTM编码器和两个解码器（事件级和直播间级）
    """
    def __init__(self, input_dim, hidden_dim, num_event_types, num_quality_levels):
        super(LHawkesModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_event_types = num_event_types
        
        # 事件类型嵌入
        self.event_embedding = nn.Embedding(num_event_types, input_dim)
        
        # ODE-LSTM编码器
        self.encoder = ODELSTMCell(input_dim, hidden_dim)
        
        # 事件级解码器 - 预测下一个事件类型和时间
        self.event_type_decoder = nn.Linear(hidden_dim, num_event_types)
        self.event_time_decoder = nn.Linear(hidden_dim, 1)
        
        # 直播间级解码器 - 预测直播质量
        self.room_quality_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_quality_levels)
        )
        
        # 用于时间预测的激活函数参数
        self.s = nn.Parameter(torch.tensor(1.0))  # 公式17中的比例超参数
        
    def forward(self, event_sequences, timestamps):
        """
        event_sequences: 事件类型序列 (batch_size, seq_len)
        timestamps: 事件时间戳序列 (batch_size, seq_len)
        """
        batch_size, seq_len = event_sequences.shape
        device = event_sequences.device
        
        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # 存储所有时间步的隐藏状态
        hidden_states = []
        
        # 处理序列中的每个事件
        for i in range(seq_len):
            # 获取当前事件的嵌入
            event_type = event_sequences[:, i]
            event_emb = self.event_embedding(event_type)
            
            # 计算时间间隔
            if i == 0:
                t_interval = torch.zeros(batch_size, device=device)
            else:
                t_interval = timestamps[:, i] - timestamps[:, i-1]
            
            # 更新ODE-LSTM状态
            h, c = self.encoder(event_emb, h, c, t_interval)
            hidden_states.append(h)
        
        # 堆叠所有隐藏状态
        hidden_states = torch.stack(hidden_states, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        # 事件级预测
        # 预测下一个事件类型的强度函数
        last_hidden = hidden_states[:, -1, :]  # 最后一个时间步的隐藏状态
        event_type_logits = self.event_type_decoder(last_hidden)  # (batch_size, num_event_types)
        
        # 应用公式17中的激活函数
        event_type_intensity = torch.log(1 + torch.exp(self.s * event_type_logits))
        
        # 预测下一个事件的时间
        event_time_pred = torch.exp(self.event_time_decoder(last_hidden))  # 确保时间为正
        
        # 直播间级预测
        # 使用平均池化聚合所有时间步的隐藏状态
        room_level_feat = torch.mean(hidden_states, dim=1)  # (batch_size, hidden_dim)
        quality_logits = self.room_quality_decoder(room_level_feat)  # (batch_size, num_quality_levels)
        
        return {
            'event_type_intensity': event_type_intensity,
            'event_time_pred': event_time_pred.squeeze(),
            'quality_logits': quality_logits,
            'hidden_states': hidden_states
        }
    
    def loss_function(self, outputs, targets):
        """
        计算总损失函数，包括事件级损失和直播间级损失
        """
        # 事件类型预测损失 (公式20)
        event_type_target = targets['next_event_type']
        event_type_intensity = outputs['event_type_intensity']
        
        # 计算风险函数 (公式19)
        risk = torch.sum(event_type_intensity, dim=1)
        # 计算事件类型损失
        event_type_loss = -torch.mean(torch.log(event_type_intensity[torch.arange(event_type_intensity.shape[0]), 
                                                                    event_type_target]) - risk)
        
        # 事件时间预测损失 (公式24)
        event_time_target = targets['next_event_time']
        event_time_pred = outputs['event_time_pred']
        event_time_loss = nn.MSELoss()(event_time_pred, event_time_target)
        
        # 事件级总损失 (公式25)
        event_level_loss = event_type_loss + event_time_loss
        
        # 直播间质量预测损失 (公式27)
        quality_target = targets['quality_level']
        quality_logits = outputs['quality_logits']
        room_level_loss = nn.CrossEntropyLoss()(quality_logits, quality_target)
        
        # 总损失 (公式28)
        total_loss = event_level_loss + room_level_loss
        
        return {
            'total_loss': total_loss,
            'event_type_loss': event_type_loss,
            'event_time_loss': event_time_loss,
            'event_level_loss': event_level_loss,
            'room_level_loss': room_level_loss
        }
