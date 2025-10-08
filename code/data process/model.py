import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np

class ODEFunc(nn.Module):
    """定义ODE函数，用于ODE-LSTM中的连续时间演化"""
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 初始化权重
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, x):
        """
        Args:
            t: 时间
            x: 隐藏状态 [batch_size, hidden_dim]
        Returns:
            隐藏状态的导数
        """
        return self.net(x)

class ODELSTMCell(nn.Module):
    """ODE-LSTM单元，结合LSTM和ODE的特性"""
    def __init__(self, input_dim, hidden_dim):
        super(ODELSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM门控
        self.fc = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
        # ODE函数
        self.ode_func = ODEFunc(hidden_dim)
        
        # 初始化权重
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden_state, time_interval):
        """
        Args:
            x: 输入特征 [batch_size, input_dim]
            hidden_state: (h, c) 隐藏状态和细胞状态
            time_interval: 时间间隔
        Returns:
            更新后的隐藏状态 (h_new, c_new)
        """
        h_prev, c_prev = hidden_state
        
        # 计算LSTM门控
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.fc(combined)
        i, f, o, c_tilde = torch.chunk(gates, 4, dim=1)
        
        # 激活函数
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c_tilde = torch.tanh(c_tilde)
        
        # 更新细胞状态
        c_new = f * c_prev + i * c_tilde
        
        # 使用ODE演化隐藏状态
        if time_interval > 0:
            # ODE求解
            t = torch.tensor([0, time_interval], device=x.device)
            h_ode = odeint(self.ode_func, c_new, t, method='dopri5')[1]
            h_new = o * torch.tanh(h_ode)
        else:
            h_new = o * torch.tanh(c_new)
            
        return h_new, c_new

class EventEncoder(nn.Module):
    """事件编码器，将用户行为事件编码为特征向量"""
    def __init__(self, behavior_type_dim, product_feat_dim, hidden_dim):
        super(EventEncoder, self).__init__()
        self.behavior_embedding = nn.Embedding(behavior_type_dim, hidden_dim // 4)
        self.product_projection = nn.Linear(product_feat_dim, hidden_dim // 4)
        
        # 其他事件特征的投影
        self.event_feature_projection = nn.Linear(4, hidden_dim // 4)  # duration, interaction_intensity, event_index, session_duration
        
        # 合并所有特征
        self.combined_projection = nn.Linear(hidden_dim // 4 * 3, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, behavior_types, product_features, durations, 
                interaction_intensities, event_indices, session_durations):
        """
        Args:
            behavior_types: 行为类型 [batch_size, seq_len]
            product_features: 产品特征 [batch_size, seq_len, product_feat_dim]
            durations: 观看时长 [batch_size, seq_len]
            interaction_intensities: 交互强度 [batch_size, seq_len]
            event_indices: 事件索引 [batch_size, seq_len]
            session_durations: 会话时长 [batch_size, seq_len]
        Returns:
            编码后的事件特征 [batch_size, seq_len, hidden_dim]
        """
        # 行为类型嵌入
        behavior_emb = self.behavior_embedding(behavior_types)  # [batch_size, seq_len, hidden_dim//4]
        
        # 产品特征投影
        product_emb = self.product_projection(product_features)  # [batch_size, seq_len, hidden_dim//4]
        
        # 其他事件特征
        event_features = torch.stack([
            durations, 
            interaction_intensities,
            event_indices.float(),
            session_durations
        ], dim=-1)  # [batch_size, seq_len, 4]
        
        event_emb = self.event_feature_projection(event_features)  # [batch_size, seq_len, hidden_dim//4]
        
        # 合并所有特征
        combined = torch.cat([behavior_emb, product_emb, event_emb], dim=-1)
        event_encoding = self.layer_norm(F.relu(self.combined_projection(combined)))
        
        return event_encoding

class UserRoomEncoder(nn.Module):
    """用户和直播间特征编码器"""
    def __init__(self, user_feat_dim, room_feat_dim, hidden_dim):
        super(UserRoomEncoder, self).__init__()
        self.user_projection = nn.Sequential(
            nn.Linear(user_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.room_projection = nn.Sequential(
            nn.Linear(room_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.combined_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, user_features, room_features):
        """
        Args:
            user_features: 用户特征 [batch_size, user_feat_dim]
            room_features: 直播间特征 [batch_size, room_feat_dim]
        Returns:
            合并后的用户-直播间特征 [batch_size, hidden_dim]
        """
        user_emb = self.user_projection(user_features)
        room_emb = self.room_projection(room_features)
        
        combined = torch.cat([user_emb, room_emb], dim=-1)
        return self.combined_projection(combined)

class LHawkesModel(nn.Module):
    """Live Broadcast Neural Hawkes (LHawkes) 模型"""
    def __init__(self, config):
        super(LHawkesModel, self).__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.num_behavior_types = config['num_behavior_types']
        
        # 编码器
        self.event_encoder = EventEncoder(
            behavior_type_dim=config['num_behavior_types'],
            product_feat_dim=config['product_feat_dim'],
            hidden_dim=hidden_dim
        )
        
        self.user_room_encoder = UserRoomEncoder(
            user_feat_dim=config['user_feat_dim'],
            room_feat_dim=config['room_feat_dim'],
            hidden_dim=hidden_dim
        )
        
        # ODE-LSTM单元
        self.ode_lstm_cell = ODELSTMCell(
            input_dim=hidden_dim,  # 事件特征维度
            hidden_dim=hidden_dim
        )
        
        # 事件级解码器 - 预测下一个事件类型和时间
        self.event_type_decoder = nn.Linear(hidden_dim, config['num_behavior_types'])
        self.event_time_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 直播间级解码器 - 预测质量等级和取消收藏
        self.quality_level_decoder = nn.Linear(hidden_dim, config['num_quality_levels'])
        self.bookmark_removal_decoder = nn.Linear(hidden_dim, 1)
        
        # 用于Hawkes过程的参数
        self.alpha = nn.Parameter(torch.randn(config['num_behavior_types'], config['num_behavior_types']))
        self.beta = nn.Parameter(torch.randn(config['num_behavior_types']))
        
        # 温度参数用于控制强度函数的曲率
        self.s = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, batch):
        """
        Args:
            batch: 批处理数据
        Returns:
            模型输出，包括各种预测结果和损失
        """
        # 解包批处理数据
        user_features = batch['user_features']
        room_features = batch['room_features']
        behavior_types = batch['behavior_types']
        timestamps = batch['timestamps']
        product_features = batch['product_features']
        durations = batch['durations']
        interaction_intensities = batch['interaction_intensities']
        event_indices = batch['event_indices']
        session_durations = batch['session_durations']
        seq_lengths = batch['seq_lengths']
        quality_levels = batch['quality_levels']
        has_removals = batch['has_removals']
        
        batch_size, max_seq_len = behavior_types.shape
        
        # 编码用户和直播间特征
        user_room_emb = self.user_room_encoder(user_features, room_features)  # [batch_size, hidden_dim]
        
        # 编码事件特征
        event_encodings = self.event_encoder(
            behavior_types, 
            product_features, 
            durations, 
            interaction_intensities, 
            event_indices, 
            session_durations
        )  # [batch_size, max_seq_len, hidden_dim]
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_dim, device=user_features.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=user_features.device)
        
        # 添加用户-直播间嵌入到初始隐藏状态
        h = h + user_room_emb
        
        # 存储所有时间步的隐藏状态
        hidden_states = []
        event_type_logits = []
        event_time_preds = []
        
        # 处理序列
        for t in range(max_seq_len):
            # 只处理有效序列长度内的时间步
            mask = (t < seq_lengths).float().unsqueeze(1)  # [batch_size, 1]
            
            # 获取当前时间步的事件特征
            event_feat = event_encodings[:, t, :]  # [batch_size, hidden_dim]
            
            # 计算时间间隔（与上一个事件的时间差）
            if t == 0:
                time_interval = torch.zeros(batch_size, device=user_features.device)
            else:
                time_interval = timestamps[:, t] - timestamps[:, t-1]
            
            # ODE-LSTM前向传播
            h_new, c_new = self.ode_lstm_cell(event_feat, (h, c), time_interval)
            
            # 应用掩码，只更新有效序列
            h = h * (1 - mask) + h_new * mask
            c = c * (1 - mask) + c_new * mask
            
            # 记录隐藏状态
            hidden_states.append(h.unsqueeze(1))  # [batch_size, 1, hidden_dim]
            
            # 预测下一个事件类型和时间
            type_logit = self.event_type_decoder(h)  # [batch_size, num_behavior_types]
            time_pred = self.event_time_decoder(h).squeeze(1)  # [batch_size]
            
            event_type_logits.append(type_logit.unsqueeze(1))  # [batch_size, 1, num_behavior_types]
            event_time_preds.append(time_pred.unsqueeze(1))  # [batch_size, 1]
        
        # 堆叠结果
        hidden_states = torch.cat(hidden_states, dim=1)  # [batch_size, max_seq_len, hidden_dim]
        event_type_logits = torch.cat(event_type_logits, dim=1)  # [batch_size, max_seq_len, num_behavior_types]
        event_time_preds = torch.cat(event_time_preds, dim=1)  # [batch_size, max_seq_len]
        
        # 计算Hawkes过程强度函数
        hawkes_intensity = self._hawkes_intensity(behavior_types, timestamps, seq_lengths)
        
        # 直播间级特征 - 使用最后一个有效时间步的隐藏状态
        # 构建索引
        batch_idx = torch.arange(batch_size, device=user_features.device)
        last_idx = seq_lengths - 1
        room_level_features = hidden_states[batch_idx, last_idx, :]  # [batch_size, hidden_dim]
        
        # 预测质量等级
        quality_logits = self.quality_level_decoder(room_level_features)  # [batch_size, num_quality_levels]
        
        # 预测是否会取消收藏
        removal_logits = self.bookmark_removal_decoder(room_level_features).squeeze(1)  # [batch_size]
        
        # 计算损失
        loss = self._compute_loss(
            event_type_logits, behavior_types,
            event_time_preds, timestamps,
            hawkes_intensity,
            quality_logits, quality_levels,
            removal_logits, has_removals,
            seq_lengths
        )
        
        return {
            'loss': loss,
            'event_type_preds': torch.argmax(event_type_logits, dim=-1),
            'event_time_preds': event_time_preds,
            'quality_preds': torch.argmax(quality_logits, dim=-1),
            'removal_preds': torch.sigmoid(removal_logits) > 0.5,
            'quality_logits': quality_logits,
            'removal_logits': removal_logits
        }
    
    def _hawkes_intensity(self, behavior_types, timestamps, seq_lengths):
        """计算Hawkes过程的强度函数"""
        batch_size, max_seq_len = behavior_types.shape
        intensity = torch.zeros(batch_size, max_seq_len, self.num_behavior_types, device=behavior_types.device)
        
        for i in range(batch_size):
            seq_len = seq_lengths[i]
            for t in range(seq_len):
                current_type = behavior_types[i, t]
                current_time = timestamps[i, t]
                
                # 基础强度
                base_intensity = F.softplus(self.beta)
                
                # 历史事件的影响
                for s in range(t):
                    past_type = behavior_types[i, s]
                    past_time = timestamps[i, s]
                    time_diff = current_time - past_time
                    
                    # 影响随时间衰减
                    influence = self.alpha[past_type, current_type] * torch.exp(-time_diff)
                    base_intensity += influence
                
                # 应用激活函数确保强度非负
                intensity[i, t] = self._activation(base_intensity)
        
        return intensity
    
    def _activation(self, x):
        """强度函数的激活函数，确保非负性并控制曲率"""
        return torch.log(1 + torch.exp(self.s * x)) / self.s
    
    def _compute_loss(self, event_type_logits, behavior_types, event_time_preds, timestamps,
                      hawkes_intensity, quality_logits, quality_levels, removal_logits, has_removals, seq_lengths):
        """计算总损失"""
        batch_size, max_seq_len = behavior_types.shape
        
        # 创建掩码，只考虑有效序列部分
        mask = torch.zeros(batch_size, max_seq_len, device=event_type_logits.device)
        for i in range(batch_size):
            mask[i, :seq_lengths[i]] = 1.0
        
        # 1. 事件类型预测损失 (忽略最后一个事件，因为没有下一个事件)
        type_loss = F.cross_entropy(
            event_type_logits[:, :-1].transpose(1, 2),  # [batch_size, num_types, max_seq_len-1]
            behavior_types[:, 1:],  # [batch_size, max_seq_len-1]
            reduction='none'
        )
        type_loss = (type_loss * mask[:, :-1]).sum() / mask[:, :-1].sum()
        
        # 2. 事件时间预测损失 (忽略最后一个事件)
        time_diff = timestamps[:, 1:] - timestamps[:, :-1]  # 实际时间差
        time_loss = F.mse_loss(
            event_time_preds[:, :-1] * mask[:, :-1], 
            time_diff * mask[:, :-1],
            reduction='sum'
        ) / mask[:, :-1].sum()
        
        # 3. Hawkes过程损失
        hawkes_loss = -torch.log(torch.clamp(hawkes_intensity, min=1e-8)).sum() / mask.sum()
        
        # 4. 质量等级预测损失
        quality_loss = F.cross_entropy(quality_logits, quality_levels)
        
        # 5. 取消收藏预测损失
        removal_loss = F.binary_cross_entropy_with_logits(removal_logits, has_removals)
        
        # 总损失 (带权重)
        total_loss = (
            self.config['type_loss_weight'] * type_loss +
            self.config['time_loss_weight'] * time_loss +
            self.config['hawkes_loss_weight'] * hawkes_loss +
            self.config['quality_loss_weight'] * quality_loss +
            self.config['removal_loss_weight'] * removal_loss
        )
        
        return total_loss
