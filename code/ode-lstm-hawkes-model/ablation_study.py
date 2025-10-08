import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score
import time

# 导入必要的模块
from model import LHawkesModel
from data_processing import load_and_preprocess_data
from train import train_model

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class ODELSTMAblation(LHawkesModel):
    """
    仅包含ODE-LSTM编码器的消融模型
    移除了事件级解码器中的Hawkes过程和直播间级解码器
    """
    def __init__(self, input_dim, hidden_dim, num_event_types, num_quality_levels):
        super().__init__(input_dim, hidden_dim, num_event_types, num_quality_levels)
        
    def loss_function(self, outputs, targets):
        """仅使用事件级损失"""
        # 事件类型预测损失
        event_type_target = targets['next_event_type']
        event_type_intensity = outputs['event_type_intensity']
        
        risk = torch.sum(event_type_intensity, dim=1)
        event_type_loss = -torch.mean(torch.log(event_type_intensity[torch.arange(event_type_intensity.shape[0]), 
                                                                    event_type_target]) - risk)
        
        # 事件时间预测损失
        event_time_target = targets['next_event_time']
        event_time_pred = outputs['event_time_pred']
        event_time_loss = nn.MSELoss()(event_time_pred, event_time_target)
        
        # 仅返回事件级损失
        total_loss = event_type_loss + event_time_loss
        
        return {
            'total_loss': total_loss,
            'event_type_loss': event_type_loss,
            'event_time_loss': event_time_loss,
            'event_level_loss': total_loss,
            'room_level_loss': torch.tensor(0.0, device=device)
        }

class EventDecoderAblation(LHawkesModel):
    """
    仅包含事件级解码器的消融模型
    使用标准LSTM代替ODE-LSTM编码器，移除直播间级解码器
    """
    def __init__(self, input_dim, hidden_dim, num_event_types, num_quality_levels):
        super().__init__(input_dim, hidden_dim, num_event_types, num_quality_levels)
        # 替换为标准LSTM
        self.encoder = nn.LSTMCell(input_dim, hidden_dim)
        
    def forward(self, event_sequences, timestamps):
        batch_size, seq_len = event_sequences.shape
        device = event_sequences.device
        
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_states = []
        
        for i in range(seq_len):
            event_type = event_sequences[:, i]
            event_emb = self.event_embedding(event_type)
            
            # 使用标准LSTM更新，忽略时间间隔
            h, c = self.encoder(event_emb, (h, c))
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        last_hidden = hidden_states[:, -1, :]
        
        event_type_logits = self.event_type_decoder(last_hidden)
        event_type_intensity = torch.log(1 + torch.exp(self.s * event_type_logits))
        event_time_pred = torch.exp(self.event_time_decoder(last_hidden))
        
        # 不计算质量预测
        quality_logits = torch.zeros(batch_size, self.num_quality_levels, device=device)
        
        return {
            'event_type_intensity': event_type_intensity,
            'event_time_pred': event_time_pred.squeeze(),
            'quality_logits': quality_logits,
            'hidden_states': hidden_states
        }
    
    def loss_function(self, outputs, targets):
        """仅使用事件级损失"""
        event_type_target = targets['next_event_type']
        event_type_intensity = outputs['event_type_intensity']
        
        risk = torch.sum(event_type_intensity, dim=1)
        event_type_loss = -torch.mean(torch.log(event_type_intensity[torch.arange(event_type_intensity.shape[0]), 
                                                                    event_type_target]) - risk)
        
        event_time_target = targets['next_event_time']
        event_time_pred = outputs['event_time_pred']
        event_time_loss = nn.MSELoss()(event_time_pred, event_time_target)
        
        total_loss = event_type_loss + event_time_loss
        
        return {
            'total_loss': total_loss,
            'event_type_loss': event_type_loss,
            'event_time_loss': event_time_loss,
            'event_level_loss': total_loss,
            'room_level_loss': torch.tensor(0.0, device=device)
        }

class RoomDecoderAblation(LHawkesModel):
    """
    仅包含直播间级解码器的消融模型
    使用标准LSTM代替ODE-LSTM编码器，移除事件级解码器
    """
    def __init__(self, input_dim, hidden_dim, num_event_types, num_quality_levels):
        super().__init__(input_dim, hidden_dim, num_event_types, num_quality_levels)
        # 替换为标准LSTM
        self.encoder = nn.LSTMCell(input_dim, hidden_dim)
        
    def forward(self, event_sequences, timestamps):
        batch_size, seq_len = event_sequences.shape
        device = event_sequences.device
        
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_states = []
        
        for i in range(seq_len):
            event_type = event_sequences[:, i]
            event_emb = self.event_embedding(event_type)
            
            # 使用标准LSTM更新，忽略时间间隔
            h, c = self.encoder(event_emb, (h, c))
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        room_level_feat = torch.mean(hidden_states, dim=1)
        quality_logits = self.room_quality_decoder(room_level_feat)
        
        # 不计算事件预测
        event_type_intensity = torch.zeros(batch_size, self.num_event_types, device=device)
        event_time_pred = torch.zeros(batch_size, device=device)
        
        return {
            'event_type_intensity': event_type_intensity,
            'event_time_pred': event_time_pred.squeeze(),
            'quality_logits': quality_logits,
            'hidden_states': hidden_states
        }
    
    def loss_function(self, outputs, targets):
        """仅使用直播间级损失"""
        quality_target = targets['quality_level']
        quality_logits = outputs['quality_logits']
        room_level_loss = nn.CrossEntropyLoss()(quality_logits, quality_target)
        
        return {
            'total_loss': room_level_loss,
            'event_type_loss': torch.tensor(0.0, device=device),
            'event_time_loss': torch.tensor(0.0, device=device),
            'event_level_loss': torch.tensor(0.0, device=device),
            'room_level_loss': room_level_loss
        }

def run_ablation_study(hidden_dim=64, batch_size=128, learning_rate=0.001, 
                      num_epochs=100, patience=10):
    """
    运行消融实验，比较不同模型组件的贡献
    """
    # 加载数据
    data = load_and_preprocess_data()
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        data['train']['events'],
        data['train']['timestamps'],
        data['train']['next_event_type'],
        data['train']['next_event_time'],
        data['train']['quality_level']
    )
    val_dataset = TensorDataset(
        data['val']['events'],
        data['val']['timestamps'],
        data['val']['next_event_type'],
        data['val']['next_event_time'],
        data['val']['quality_level']
    )
    test_dataset = TensorDataset(
        data['test']['events'],
        data['test']['timestamps'],
        data['test']['next_event_type'],
        data['test']['next_event_time'],
        data['test']['quality_level']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取数据参数
    input_dim = 16
    num_event_types = len(torch.unique(data['train']['next_event_type']))
    num_quality_levels = len(torch.unique(data['train']['quality_level']))
    
    # 定义要比较的模型
    models = {
        'Baseline (LSTM)': LHawkesModel(input_dim, hidden_dim, num_event_types, num_quality_levels),
        'A-only (ODE-LSTM)': ODELSTMAblation(input_dim, hidden_dim, num_event_types, num_quality_levels),
        'B-only (Event Decoder)': EventDecoderAblation(input_dim, hidden_dim, num_event_types, num_quality_levels),
        'C-only (Room Decoder)': RoomDecoderAblation(input_dim, hidden_dim, num_event_types, num_quality_levels),
        'Full Model': LHawkesModel(input_dim, hidden_dim, num_event_types, num_quality_levels)
    }
    
    # 创建保存结果的目录
    ablation_dir = 'ablation_results'
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)
    
    # 存储所有模型的结果
    all_results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\n===== 训练模型: {name} =====")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        best_val_loss = float('inf')
        counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                events, timestamps, next_event_type, next_event_time, quality_level = batch
                events = events.to(device)
                timestamps = timestamps.to(device)
                next_event_type = next_event_type.to(device)
                next_event_time = next_event_time.to(device)
                quality_level = quality_level.to(device)
                
                optimizer.zero_grad()
                outputs = model(events, timestamps)
                targets = {
                    'next_event_type': next_event_type,
                    'next_event_time': next_event_time,
                    'quality_level': quality_level
                }
                losses = model.loss_function(outputs, targets)
                losses['total_loss'].backward()
                optimizer.step()
                
                train_loss += losses['total_loss'].item() * events.size(0)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    events, timestamps, next_event_type, next_event_time, quality_level = batch
                    events = events.to(device)
                    timestamps = timestamps.to(device)
                    next_event_type = next_event_type.to(device)
                    next_event_time = next_event_time.to(device)
                    quality_level = quality_level.to(device)
                    
                    outputs = model(events, timestamps)
                    targets = {
                        'next_event_type': next_event_type,
                        'next_event_time': next_event_time,
                        'quality_level': quality_level
                    }
                    losses = model.loss_function(outputs, targets)
                    val_loss += losses['total_loss'].item() * events.size(0)
            
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{ablation_dir}/{name.replace(' ', '_')}_best.pth")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于第 {epoch+1} 轮")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        training_time = (time.time() - start_time) / 60
        print(f"{name} 训练完成，耗时: {training_time:.2f} 分钟")
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title(f'{name} 训练曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{ablation_dir}/{name.replace(' ', '_')}_loss.png")
        plt.close()
        
        # 在测试集上评估
        model.load_state_dict(torch.load(f"{ablation_dir}/{name.replace(' ', '_')}_best.pth", map_location=device))
        model.eval()
        
        all_event_type_preds = []
        all_event_type_true = []
        all_quality_preds = []
        all_quality_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                events, timestamps, next_event_type, _, quality_level = batch
                events = events.to(device)
                timestamps = timestamps.to(device)
                
                outputs = model(events, timestamps)
                event_type_preds = torch.argmax(outputs['event_type_intensity'], dim=1)
                quality_preds = torch.argmax(outputs['quality_logits'], dim=1)
                
                all_event_type_preds.extend(event_type_preds.cpu().numpy())
                all_event_type_true.extend(next_event_type.numpy())
                all_quality_preds.extend(quality_preds.cpu().numpy())
                all_quality_true.extend(quality_level.numpy())
        
        # 计算评估指标
        event_acc = accuracy_score(all_event_type_true, all_event_type_preds)
        event_f1 = f1_score(all_event_type_true, all_event_type_preds, average='macro')
        
        quality_acc = accuracy_score(all_quality_true, all_quality_preds)
        quality_mcc = matthews_corrcoef(all_quality_true, all_quality_preds)
        
        results = {
            'event_accuracy': event_acc,
            'event_f1': event_f1,
            'quality_accuracy': quality_acc,
            'quality_mcc': quality_mcc,
            'training_time': training_time
        }
        
        all_results[name] = results
        print(f"{name} 测试结果:")
        print(f"事件类型预测 - 准确率: {event_acc:.4f}, F1: {event_f1:.4f}")
        print(f"直播质量预测 - 准确率: {quality_acc:.4f}, MCC: {quality_mcc:.4f}")
    
    # 汇总并可视化消融实验结果
    models = list(all_results.keys())
    
    # 事件预测性能
    event_acc = [all_results[m]['event_accuracy'] for m in models]
    event_f1 = [all_results[m]['event_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, event_acc, width, label='准确率 (ACC)')
    plt.bar(x + width/2, event_f1, width, label='F1分数')
    
    plt.xlabel('模型变体')
    plt.ylabel('分数')
    plt.title('不同模型变体的事件类型预测性能')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ablation_dir}/ablation_event_results.png")
    plt.close()
    
    # 质量预测性能
    quality_acc = [all_results[m]['quality_accuracy'] for m in models]
    quality_mcc = [all_results[m]['quality_mcc'] for m in models]
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, quality_acc, width, label='准确率 (ACC)')
    plt.bar(x + width/2, quality_mcc, width, label='马修斯相关系数 (MCC)')
    
    plt.xlabel('模型变体')
    plt.ylabel('分数')
    plt.title('不同模型变体的直播质量预测性能')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ablation_dir}/ablation_quality_results.png")
    plt.close()
    
    # 训练时间比较
    training_time = [all_results[m]['training_time'] for m in models]
    
    plt.figure(figsize=(14, 6))
    plt.bar(models, training_time)
    plt.xlabel('模型变体')
    plt.ylabel('训练时间 (分钟)')
    plt.title('不同模型变体的训练时间比较')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{ablation_dir}/ablation_training_time.png")
    plt.close()
    
    # 保存结果为CSV
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(f"{ablation_dir}/ablation_results.csv")
    
    print("\n消融实验完成，结果已保存到 ablation_results 目录")
    return all_results

if __name__ == "__main__":
    import argparse
    import torch.optim as optim
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行ODE-LSTM Hawkes模型的消融实验')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 运行消融实验
    results = run_ablation_study(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        patience=args.patience
    )
