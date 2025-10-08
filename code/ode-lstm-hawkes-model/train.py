import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import LHawkesModel
from data_processing import load_and_preprocess_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def train_model(hidden_dim=64, batch_size=128, learning_rate=0.001, 
                num_epochs=100, patience=10, save_path='models'):
    """
    训练ODE-LSTM Hawkes模型
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取数据参数
    input_dim = 16  # 事件类型嵌入维度
    num_event_types = len(torch.unique(data['train']['next_event_type']))
    num_quality_levels = len(torch.unique(data['train']['quality_level']))
    
    # 创建模型
    model = LHawkesModel(input_dim, hidden_dim, num_event_types, num_quality_levels)
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 创建保存模型的目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_event_loss': [],
        'val_event_loss': [],
        'train_room_loss': [],
        'val_room_loss': []
    }
    
    # 早停机制
    best_val_loss = float('inf')
    counter = 0
    
    # 训练模型
    print(f"开始训练，隐藏维度: {hidden_dim}, 批次大小: {batch_size}, 学习率: {learning_rate}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0
        train_event_loss = 0.0
        train_room_loss = 0.0
        
        for batch in train_loader:
            # 准备批次数据
            events, timestamps, next_event_type, next_event_time, quality_level = batch
            events = events.to(device)
            timestamps = timestamps.to(device)
            next_event_type = next_event_type.to(device)
            next_event_time = next_event_time.to(device)
            quality_level = quality_level.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(events, timestamps)
            
            # 准备目标
            targets = {
                'next_event_type': next_event_type,
                'next_event_time': next_event_time,
                'quality_level': quality_level
            }
            
            # 计算损失
            losses = model.loss_function(outputs, targets)
            
            # 反向传播和优化
            losses['total_loss'].backward()
            optimizer.step()
            
            # 累加损失
            train_total_loss += losses['total_loss'].item() * events.size(0)
            train_event_loss += losses['event_level_loss'].item() * events.size(0)
            train_room_loss += losses['room_level_loss'].item() * events.size(0)
        
        # 计算平均训练损失
        train_total_loss /= len(train_loader.dataset)
        train_event_loss /= len(train_loader.dataset)
        train_room_loss /= len(train_loader.dataset)
        
        # 在验证集上评估
        model.eval()
        val_total_loss = 0.0
        val_event_loss = 0.0
        val_room_loss = 0.0
        
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
                
                val_total_loss += losses['total_loss'].item() * events.size(0)
                val_event_loss += losses['event_level_loss'].item() * events.size(0)
                val_room_loss += losses['room_level_loss'].item() * events.size(0)
        
        # 计算平均验证损失
        val_total_loss /= len(val_loader.dataset)
        val_event_loss /= len(val_loader.dataset)
        val_room_loss /= len(val_loader.dataset)
        
        # 记录损失
        history['train_loss'].append(train_total_loss)
        history['val_loss'].append(val_total_loss)
        history['train_event_loss'].append(train_event_loss)
        history['val_event_loss'].append(val_event_loss)
        history['train_room_loss'].append(train_room_loss)
        history['val_room_loss'].append(val_room_loss)
        
        # 打印训练进度
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_total_loss:.4f}, "
                  f"Val Loss: {val_total_loss:.4f}, "
                  f"Event Loss: {val_event_loss:.4f}, "
                  f"Room Loss: {val_room_loss:.4f}")
        
        # 早停检查
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'hidden_dim': hidden_dim,
                'num_event_types': num_event_types,
                'num_quality_levels': num_quality_levels
            }, f"{save_path}/best_model.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停于第 {epoch+1} 轮")
                break
    
    # 计算训练时间
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # 转换为分钟
    print(f"训练完成，总耗时: {training_time:.2f} 分钟")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    # 总损失
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 事件级损失
    plt.subplot(1, 3, 2)
    plt.plot(history['train_event_loss'], label='Train')
    plt.plot(history['val_event_loss'], label='Validation')
    plt.title('Event Level Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 直播间级损失
    plt.subplot(1, 3, 3)
    plt.plot(history['train_room_loss'], label='Train')
    plt.plot(history['val_room_loss'], label='Validation')
    plt.title('Room Level Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_losses.png")
    plt.close()
    
    # 保存训练历史
    np.save(f"{save_path}/training_history.npy", history)
    
    return model, history

if __name__ == "__main__":
    import argparse
    import time
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练ODE-LSTM Hawkes模型')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--save_path', type=str, default='models', help='模型保存路径')
    
    args = parser.parse_args()
    
    # 训练模型
    model, history = train_model(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        save_path=args.save_path
    )
