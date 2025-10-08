import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_synthetic_data(num_users=2000, num_livestreams=5000, num_records=12000, 
                           start_date='2020-08-01', end_date='2022-12-31'):
    """
    生成模拟的直播购物用户行为数据
    数据格式类似于论文中描述的真实数据集
    """
    # 设置随机种子确保可重复性
    np.random.seed(42)
    
    # 生成用户ID和直播间ID
    user_ids = np.random.randint(10000, 10000 + num_users, size=num_records)
    livestream_ids = np.random.randint(50000, 50000 + num_livestreams, size=num_records)
    
    # 生成时间戳（均匀分布在指定日期范围内）
    start_ts = pd.Timestamp(start_date).timestamp()
    end_ts = pd.Timestamp(end_date).timestamp()
    timestamps = np.random.uniform(start_ts, end_ts, size=num_records)
    timestamps = np.sort(timestamps)  # 按时间排序
    
    # 定义事件类型：0=观看, 1=下单, 2=收藏, 3=取消收藏
    event_types = np.random.choice([0, 1, 2, 3], size=num_records, 
                                  p=[0.7, 0.1, 0.15, 0.05])  # 观看行为最常见
    
    # 生成产品ID
    product_ids = np.random.randint(1, 500, size=num_records)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'livestream_id': livestream_ids,
        'event_type': event_types,
        'timestamp': timestamps,
        'product_id': product_ids
    })
    
    # 转换时间戳为datetime格式
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # 为每个用户-直播间对生成开始和结束时间
    df['start_time'] = df.groupby(['user_id', 'livestream_id'])['timestamp'].transform('min')
    df['end_time'] = df.groupby(['user_id', 'livestream_id'])['timestamp'].transform('max')
    
    # 计算观看总时长
    df['watch_duration'] = df['end_time'] - df['start_time']
    
    # 计算观看-下单时间比率 (用于质量评估)
    # 对每个用户-直播间对，找到第一个下单事件的时间
    first_order_time = df[df['event_type'] == 1].groupby(['user_id', 'livestream_id'])['timestamp'].min()
    # 合并回主DataFrame
    df = df.merge(first_order_time.rename('first_order_time'), 
                 on=['user_id', 'livestream_id'], how='left')
    # 计算从开始观看至首次下单的时间
    df['time_to_order'] = df['first_order_time'] - df['start_time']
    # 计算观看-下单比率 (观看时长 / 下单时间)
    df['watch_to_order_ratio'] = df['watch_duration'] / (df['time_to_order'] + 1e-8)  # 避免除零
    
    # 对没有下单行为的记录，设置一个较低的比率
    df.loc[df['event_type'] != 1, 'watch_to_order_ratio'] = np.random.uniform(0, 0.1, 
                                                                             size=sum(df['event_type'] != 1))
    
    # 按照论文3.2节的方法计算直播质量等级
    # 使用20th, 40th, 60th, 80th分位数作为阈值
    thresholds = df['watch_to_order_ratio'].quantile([0.2, 0.4, 0.6, 0.8]).values
    
    # 为每条记录分配质量等级
    def assign_quality_level(ratio):
        if ratio < thresholds[0]:
            return 0  # 低质量
        elif ratio < thresholds[1]:
            return 1  # 中等质量
        elif ratio < thresholds[2]:
            return 2  # 良好质量
        elif ratio < thresholds[3]:
            return 3  # 高质量
        else:
            return 4  # 极高质量
    
    df['quality_level'] = df['watch_to_order_ratio'].apply(assign_quality_level)
    
    # 保存数据
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/livestream_behavior_data.csv', index=False)
    
    print(f"生成的数据集包含 {len(df)} 条记录")
    print(f"质量等级阈值: {thresholds}")
    print(f"质量等级分布:\n{df['quality_level'].value_counts().sort_index()}")
    
    # 可视化质量等级分布
    plt.figure(figsize=(10, 6))
    sns.countplot(x='quality_level', data=df)
    plt.title('Distribution of Live Streaming Quality Levels')
    plt.xlabel('Quality Level')
    plt.ylabel('Count')
    plt.savefig('data/quality_level_distribution.png')
    plt.close()
    
    # 可视化watch_to_order_ratio分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['watch_to_order_ratio'], kde=True)
    for threshold in thresholds:
        plt.axvline(threshold, color='r', linestyle='--')
    plt.title('Distribution of Watch-to-Order Ratio with Thresholds')
    plt.xlabel('Watch-to-Order Ratio')
    plt.ylabel('Count')
    plt.savefig('data/watch_to_order_ratio_distribution.png')
    plt.close()
    
    return df

def create_sequences(df, max_seq_length=10):
    """
    将原始数据转换为模型输入的序列数据
    """
    # 按用户和直播间分组，创建行为序列
    sequences = []
    
    # 按用户ID和直播间ID分组
    for (user_id, livestream_id), group in df.groupby(['user_id', 'livestream_id']):
        # 按时间排序
        group = group.sort_values('timestamp')
        
        # 提取事件类型和时间戳
        event_types = group['event_type'].values
        timestamps = group['timestamp'].values
        quality_level = group['quality_level'].iloc[0]  # 同一直播间质量等级相同
        
        # 创建序列
        seq_len = len(event_types)
        if seq_len < 2:  # 需要至少两个事件才能预测下一个
            continue
            
        # 生成多个子序列（滑动窗口）
        for i in range(1, seq_len):
            start_idx = max(0, i - max_seq_length)
            end_idx = i
            
            # 当前序列
            sub_events = event_types[start_idx:end_idx]
            sub_timestamps = timestamps[start_idx:end_idx]
            
            # 目标：下一个事件
            next_event = event_types[i]
            next_event_time = timestamps[i] - timestamps[end_idx-1]  # 相对时间
            
            # 填充序列到固定长度
            if len(sub_events) < max_seq_length:
                pad_length = max_seq_length - len(sub_events)
                sub_events = np.pad(sub_events, (pad_length, 0), mode='constant')
                sub_timestamps = np.pad(sub_timestamps, (pad_length, 0), mode='constant')
            
            sequences.append({
                'user_id': user_id,
                'livestream_id': livestream_id,
                'event_sequence': sub_events,
                'timestamps': sub_timestamps,
                'next_event_type': next_event,
                'next_event_time': next_event_time,
                'quality_level': quality_level
            })
    
    # 转换为DataFrame
    seq_df = pd.DataFrame(sequences)
    print(f"生成的序列数据包含 {len(seq_df)} 条记录")
    
    # 保存序列数据
    seq_df.to_pickle('data/sequence_data.pkl')
    
    return seq_df

def prepare_data_for_model(seq_df, test_size=0.15, val_size=0.15):
    """
    准备模型训练、验证和测试数据
    """
    # 转换为PyTorch张量
    X_events = np.stack(seq_df['event_sequence'].values)
    X_timestamps = np.stack(seq_df['timestamps'].values)
    
    # 标准化时间戳（相对时间）
    X_timestamps = (X_timestamps - X_timestamps.mean()) / (X_timestamps.std() + 1e-8)
    
    # 目标变量
    y_event_type = seq_df['next_event_type'].values
    y_event_time = seq_df['next_event_time'].values
    y_quality = seq_df['quality_level'].values
    
    # 划分训练集、验证集和测试集
    # 先划分训练集和临时集
    X_events_train, X_events_temp, X_ts_train, X_ts_temp, \
    y_et_train, y_et_temp, y_etime_train, y_etime_temp, \
    y_q_train, y_q_temp = train_test_split(
        X_events, X_timestamps, y_event_type, y_event_time, y_quality,
        test_size=test_size + val_size, random_state=42, stratify=y_quality
    )
    
    # 从临时集中划分验证集和测试集
    test_ratio = test_size / (test_size + val_size)
    X_events_val, X_events_test, X_ts_val, X_ts_test, \
    y_et_val, y_et_test, y_etime_val, y_etime_test, \
    y_q_val, y_q_test = train_test_split(
        X_events_temp, X_ts_temp, y_et_temp, y_etime_temp, y_q_temp,
        test_size=test_ratio, random_state=42, stratify=y_q_temp
    )
    
    # 转换为PyTorch张量
    data = {
        'train': {
            'events': torch.LongTensor(X_events_train),
            'timestamps': torch.FloatTensor(X_ts_train),
            'next_event_type': torch.LongTensor(y_et_train),
            'next_event_time': torch.FloatTensor(y_etime_train),
            'quality_level': torch.LongTensor(y_q_train)
        },
        'val': {
            'events': torch.LongTensor(X_events_val),
            'timestamps': torch.FloatTensor(X_ts_val),
            'next_event_type': torch.LongTensor(y_et_val),
            'next_event_time': torch.FloatTensor(y_etime_val),
            'quality_level': torch.LongTensor(y_q_val)
        },
        'test': {
            'events': torch.LongTensor(X_events_test),
            'timestamps': torch.FloatTensor(X_ts_test),
            'next_event_type': torch.LongTensor(y_et_test),
            'next_event_time': torch.FloatTensor(y_etime_test),
            'quality_level': torch.LongTensor(y_q_test)
        }
    }
    
    # 保存处理好的数据
    torch.save(data, 'data/model_data.pt')
    
    print(f"训练集大小: {len(X_events_train)}")
    print(f"验证集大小: {len(X_events_val)}")
    print(f"测试集大小: {len(X_events_test)}")
    
    return data

def load_and_preprocess_data():
    """
    加载并预处理数据的完整流程
    """
    # 检查数据是否已存在
    if not os.path.exists('data/livestream_behavior_data.csv'):
        print("生成合成数据...")
        df = generate_synthetic_data()
    else:
        print("加载已存在的数据...")
        df = pd.read_csv('data/livestream_behavior_data.csv')
    
    # 检查序列数据是否已存在
    if not os.path.exists('data/sequence_data.pkl'):
        print("创建序列数据...")
        seq_df = create_sequences(df)
    else:
        print("加载已存在的序列数据...")
        seq_df = pd.read_pickle('data/sequence_data.pkl')
    
    # 检查模型数据是否已存在
    if not os.path.exists('data/model_data.pt'):
        print("准备模型数据...")
        data = prepare_data_for_model(seq_df)
    else:
        print("加载已存在的模型数据...")
        data = torch.load('data/model_data.pt')
    
    return data

if __name__ == "__main__":
    # 生成并预处理数据
    data = load_and_preprocess_data()
    print("数据准备完成!")
