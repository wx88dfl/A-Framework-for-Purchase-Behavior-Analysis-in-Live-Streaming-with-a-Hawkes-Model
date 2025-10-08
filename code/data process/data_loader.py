import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import re
from datetime import datetime

class LiveStreamDataset(Dataset):
    def __init__(self, data_dir, split='train', temporal_split=True):
        """
        初始化直播电商数据集
        
        Args:
            data_dir: 数据文件夹路径
            split: 数据集划分 ('train', 'val', 'test')
            temporal_split: 是否按时间划分数据集
        """
        self.data_dir = data_dir
        self.split = split
        self.temporal_split = temporal_split
        
        # 加载所有CSV文件
        self.users = pd.read_csv(os.path.join(data_dir, 'users.csv'))
        self.live_rooms = pd.read_csv(os.path.join(data_dir, 'live_rooms.csv'))
        self.behaviors = pd.read_csv(os.path.join(data_dir, 'user_behavior_sequences.csv'))
        self.quality_labels = pd.read_csv(os.path.join(data_dir, 'quality_labels.csv'))
        self.products = pd.read_csv(os.path.join(data_dir, 'products.csv'))
        self.bookmark_removals = pd.read_csv(os.path.join(data_dir, 'bookmark_removals.csv'))
        
        # 预处理数据
        self._preprocess_data()
        
        # 划分数据集
        self._split_data()
        
        # 构建特征处理器
        self._build_feature_processors()
        
        # 训练Word2Vec模型用于文本特征
        self._train_word_embeddings()
        
        # 构建行为序列
        self._build_behavior_sequences()
    
    def _preprocess_data(self):
        """预处理所有数据，处理缺失值和格式转换"""
        # 转换日期时间格式
        self.users['registration_date'] = pd.to_datetime(self.users['registration_date'])
        self.live_rooms['Start_time'] = pd.to_datetime(self.live_rooms['Start_time'])
        self.live_rooms['end_time'] = pd.to_datetime(self.live_rooms['end_time'])
        self.behaviors['Timestamp'] = pd.to_datetime(self.behaviors['Timestamp'])
        
        # 处理缺失值
        self.users = self.users.fillna({
            'gender': 'unknown',
            'age': self.users['age'].median(),
            'city': 'unknown'
        })
        
        self.live_rooms = self.live_rooms.fillna({
            'Room_title': 'unknown',
            'Avg.rating': self.live_rooms['Avg.rating'].median(),
            'Historic_Success_rate': self.live_rooms['Historic_Success_rate'].median()
        })
        
        # 合并质量标签到直播间数据
        self.live_rooms = self.live_rooms.merge(
            self.quality_labels, 
            left_on='Room_id', 
            right_index=True,
            how='left'
        )
        
        # 为每个用户-直播间对标记是否有取消收藏行为
        removal_flags = self.bookmark_removals.groupby(['User_id', 'room_id']).size().reset_index(name='has_removal')
        self.behaviors = self.behaviors.merge(removal_flags, on=['User_id', 'room_id'], how='left')
        self.behaviors['has_removal'] = self.behaviors['has_removal'].fillna(0).astype(int)
    
    def _split_data(self):
        """划分训练集、验证集和测试集"""
        if self.temporal_split:
            # 按时间划分（论文中使用的方法）
            # 假设行为数据按时间排序
            self.behaviors = self.behaviors.sort_values('Timestamp')
            total = len(self.behaviors)
            train_idx = int(total * 0.7)
            val_idx = int(total * 0.85)
            
            if self.split == 'train':
                self.behaviors = self.behaviors.iloc[:train_idx]
            elif self.split == 'val':
                self.behaviors = self.behaviors.iloc[train_idx:val_idx]
            else:  # test
                self.behaviors = self.behaviors.iloc[val_idx:]
        else:
            # 按用户ID划分
            unique_users = self.behaviors['User_id'].unique()
            np.random.shuffle(unique_users)
            train_users = unique_users[:int(len(unique_users)*0.7)]
            val_users = unique_users[int(len(unique_users)*0.7):int(len(unique_users)*0.85)]
            test_users = unique_users[int(len(unique_users)*0.85):]
            
            if self.split == 'train':
                self.behaviors = self.behaviors[self.behaviors['User_id'].isin(train_users)]
            elif self.split == 'val':
                self.behaviors = self.behaviors[self.behaviors['User_id'].isin(val_users)]
            else:  # test
                self.behaviors = self.behaviors[self.behaviors['User_id'].isin(test_users)]
    
    def _build_feature_processors(self):
        """构建特征处理器，用于标准化和编码特征"""
        # 用户特征处理
        user_numerical_features = ['age', 'platform_experience_months']
        user_categorical_features = ['gender', 'user_tier', 'city']
        
        self.user_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), user_numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), user_categorical_features)
            ])
        
        # 拟合用户预处理模型
        self.user_preprocessor.fit(self.users[user_numerical_features + user_categorical_features])
        
        # 直播间特征处理
        room_numerical_features = ['Total products', 'Follower_count', 'Streamer_grade', 
                                  'Avg.rating', 'Historic_Success_rate']
        room_categorical_features = ['Category']
        
        self.room_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), room_numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), room_categorical_features)
            ])
        
        # 拟合直播间预处理模型
        self.room_preprocessor.fit(self.live_rooms[room_numerical_features + room_categorical_features])
        
        # 产品特征处理
        product_numerical_features = ['Price', 'Stock_quantity', 'Discount_rate', 
                                     'Product_rating', 'Sales-volume']
        product_categorical_features = ['Category', 'Is_Simited', 'Is_fot']
        
        self.product_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), product_numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), product_categorical_features)
            ])
        
        # 拟合产品预处理模型
        self.product_preprocessor.fit(self.products[product_numerical_features + product_categorical_features])
    
    def _train_word_embeddings(self, vector_size=100, window=5, min_count=1):
        """训练Word2Vec模型用于文本特征提取"""
        # 预处理文本数据
        def preprocess_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return text.split()
        
        # 收集所有文本数据（直播间标题和产品名称）
        room_titles = self.live_rooms['Room_title'].apply(preprocess_text).tolist()
        product_names = self.products['Productname'].apply(preprocess_text).tolist()
        all_texts = room_titles + product_names
        
        # 训练Word2Vec模型
        self.word2vec_model = Word2Vec(
            sentences=all_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        
        self.text_embedding_size = vector_size
    
    def _get_text_embedding(self, text):
        """获取文本的嵌入向量（平均词向量）"""
        words = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower()).split()
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        
        if len(vectors) == 0:
            return np.zeros(self.text_embedding_size)
        
        return np.mean(vectors, axis=0)
    
    def _build_behavior_sequences(self):
        """构建用户行为序列"""
        # 按用户和直播间分组，创建行为序列
        self.sequence_groups = self.behaviors.groupby(['User_id', 'room_id'])
        
        # 获取所有唯一的用户-直播间对
        self.user_room_pairs = list(self.sequence_groups.groups.keys())
        
        # 预计算每个序列的信息
        self.sequence_info = []
        for user_id, room_id in self.user_room_pairs:
            # 获取该用户在该直播间的所有行为
            sequence = self.sequence_groups.get_group((user_id, room_id)).sort_values('Timestamp')
            
            # 获取质量等级（如果存在）
            quality_level = self.live_rooms[self.live_rooms['Room_id'] == room_id]['Quality_Level'].values[0] \
                            if room_id in self.live_rooms['Room_id'].values else 0
            
            # 标记是否有取消收藏行为
            has_removal = 1 if ((self.bookmark_removals['User_id'] == user_id) & 
                               (self.bookmark_removals['room_id'] == room_id)).any() else 0
            
            self.sequence_info.append({
                'user_id': user_id,
                'room_id': room_id,
                'sequence_length': len(sequence),
                'quality_level': quality_level,
                'has_removal': has_removal
            })
    
    def _get_user_features(self, user_id):
        """获取用户特征向量"""
        user_data = self.users[self.users['user_id'] == user_id].iloc[0]
        user_features = self.user_preprocessor.transform(
            user_data[['age', 'platform_experience_months', 'gender', 'user_tier', 'city']].values.reshape(1, -1)
        ).flatten()
        return user_features
    
    def _get_room_features(self, room_id):
        """获取直播间特征向量（包括文本嵌入）"""
        room_data = self.live_rooms[self.live_rooms['Room_id'] == room_id].iloc[0]
        
        # 结构化特征
        structured_features = self.room_preprocessor.transform(
            room_data[['Total products', 'Follower_count', 'Streamer_grade', 
                      'Avg.rating', 'Historic_Success_rate', 'Category']].values.reshape(1, -1)
        ).flatten()
        
        # 文本嵌入特征
        title_embedding = self._get_text_embedding(room_data['Room_title'])
        
        # 合并特征
        return np.concatenate([structured_features, title_embedding])
    
    def _get_product_features(self, product_id):
        """获取产品特征向量（包括文本嵌入）"""
        if product_id not in self.products['Product_i'].values:
            return np.zeros(self.product_preprocessor.transformers_[0][1].n_features_in_ + 
                           self.product_preprocessor.transformers_[1][1].n_features_in_ +
                           self.text_embedding_size)
        
        product_data = self.products[self.products['Product_i'] == product_id].iloc[0]
        
        # 结构化特征
        structured_features = self.product_preprocessor.transform(
            product_data[['Price', 'Stock_quantity', 'Discount_rate', 
                         'Product_rating', 'Sales-volume', 'Category', 
                         'Is_Simited', 'Is_fot']].values.reshape(1, -1)
        ).flatten()
        
        # 文本嵌入特征
        name_embedding = self._get_text_embedding(product_data['Productname'])
        
        # 合并特征
        return np.concatenate([structured_features, name_embedding])
    
    def __len__(self):
        """返回序列数量"""
        return len(self.user_room_pairs)
    
    def __getitem__(self, idx):
        """获取指定索引的序列数据"""
        user_id, room_id = self.user_room_pairs[idx]
        sequence_data = self.sequence_groups.get_group((user_id, room_id)).sort_values('Timestamp')
        
        # 获取用户、直播间特征
        user_features = self._get_user_features(user_id)
        room_features = self._get_room_features(room_id)
        
        # 处理行为序列
        behavior_types = []
        timestamps = []
        product_features_list = []
        durations = []
        interaction_intensities = []
        event_indices = []
        session_durations = []
        
        # 转换时间为相对时间（相对于第一个事件）
        first_timestamp = sequence_data['Timestamp'].iloc[0].timestamp()
        
        for _, row in sequence_data.iterrows():
            behavior_types.append(row['Behavior_type'] - 1)  # 转为0-based索引
            timestamps.append(row['Timestamp'].timestamp() - first_timestamp)  # 相对时间
            durations.append(row['Duration'] if not pd.isna(row['Duration']) else 0)
            interaction_intensities.append(row['Interaction-intensity'])
            event_indices.append(row['Event_index_in_dession'])
            session_durations.append(row['Session duration'])
            
            # 获取产品特征
            if not pd.isna(row['Product_id']):
                product_features = self._get_product_features(row['Product_id'])
            else:
                # 如果没有产品ID，使用零向量
                product_features = np.zeros(self.product_preprocessor.transformers_[0][1].n_features_in_ + 
                                           self.product_preprocessor.transformers_[1][1].n_features_in_ +
                                           self.text_embedding_size)
            
            product_features_list.append(product_features)
        
        # 构建输入特征
        input_features = {
            'user_features': torch.FloatTensor(user_features),
            'room_features': torch.FloatTensor(room_features),
            'behavior_types': torch.LongTensor(behavior_types),
            'timestamps': torch.FloatTensor(timestamps),
            'product_features': torch.FloatTensor(product_features_list),
            'durations': torch.FloatTensor(durations),
            'interaction_intensities': torch.FloatTensor(interaction_intensities),
            'event_indices': torch.LongTensor(event_indices),
            'session_durations': torch.FloatTensor(session_durations),
            'sequence_length': len(behavior_types),
            'quality_level': torch.LongTensor([self.sequence_info[idx]['quality_level']]),
            'has_removal': torch.FloatTensor([self.sequence_info[idx]['has_removal']])
        }
        
        return input_features

def collate_fn(batch):
    """
    自定义批处理函数，处理变长序列
    
    Args:
        batch: 数据批次
        
    Returns:
        整理后的批处理数据
    """
    # 按序列长度排序（降序）
    batch.sort(key=lambda x: x['sequence_length'], reverse=True)
    
    # 提取各个特征
    user_features = torch.stack([item['user_features'] for item in batch])
    room_features = torch.stack([item['room_features'] for item in batch])
    quality_levels = torch.cat([item['quality_level'] for item in batch])
    has_removals = torch.cat([item['has_removal'] for item in batch])
    
    # 处理变长序列
    max_seq_len = max(item['sequence_length'] for item in batch)
    batch_size = len(batch)
    
    # 初始化张量
    behavior_types = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    timestamps = torch.zeros(batch_size, max_seq_len)
    durations = torch.zeros(batch_size, max_seq_len)
    interaction_intensities = torch.zeros(batch_size, max_seq_len)
    event_indices = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    session_durations = torch.zeros(batch_size, max_seq_len)
    
    # 产品特征的维度
    product_feat_dim = batch[0]['product_features'].shape[1]
    product_features = torch.zeros(batch_size, max_seq_len, product_feat_dim)
    
    # 序列长度
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # 填充数据
    for i, item in enumerate(batch):
        seq_len = item['sequence_length']
        behavior_types[i, :seq_len] = item['behavior_types']
        timestamps[i, :seq_len] = item['timestamps']
        product_features[i, :seq_len] = item['product_features']
        durations[i, :seq_len] = item['durations']
        interaction_intensities[i, :seq_len] = item['interaction_intensities']
        event_indices[i, :seq_len] = item['event_indices']
        session_durations[i, :seq_len] = item['session_durations']
        seq_lengths[i] = seq_len
    
    return {
        'user_features': user_features,
        'room_features': room_features,
        'behavior_types': behavior_types,
        'timestamps': timestamps,
        'product_features': product_features,
        'durations': durations,
        'interaction_intensities': interaction_intensities,
        'event_indices': event_indices,
        'session_durations': session_durations,
        'seq_lengths': seq_lengths,
        'quality_levels': quality_levels,
        'has_removals': has_removals
    }

def get_data_loaders(data_dir, batch_size=32, temporal_split=True):
    """
    获取训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据文件夹路径
        batch_size: 批处理大小
        temporal_split: 是否按时间划分数据集
        
    Returns:
        三个数据加载器 (train_loader, val_loader, test_loader)
    """
    train_dataset = LiveStreamDataset(data_dir, split='train', temporal_split=temporal_split)
    val_dataset = LiveStreamDataset(data_dir, split='val', temporal_split=temporal_split)
    test_dataset = LiveStreamDataset(data_dir, split='test', temporal_split=temporal_split)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader
