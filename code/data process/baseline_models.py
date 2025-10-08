import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class LSTMModel(nn.Module):
    """标准LSTM模型，作为基线"""
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']
        
        # 编码器
        self.event_encoder = EventEncoder(
            behavior_type_dim=config['num_behavior_types'],
            product_feat_dim=config['product_feat_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        self.user_room_encoder = UserRoomEncoder(
            user_feat_dim=config['user_feat_dim'],
            room_feat_dim=config['room_feat_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        # 标准LSTM层
        self.lstm = nn.LSTM(
            input_size=config['hidden_dim'],
            hidden_size=config['hidden_dim'],
            batch_first=True
        )
        
        # 解码器
        self.event_type_decoder = nn.Linear(config['hidden_dim'], config['num_behavior_types'])
        self.event_time_decoder = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'] // 2, 1)
        )
        
        self.quality_level_decoder = nn.Linear(config['hidden_dim'], config['num_quality_levels'])
        self.bookmark_removal_decoder = nn.Linear(config['hidden_dim'], 1)
    
    def forward(self, batch):
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
        
        # 编码用户和直播间特征
        user_room_emb = self.user_room_encoder(user_features, room_features)
        
        # 编码事件特征
        event_encodings = self.event_encoder(
            behavior_types, 
            product_features, 
            durations, 
            interaction_intensities, 
            event_indices, 
            session_durations
        )
        
        # 处理变长序列
        packed_events = nn.utils.rnn.pack_padded_sequence(
            event_encodings, 
            seq_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=True
        )
        
        # LSTM前向传播
        packed_output, (h_n, c_n) = self.lstm(packed_events)
        
        # 解包输出
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 预测事件类型和时间
        event_type_logits = self.event_type_decoder(lstm_output)
        event_time_preds = self.event_time_decoder(lstm_output).squeeze(2)
        
        # 直播间级预测（使用最后一个时间步的隐藏状态）
        room_level_features = h_n.squeeze(0)
        quality_logits = self.quality_level_decoder(room_level_features)
        removal_logits = self.bookmark_removal_decoder(room_level_features).squeeze(1)
        
        # 计算损失
        loss = self._compute_loss(
            event_type_logits, behavior_types,
            event_time_preds, timestamps,
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
    
    def _compute_loss(self, event_type_logits, behavior_types, event_time_preds, timestamps,
                      quality_logits, quality_levels, removal_logits, has_removals, seq_lengths):
        """计算损失（与LHawkes模型类似，但没有Hawkes过程损失）"""
        batch_size, max_seq_len = behavior_types.shape
        
        # 创建掩码
        mask = torch.zeros(batch_size, max_seq_len, device=event_type_logits.device)
        for i in range(batch_size):
            mask[i, :seq_lengths[i]] = 1.0
        
        # 事件类型预测损失
        type_loss = F.cross_entropy(
            event_type_logits[:, :-1].transpose(1, 2),
            behavior_types[:, 1:],
            reduction='none'
        )
        type_loss = (type_loss * mask[:, :-1]).sum() / mask[:, :-1].sum()
        
        # 事件时间预测损失
        time_diff = timestamps[:, 1:] - timestamps[:, :-1]
        time_loss = F.mse_loss(
            event_time_preds[:, :-1] * mask[:, :-1],
            time_diff * mask[:, :-1],
            reduction='sum'
        ) / mask[:, :-1].sum()
        
        # 质量等级预测损失
        quality_loss = F.cross_entropy(quality_logits, quality_levels)
        
        # 取消收藏预测损失
        removal_loss = F.binary_cross_entropy_with_logits(removal_logits, has_removals)
        
        # 总损失
        total_loss = (
            self.config['type_loss_weight'] * type_loss +
            self.config['time_loss_weight'] * time_loss +
            self.config['quality_loss_weight'] * quality_loss +
            self.config['removal_loss_weight'] * removal_loss
        )
        
        return total_loss

class GRUModel(nn.Module):
    """GRU模型，作为另一个基线"""
    def __init__(self, config):
        super(GRUModel, self).__init__()
        self.config = config
        
        # 编码器（与LSTM模型相同）
        self.event_encoder = EventEncoder(
            behavior_type_dim=config['num_behavior_types'],
            product_feat_dim=config['product_feat_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        self.user_room_encoder = UserRoomEncoder(
            user_feat_dim=config['user_feat_dim'],
            room_feat_dim=config['room_feat_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        # GRU层
        self.gru = nn.GRU(
            input_size=config['hidden_dim'],
            hidden_size=config['hidden_dim'],
            batch_first=True
        )
        
        # 解码器（与LSTM模型相同）
        self.event_type_decoder = nn.Linear(config['hidden_dim'], config['num_behavior_types'])
        self.event_time_decoder = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'] // 2, 1)
        )
        
        self.quality_level_decoder = nn.Linear(config['hidden_dim'], config['num_quality_levels'])
        self.bookmark_removal_decoder = nn.Linear(config['hidden_dim'], 1)
    
    def forward(self, batch):
        # 解包批处理数据（与LSTM模型相同）
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
        
        # 编码特征（与LSTM模型相同）
        user_room_emb = self.user_room_encoder(user_features, room_features)
        event_encodings = self.event_encoder(
            behavior_types, 
            product_features, 
            durations, 
            interaction_intensities, 
            event_indices, 
            session_durations
        )
        
        # 处理变长序列
        packed_events = nn.utils.rnn.pack_padded_sequence(
            event_encodings, 
            seq_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=True
        )
        
        # GRU前向传播
        packed_output, h_n = self.gru(packed_events)
        
        # 解包输出
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 预测（与LSTM模型相同）
        event_type_logits = self.event_type_decoder(gru_output)
        event_time_preds = self.event_time_decoder(gru_output).squeeze(2)
        
        room_level_features = h_n.squeeze(0)
        quality_logits = self.quality_level_decoder(room_level_features)
        removal_logits = self.bookmark_removal_decoder(room_level_features).squeeze(1)
        
        # 计算损失（与LSTM模型相同）
        loss = self._compute_loss(
            event_type_logits, behavior_types,
            event_time_preds, timestamps,
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
    
    def _compute_loss(self, event_type_logits, behavior_types, event_time_preds, timestamps,
                      quality_logits, quality_levels, removal_logits, has_removals, seq_lengths):
        """与LSTM模型使用相同的损失计算"""
        batch_size, max_seq_len = behavior_types.shape
        
        mask = torch.zeros(batch_size, max_seq_len, device=event_type_logits.device)
        for i in range(batch_size):
            mask[i, :seq_lengths[i]] = 1.0
        
        # 事件类型预测损失
        type_loss = F.cross_entropy(
            event_type_logits[:, :-1].transpose(1, 2),
            behavior_types[:, 1:],
            reduction='none'
        )
        type_loss = (type_loss * mask[:, :-1]).sum() / mask[:, :-1].sum()
        
        # 事件时间预测损失
        time_diff = timestamps[:, 1:] - timestamps[:, :-1]
        time_loss = F.mse_loss(
            event_time_preds[:, :-1] * mask[:, :-1],
            time_diff * mask[:, :-1],
            reduction='sum'
        ) / mask[:, :-1].sum()
        
        # 质量等级和取消收藏预测损失
        quality_loss = F.cross_entropy(quality_logits, quality_levels)
        removal_loss = F.binary_cross_entropy_with_logits(removal_logits, has_removals)
        
        # 总损失
        total_loss = (
            self.config['type_loss_weight'] * type_loss +
            self.config['time_loss_weight'] * time_loss +
            self.config['quality_loss_weight'] * quality_loss +
            self.config['removal_loss_weight'] * removal_loss
        )
        
        return total_loss

class SKLearnBaseline:
    """基于scikit-learn的基线模型包装器"""
    def __init__(self, model_type='lr'):
        """
        Args:
            model_type: 模型类型 ('lr' 表示逻辑回归, 'rf' 表示随机森林)
        """
        if model_type == 'lr':
            self.model_quality = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.model_removal = LogisticRegression(max_iter=1000, class_weight='balanced')
        elif model_type == 'rf':
            self.model_quality = RandomForestClassifier(n_estimators=100, class_weight='balanced')
            self.model_removal = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.is_trained = False
    
    def extract_features(self, batch):
        """从批处理数据中提取特征用于sklearn模型"""
        # 对于传统机器学习模型，我们使用聚合特征
        user_features = batch['user_features'].cpu().numpy()
        room_features = batch['room_features'].cpu().numpy()
        seq_lengths = batch['seq_lengths'].cpu().numpy()
        behavior_types = batch['behavior_types'].cpu().numpy()
        interaction_intensities = batch['interaction_intensities'].cpu().numpy()
        
        # 聚合行为特征
        batch_size = user_features.shape[0]
        agg_features = []
        
        for i in range(batch_size):
            seq_len = seq_lengths[i]
            
            # 行为类型统计
            behavior_counts = np.bincount(behavior_types[i, :seq_len], minlength=4)
            behavior_ratio = behavior_counts / seq_len if seq_len > 0 else np.zeros(4)
            
            # 交互强度统计
            if seq_len > 0:
                int_mean = np.mean(interaction_intensities[i, :seq_len])
                int_std = np.std(interaction_intensities[i, :seq_len])
                int_max = np.max(interaction_intensities[i, :seq_len])
            else:
                int_mean = int_std = int_max = 0.0
            
            # 合并所有特征
            features = np.concatenate([
                user_features[i],
                room_features[i],
                [seq_len],
                behavior_counts,
                behavior_ratio,
                [int_mean, int_std, int_max]
            ])
            
            agg_features.append(features)
        
        return np.array(agg_features)
    
    def fit(self, train_loader):
        """训练模型"""
        all_features = []
        all_quality_labels = []
        all_removal_labels = []
        
        # 收集所有训练数据
        for batch in train_loader:
            features = self.extract_features(batch)
            quality_labels = batch['quality_levels'].cpu().numpy()
            removal_labels = batch['has_removals'].cpu().numpy()
            
            all_features.append(features)
            all_quality_labels.append(quality_labels)
            all_removal_labels.append(removal_labels)
        
        # 合并所有批次
        X = np.vstack(all_features)
        y_quality = np.concatenate(all_quality_labels)
        y_removal = np.concatenate(all_removal_labels)
        
        # 训练模型
        self.model_quality.fit(X, y_quality)
        self.model_removal.fit(X, y_removal)
        
        self.is_trained = True
    
    def predict(self, batch):
        """预测"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
        
        features = self.extract_features(batch)
        
        # 预测
        quality_preds = self.model_quality.predict(features)
        removal_preds = self.model_removal.predict(features)
        quality_proba = self.model_quality.predict_proba(features)
        
        # 转换为torch张量
        return {
            'quality_preds': torch.tensor(quality_preds),
            'removal_preds': torch.tensor(removal_preds),
            'quality_logits': torch.tensor(quality_proba)
        }

# 从model.py导入所需的编码器类
from model import EventEncoder, UserRoomEncoder
