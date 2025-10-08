import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score

class Trainer:
    """模型训练器"""
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器和学习率调度器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 日志和模型保存设置
        self.log_dir = config['log_dir']
        self.model_dir = config['model_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 记录最佳性能
        self.best_val_mcc = -float('inf')
        self.best_val_loss = float('inf')
        
        # 初始化日志
        self._init_logs()
    
    def _init_logs(self):
        """初始化日志文件"""
        self.train_log = {
            'loss': [],
            'quality_acc': [],
            'removal_acc': [],
            'removal_recall': [],
            'removal_f1': []
        }
        
        self.val_log = {
            'loss': [],
            'quality_acc': [],
            'quality_mcc': [],
            'removal_acc': [],
            'removal_recall': [],
            'removal_f1': []
        }
    
    def _update_logs(self, log, metrics):
        """更新日志"""
        for key, value in metrics.items():
            if key in log:
                log[key].append(value)
    
    def _save_logs(self):
        """保存日志到文件"""
        for log_name, log_data in [('train', self.train_log), ('val', self.val_log)]:
            with open(os.path.join(self.log_dir, f'{log_name}_log.txt'), 'w') as f:
                for epoch in range(len(log_data['loss'])):
                    f.write(f"Epoch {epoch+1}:\n")
                    for key, values in log_data.items():
                        f.write(f"  {key}: {values[epoch]:.4f}\n")
    
    def _compute_metrics(self, preds, targets):
        """计算评估指标"""
        # 质量等级预测指标
        quality_preds = preds['quality_preds'].cpu().numpy()
        quality_targets = targets['quality_levels'].cpu().numpy()
        
        quality_acc = accuracy_score(quality_targets, quality_preds)
        quality_mcc = matthews_corrcoef(quality_targets, quality_preds)
        
        # 取消收藏预测指标
        removal_preds = preds['removal_preds'].cpu().numpy()
        removal_targets = targets['has_removals'].cpu().numpy()
        
        removal_acc = accuracy_score(removal_targets, removal_preds)
        removal_recall = recall_score(removal_targets, removal_preds)
        removal_f1 = f1_score(removal_targets, removal_preds)
        
        return {
            'quality_acc': quality_acc,
            'quality_mcc': quality_mcc,
            'removal_acc': removal_acc,
            'removal_recall': removal_recall,
            'removal_f1': removal_f1
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = {
            'quality_preds': [],
            'removal_preds': []
        }
        all_targets = {
            'quality_levels': [],
            'has_removals': []
        }
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = outputs['loss']
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item() * batch['user_features'].size(0)
            
            # 收集预测结果和目标
            for key in all_preds:
                all_preds[key].append(outputs[key].cpu())
            for key in all_targets:
                all_targets[key].append(batch[key].cpu())
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        # 合并所有批次的预测结果
        for key in all_preds:
            all_preds[key] = torch.cat(all_preds[key], dim=0)
        for key in all_targets:
            all_targets[key] = torch.cat(all_targets[key], dim=0)
        
        # 计算指标
        metrics = self._compute_metrics(all_preds, all_targets)
        
        return {
            'loss': avg_loss,
            **metrics
        }
    
    def evaluate(self, data_loader, split='val'):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds = {
            'quality_preds': [],
            'removal_preds': [],
            'quality_logits': []
        }
        all_targets = {
            'quality_levels': [],
            'has_removals': []
        }
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f'Evaluating {split}')
            for batch in progress_bar:
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # 累积损失
                total_loss += loss.item() * batch['user_features'].size(0)
                
                # 收集预测结果和目标
                for key in all_preds:
                    if key in outputs:
                        all_preds[key].append(outputs[key].cpu())
                for key in all_targets:
                    all_targets[key].append(batch[key].cpu())
                
                # 更新进度条
                progress_bar.set_postfix(loss=loss.item())
        
        # 计算平均损失
        avg_loss = total_loss / len(data_loader.dataset)
        
        # 合并所有批次的预测结果
        for key in all_preds:
            if all_preds[key]:
                all_preds[key] = torch.cat(all_preds[key], dim=0)
        for key in all_targets:
            all_targets[key] = torch.cat(all_targets[key], dim=0)
        
        # 计算指标
        metrics = self._compute_metrics(all_preds, all_targets)
        
        return {
            'loss': avg_loss,** metrics,
            'preds': all_preds,
            'targets': all_targets
        }
    
    def train(self, num_epochs):
        """训练模型"""
        print(f"开始训练，使用设备: {self.device}")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch()
            self._update_logs(self.train_log, train_metrics)
            
            # 打印训练结果
            print(f"训练结果 - 损失: {train_metrics['loss']:.4f}, "
                  f"质量准确率: {train_metrics['quality_acc']:.4f}, "
                  f"取消收藏F1: {train_metrics['removal_f1']:.4f}")
            
            # 验证
            val_metrics = self.evaluate(self.val_loader, split='val')
            self._update_logs(self.val_log, val_metrics)
            
            # 打印验证结果
            print(f"验证结果 - 损失: {val_metrics['loss']:.4f}, "
                  f"质量准确率: {val_metrics['quality_acc']:.4f}, "
                  f"质量MCC: {val_metrics['quality_mcc']:.4f}, "
                  f"取消收藏F1: {val_metrics['removal_f1']:.4f}")
            
            # 更新学习率调度器
            self.scheduler.step(val_metrics['loss'])
            
            # 保存最佳模型
            if val_metrics['quality_mcc'] > self.best_val_mcc:
                self.best_val_mcc = val_metrics['quality_mcc']
                self.best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_mcc': self.best_val_mcc,
                }, os.path.join(self.model_dir, 'best_model.pth'))
                print(f"保存新的最佳模型 (MCC: {self.best_val_mcc:.4f})")
        
        # 训练结束后保存日志
        self._save_logs()
        
        # 计算总训练时间
        total_time = time.time() - start_time
        print(f"\n训练完成，总时间: {total_time:.2f}秒")
        print(f"最佳验证MCC: {self.best_val_mcc:.4f}")
        
        # 加载最佳模型并在测试集上评估
        print("\n在测试集上评估最佳模型...")
        checkpoint = torch.load(os.path.join(self.model_dir, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = self.evaluate(self.test_loader, split='test')
        
        print(f"测试结果 - 损失: {test_results['loss']:.4f}, "
              f"质量准确率: {test_results['quality_acc']:.4f}, "
              f"质量MCC: {test_results['quality_mcc']:.4f}, "
              f"取消收藏准确率: {test_results['removal_acc']:.4f}, "
              f"取消收藏召回率: {test_results['removal_recall']:.4f}, "
              f"取消收藏F1: {test_results['removal_f1']:.4f}")
        
        return test_results

class BaselineTrainer:
    """基线模型训练器"""
    def __init__(self, model, train_loader, val_loader, test_loader, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        
        # 日志设置
        self.log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.is_trained = False
    
    def evaluate(self, data_loader, split='test'):
        """评估基线模型"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用train方法")
        
        all_preds = {
            'quality_preds': [],
            'removal_preds': []
        }
        all_targets = {
            'quality_levels': [],
            'has_removals': []
        }
        
        for batch in tqdm(data_loader, desc=f'Evaluating {split}'):
            # 预测
            outputs = self.model.predict(batch)
            
            # 收集预测结果和目标
            all_preds['quality_preds'].append(outputs['quality_preds'])
            all_preds['removal_preds'].append(outputs['removal_preds'])
            
            all_targets['quality_levels'].append(batch['quality_levels'].cpu())
            all_targets['has_removals'].append(batch['has_removals'].cpu())
        
        # 合并所有批次的预测结果
        for key in all_preds:
            all_preds[key] = torch.cat(all_preds[key], dim=0)
        for key in all_targets:
            all_targets[key] = torch.cat(all_targets[key], dim=0)
        
        # 计算指标
        quality_preds = all_preds['quality_preds'].numpy()
        quality_targets = all_targets['quality_levels'].numpy()
        
        quality_acc = accuracy_score(quality_targets, quality_preds)
        quality_mcc = matthews_corrcoef(quality_targets, quality_preds)
        
        removal_preds = all_preds['removal_preds'].numpy()
        removal_targets = all_targets['has_removals'].numpy()
        
        removal_acc = accuracy_score(removal_targets, removal_preds)
        removal_recall = recall_score(removal_targets, removal_preds)
        removal_f1 = f1_score(removal_targets, removal_preds)
        
        metrics = {
            'quality_acc': quality_acc,
            'quality_mcc': quality_mcc,
            'removal_acc': removal_acc,
            'removal_recall': removal_recall,
            'removal_f1': removal_f1,
            'preds': all_preds,
            'targets': all_targets
        }
        
        print(f"{split}结果 - "
              f"质量准确率: {quality_acc:.4f}, "
              f"质量MCC: {quality_mcc:.4f}, "
              f"取消收藏准确率: {removal_acc:.4f}, "
              f"取消收藏召回率: {removal_recall:.4f}, "
              f"取消收藏F1: {removal_f1:.4f}")
        
        return metrics
    
    def train(self):
        """训练基线模型"""
        print("开始训练基线模型...")
        start_time = time.time()
        
        self.model.fit(self.train_loader)
        self.is_trained = True
        
        total_time = time.time() - start_time
        print(f"训练完成，总时间: {total_time:.2f}秒")
        
        # 在验证集上评估
        print("\n在验证集上评估...")
        val_metrics = self.evaluate(self.val_loader, split='val')
        
        # 在测试集上评估
        print("\n在测试集上评估...")
        test_metrics = self.evaluate(self.test_loader, split='test')
        
        # 保存评估结果
        with open(os.path.join(self.log_dir, 'baseline_results.txt'), 'w') as f:
            f.write("验证集结果:\n")
            for key, value in val_metrics.items():
                if key not in ['preds', 'targets']:
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n测试集结果:\n")
            for key, value in test_metrics.items():
                if key not in ['preds', 'targets']:
                    f.write(f"  {key}: {value:.4f}\n")
        
        return test_metrics
