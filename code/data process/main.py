import torch
import numpy as np
import os
import argparse
import random
from data_loader import get_data_loaders
from model import LHawkesModel
from baseline_models import LSTMModel, GRUModel, SKLearnBaseline
from trainer import Trainer, BaselineTrainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

def set_seed(seed=42):
    """设置随机种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_config(args, data_loader):
    """获取模型配置参数"""
    # 获取特征维度信息（从数据加载器中）
    sample_batch = next(iter(data_loader))
    user_feat_dim = sample_batch['user_features'].shape[1]
    room_feat_dim = sample_batch['room_features'].shape[1]
    product_feat_dim = sample_batch['product_features'].shape[2]
    
    return {
        'hidden_dim': args.hidden_dim,
        'user_feat_dim': user_feat_dim,
        'room_feat_dim': room_feat_dim,
        'product_feat_dim': product_feat_dim,
        'num_behavior_types': 4,  # 1: Watch, 2: Order, 3: Favorite, 4: Cancel Favorite
        'num_quality_levels': 5,  # 0-4
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'type_loss_weight': 1.0,
        'time_loss_weight': 0.1,
        'hawkes_loss_weight': 0.5,
        'quality_loss_weight': 1.0,
        'removal_loss_weight': 1.0,
        'log_dir': os.path.join(args.log_dir, f"lh_{args.hidden_dim}"),
        'model_dir': os.path.join(args.model_dir, f"lh_{args.hidden_dim}")
    }

def run_ablation_studies(args, train_loader, val_loader, test_loader):
    """运行消融实验"""
    print("开始消融实验...")
    
    # 获取基础配置
    base_config = get_model_config(args, train_loader)
    
    # 消融实验配置 - 不同组件组合
    ablation_configs = [
        {
            'name': 'Baseline',
            'use_ode': False,
            'use_hawkes': False,
            'use_joint_optim': False
        },
        {
            'name': 'A-only (ODE-LSTM)',
            'use_ode': True,
            'use_hawkes': False,
            'use_joint_optim': False
        },
        {
            'name': 'B-only (Hawkes)',
            'use_ode': False,
            'use_hawkes': True,
            'use_joint_optim': False
        },
        {
            'name': 'C-only (Joint Optim)',
            'use_ode': False,
            'use_hawkes': False,
            'use_joint_optim': True
        },
        {
            'name': 'A+B',
            'use_ode': True,
            'use_hawkes': True,
            'use_joint_optim': False
        },
        {
            'name': 'A+C',
            'use_ode': True,
            'use_hawkes': False,
            'use_joint_optim': True
        },
        {
            'name': 'Full Model',
            'use_ode': True,
            'use_hawkes': True,
            'use_joint_optim': True
        }
    ]
    
    # 存储所有消融实验结果
    ablation_results = []
    
    for config in ablation_configs:
        print(f"\n运行消融实验: {config['name']}")
        print("-" * 50)
        
        # 修改配置
        model_config = base_config.copy()
        model_config['log_dir'] = os.path.join(args.log_dir, f"ablation_{config['name']}")
        model_config['model_dir'] = os.path.join(args.model_dir, f"ablation_{config['name']}")
        
        # 创建模型（这里需要修改LHawkesModel以支持消融配置）
        # 实际实现中，需要在LHawkesModel中添加相应的开关参数
        model = LHawkesModel(model_config)
        
        # 创建训练器并训练
        trainer = Trainer(model, train_loader, val_loader, test_loader, model_config)
        test_results = trainer.train(args.epochs)
        
        # 保存结果
        ablation_results.append({
            'name': config['name'],
            'quality_acc': test_results['quality_acc'],
            'quality_mcc': test_results['quality_mcc'],
            'removal_acc': test_results['removal_acc'],
            'removal_recall': test_results['removal_recall'],
            'removal_f1': test_results['removal_f1']
        })
    
    # 保存消融实验结果
    results_df = pd.DataFrame(ablation_results)
    results_df.to_csv(os.path.join(args.log_dir, 'ablation_results.csv'), index=False)
    
    # 绘制消融实验结果图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='name', y='quality_acc', data=results_df)
    plt.title('Quality Prediction Accuracy')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='name', y='quality_mcc', data=results_df)
    plt.title('Quality Prediction MCC')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(x='name', y='removal_recall', data=results_df)
    plt.title('Bookmark Removal Recall')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(x='name', y='removal_f1', data=results_df)
    plt.title('Bookmark Removal F1 Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'ablation_results.png'))
    plt.close()
    
    print("消融实验完成，结果已保存")

def visualize_results(model, test_loader, log_dir):
    """可视化模型结果，包括t-SNE降维和混淆矩阵"""
    print("生成结果可视化...")
    
    # 获取测试集预测结果
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_features = []
    all_quality_labels = []
    all_removal_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Extracting features'):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 获取用户-直播间特征（假设模型有这个方法）
            user_room_emb = model.user_room_encoder(
                batch['user_features'], 
                batch['room_features']
            )
            
            all_features.append(user_room_emb.cpu().numpy())
            all_quality_labels.append(batch['quality_levels'].cpu().numpy())
            all_removal_labels.append(batch['has_removals'].cpu().numpy())
    
    # 合并所有特征和标签
    features = np.vstack(all_features)
    quality_labels = np.concatenate(all_quality_labels)
    removal_labels = np.concatenate(all_removal_labels)
    
    # 使用t-SNE降维
    print("运行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 绘制质量等级的t-SNE可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=quality_labels, 
        cmap='viridis', 
        alpha=0.7,
        s=10
    )
    plt.colorbar(scatter, label='Quality Level')
    plt.title('t-SNE Visualization of User-Room Features (Colored by Quality Level)')
    plt.savefig(os.path.join(log_dir, 'tsne_quality.png'))
    plt.close()
    
    # 绘制取消收藏的t-SNE可视化
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=removal_labels, 
        cmap='coolwarm', 
        alpha=0.7,
        s=10
    )
    plt.colorbar(scatter, label='Has Bookmark Removal')
    plt.title('t-SNE Visualization of User-Room Features (Colored by Bookmark Removal)')
    plt.savefig(os.path.join(log_dir, 'tsne_removal.png'))
    plt.close()
    
    print("可视化完成，图像已保存")

def main(args):
    """主函数"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建必要的目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        temporal_split=args.temporal_split
    )
    
    # 根据模式运行不同的实验
    if args.mode == 'ablation':
        # 运行消融实验
        run_ablation_studies(args, train_loader, val_loader, test_loader)
    else:
        # 训练提出的模型
        print("训练提出的ODE-LSTM Hawkes模型...")
        model_config = get_model_config(args, train_loader)
        model = LHawkesModel(model_config)
        
        trainer = Trainer(model, train_loader, val_loader, test_loader, model_config)
        test_results = trainer.train(args.epochs)
        
        # 可视化结果
        visualize_results(model, test_loader, model_config['log_dir'])
        
        # 如果需要，运行基线模型进行比较
        if args.include_baselines:
            print("\n" + "="*50)
            print("运行基线模型进行比较...")
            
            # 1. LSTM模型
            print("\n训练LSTM模型...")
            lstm_config = model_config.copy()
            lstm_config['log_dir'] = os.path.join(args.log_dir, f"lstm_{args.hidden_dim}")
            lstm_config['model_dir'] = os.path.join(args.model_dir, f"lstm_{args.hidden_dim}")
            
            lstm_model = LSTMModel(lstm_config)
            lstm_trainer = Trainer(lstm_model, train_loader, val_loader, test_loader, lstm_config)
            lstm_test_results = lstm_trainer.train(args.epochs)
            
            # 2. GRU模型
            print("\n训练GRU模型...")
            gru_config = model_config.copy()
            gru_config['log_dir'] = os.path.join(args.log_dir, f"gru_{args.hidden_dim}")
            gru_config['model_dir'] = os.path.join(args.model_dir, f"gru_{args.hidden_dim}")
            
            gru_model = GRUModel(gru_config)
            gru_trainer = Trainer(gru_model, train_loader, val_loader, test_loader, gru_config)
            gru_test_results = gru_trainer.train(args.epochs)
            
            # 3. 逻辑回归模型
            print("\n训练逻辑回归模型...")
            lr_model = SKLearnBaseline(model_type='lr')
            lr_trainer = BaselineTrainer(lr_model, train_loader, val_loader, test_loader, {
                'log_dir': os.path.join(args.log_dir, 'lr')
            })
            lr_test_results = lr_trainer.train()
            
            # 4. 随机森林模型
            print("\n训练随机森林模型...")
            rf_model = SKLearnBaseline(model_type='rf')
            rf_trainer = BaselineTrainer(rf_model, train_loader, val_loader, test_loader, {
                'log_dir': os.path.join(args.log_dir, 'rf')
            })
            rf_test_results = rf_trainer.train()
            
            # 汇总所有模型结果并保存
            results = {
                'Proposed Model': test_results,
                'LSTM': lstm_test_results,
                'GRU': gru_test_results,
                'Logistic Regression': lr_test_results,
                'Random Forest': rf_test_results
            }
            
            # 保存比较结果
            comparison_data = []
            for model_name, result in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Quality Accuracy': result['quality_acc'],
                    'Quality MCC': result['quality_mcc'],
                    'Removal Accuracy': result['removal_acc'],
                    'Removal Recall': result['removal_recall'],
                    'Removal F1': result['removal_f1']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(os.path.join(args.result_dir, 'model_comparison.csv'), index=False)
            
            # 绘制比较结果图
            plt.figure(figsize=(18, 12))
            
            plt.subplot(2, 2, 1)
            sns.barplot(x='Model', y='Quality Accuracy', data=comparison_df)
            plt.title('Quality Prediction Accuracy Comparison')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            sns.barplot(x='Model', y='Quality MCC', data=comparison_df)
            plt.title('Quality Prediction MCC Comparison')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 3)
            sns.barplot(x='Model', y='Removal Recall', data=comparison_df)
            plt.title('Bookmark Removal Recall Comparison')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            sns.barplot(x='Model', y='Removal F1', data=comparison_df)
            plt.title('Bookmark Removal F1 Score Comparison')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.result_dir, 'model_comparison.png'))
            plt.close()
            
            print("模型比较完成，结果已保存")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ODE-LSTM Hawkes Model for Live Streaming E-commerce')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data', help='数据文件夹路径')
    parser.add_argument('--temporal_split', action='store_true', default=True, 
                        help='是否按时间划分数据集')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=120, help='训练轮数')
    
    # 实验参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ablation'],
                        help='运行模式：训练模型或运行消融实验')
    parser.add_argument('--include_baselines', action='store_true', default=False,
                        help='是否包括基线模型比较')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 输出参数
    parser.add_argument('--log_dir', type=str, default='logs', help='日志文件夹路径')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存文件夹路径')
    parser.add_argument('--result_dir', type=str, default='results', help='结果保存文件夹路径')
    
    args = parser.parse_args()
    
    # 运行主函数
    main(args)
