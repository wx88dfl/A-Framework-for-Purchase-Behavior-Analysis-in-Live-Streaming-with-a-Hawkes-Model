import argparse
import os
import torch
from data_processing import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model, compare_with_baselines
from ablation_study import run_ablation_study
from visualization import visualize_results_comparison, visualize_threshold_analysis

def main():
    """
    主程序入口，协调数据处理、模型训练、评估和可视化
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ODE-LSTM Hawkes模型实验主程序')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='模型保存/加载路径')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'train', 'evaluate', 'ablation', 'visualize'],
                       help='运行模式: all(默认) - 运行所有步骤, train - 仅训练, evaluate - 仅评估, '
                            'ablation - 仅运行消融实验, visualize - 仅可视化结果')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    for dir in ['data', 'models', 'results', 'ablation_results', 'visualizations']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    # 根据模式执行相应的操作
    if args.mode in ['all', 'train']:
        print("===== 开始训练模型 =====")
        model, history = train_model(
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            save_path='models'
        )
    
    if args.mode in ['all', 'evaluate']:
        print("\n===== 开始评估模型 =====")
        if not os.path.exists(args.model_path):
            print(f"警告: 未找到模型文件 {args.model_path}，将使用默认模型路径")
            args.model_path = 'models/best_model.pth'
        
        results, hidden_states, preds = evaluate_model(args.model_path)
        
        # 保存隐藏状态用于可视化
        np.save('results/hidden_states.npy', hidden_states)
        np.save('results/quality_labels.npy', preds['quality'][0])
        
        # 与基线模型比较
        compare_with_baselines(args.model_path)
    
    if args.mode in ['all', 'ablation']:
        print("\n===== 开始消融实验 =====")
        ablation_results = run_ablation_study(
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs
        )
    
    if args.mode in ['all', 'visualize']:
        print("\n===== 开始生成可视化结果 =====")
        visualize_threshold_analysis()
        visualize_results_comparison()
        
        # 如果有隐藏状态数据，生成t-SNE可视化
        if os.path.exists('results/hidden_states.npy') and os.path.exists('results/quality_labels.npy'):
            from visualization import visualize_hidden_states
            hidden_states = np.load('results/hidden_states.npy')
            quality_labels = np.load('results/quality_labels.npy')
            
            visualize_hidden_states(
                hidden_states,
                quality_labels,
                '按直播质量等级的隐藏状态t-SNE可视化',
                'quality_hidden_states_tsne.png'
            )
        
        print("可视化结果已保存到 visualizations 目录")
    
    print("\n===== 所有操作完成 =====")

if __name__ == "__main__":
    main()
