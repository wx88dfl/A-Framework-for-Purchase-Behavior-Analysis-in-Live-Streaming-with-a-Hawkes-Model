import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score, confusion_matrix
import os
from model import LHawkesModel
from data_processing import load_and_preprocess_data

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def evaluate_model(model_path='models/best_model.pth', data=None):
    """
    评估训练好的模型在测试集上的性能
    """
    # 加载数据
    if data is None:
        data = load_and_preprocess_data()
    
    # 创建测试集数据加载器
    test_dataset = TensorDataset(
        data['test']['events'],
        data['test']['timestamps'],
        data['test']['next_event_type'],
        data['test']['next_event_time'],
        data['test']['quality_level']
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = 16  # 与训练时保持一致
    
    model = LHawkesModel(
        input_dim,
        checkpoint['hidden_dim'],
        checkpoint['num_event_types'],
        checkpoint['num_quality_levels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 存储预测结果和真实标签
    all_event_type_preds = []
    all_event_type_true = []
    all_event_time_preds = []
    all_event_time_true = []
    all_quality_preds = []
    all_quality_true = []
    all_hidden_states = []
    
    # 在测试集上进行评估
    with torch.no_grad():
        for batch in test_loader:
            events, timestamps, next_event_type, next_event_time, quality_level = batch
            events = events.to(device)
            timestamps = timestamps.to(device)
            
            # 模型预测
            outputs = model(events, timestamps)
            
            # 存储结果
            event_type_preds = torch.argmax(outputs['event_type_intensity'], dim=1)
            quality_preds = torch.argmax(outputs['quality_logits'], dim=1)
            
            all_event_type_preds.extend(event_type_preds.cpu().numpy())
            all_event_type_true.extend(next_event_type.numpy())
            all_event_time_preds.extend(outputs['event_time_pred'].cpu().numpy())
            all_event_time_true.extend(next_event_time.numpy())
            all_quality_preds.extend(quality_preds.cpu().numpy())
            all_quality_true.extend(quality_level.numpy())
            all_hidden_states.append(outputs['hidden_states'].cpu().numpy())
    
    # 合并隐藏状态
    all_hidden_states = np.concatenate(all_hidden_states, axis=0)
    
    # 创建结果目录
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 1. 评估事件类型预测
    event_type_acc = accuracy_score(all_event_type_true, all_event_type_preds)
    event_type_recall = recall_score(all_event_type_true, all_event_type_preds, average='macro')
    event_type_f1 = f1_score(all_event_type_true, all_event_type_preds, average='macro')
    
    print("\n事件类型预测性能:")
    print(f"准确率 (ACC): {event_type_acc:.4f}")
    print(f"召回率 (Recall): {event_type_recall:.4f}")
    print(f"F1分数: {event_type_f1:.4f}")
    
    # 事件类型混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_event_type_true, all_event_type_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['观看', '下单', '收藏', '取消收藏'],
                yticklabels=['观看', '下单', '收藏', '取消收藏'])
    plt.xlabel('预测类型')
    plt.ylabel('真实类型')
    plt.title('事件类型预测混淆矩阵')
    plt.savefig(f"{results_dir}/event_type_confusion_matrix.png")
    plt.close()
    
    # 2. 评估事件时间预测
    event_time_mae = np.mean(np.abs(np.array(all_event_time_preds) - np.array(all_event_time_true)))
    event_time_mse = np.mean((np.array(all_event_time_preds) - np.array(all_event_time_true)) **2)
    event_time_rmse = np.sqrt(event_time_mse)
    
    print("\n事件时间预测性能:")
    print(f"平均绝对误差 (MAE): {event_time_mae:.4f}")
    print(f"均方误差 (MSE): {event_time_mse:.4f}")
    print(f"均方根误差 (RMSE): {event_time_rmse:.4f}")
    
    # 绘制时间预测散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(all_event_time_true, all_event_time_preds, alpha=0.5)
    plt.plot([min(all_event_time_true), max(all_event_time_true)], 
             [min(all_event_time_true), max(all_event_time_true)], 'r--')
    plt.xlabel('真实时间')
    plt.ylabel('预测时间')
    plt.title('事件时间预测散点图')
    plt.savefig(f"{results_dir}/event_time_prediction.png")
    plt.close()
    
    # 3. 评估直播质量预测
    quality_acc = accuracy_score(all_quality_true, all_quality_preds)
    quality_mcc = matthews_corrcoef(all_quality_true, all_quality_preds)
    quality_recall = recall_score(all_quality_true, all_quality_preds, average='macro')
    quality_f1 = f1_score(all_quality_true, all_quality_preds, average='macro')
    
    print("\n直播质量预测性能:")
    print(f"准确率 (ACC): {quality_acc:.4f}")
    print(f"马修斯相关系数 (MCC): {quality_mcc:.4f}")
    print(f"召回率 (Recall): {quality_recall:.4f}")
    print(f"F1分数: {quality_f1:.4f}")
    
    # 质量等级混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_quality_true, all_quality_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测质量等级')
    plt.ylabel('真实质量等级')
    plt.title('直播质量预测混淆矩阵')
    plt.savefig(f"{results_dir}/quality_level_confusion_matrix.png")
    plt.close()
    
    # 4. 保存所有评估结果
    results = {
        'event_type': {
            'accuracy': event_type_acc,
            'recall': event_type_recall,
            'f1': event_type_f1
        },
        'event_time': {
            'mae': event_time_mae,
            'mse': event_time_mse,
            'rmse': event_time_rmse
        },
        'quality_level': {
            'accuracy': quality_acc,
            'mcc': quality_mcc,
            'recall': quality_recall,
            'f1': quality_f1
        }
    }
    
    # 保存为CSV
    results_df = pd.DataFrame({
        '任务': ['事件类型预测', '事件类型预测', '事件类型预测',
                '事件时间预测', '事件时间预测', '事件时间预测',
                '直播质量预测', '直播质量预测', '直播质量预测', '直播质量预测'],
        '指标': ['准确率 (ACC)', '召回率 (Recall)', 'F1分数',
                '平均绝对误差 (MAE)', '均方误差 (MSE)', '均方根误差 (RMSE)',
                '准确率 (ACC)', '马修斯相关系数 (MCC)', '召回率 (Recall)', 'F1分数'],
        '值': [event_type_acc, event_type_recall, event_type_f1,
              event_time_mae, event_time_mse, event_time_rmse,
              quality_acc, quality_mcc, quality_recall, quality_f1]
    })
    
    results_df.to_csv(f"{results_dir}/evaluation_results.csv", index=False)
    print(f"\n评估结果已保存到 {results_dir}/evaluation_results.csv")
    
    return results, all_hidden_states, {
        'event_type': (all_event_type_true, all_event_type_preds),
        'event_time': (all_event_time_true, all_event_time_preds),
        'quality': (all_quality_true, all_quality_preds)
    }

def compare_with_baselines(model_path='models/best_model.pth'):
    """
    与基线模型进行比较（LR, RNN, LSTM, GRU）
    注：这里仅展示比较框架，实际需要实现这些基线模型
    """
    # 加载数据
    data = load_and_preprocess_data()
    
    # 评估我们的模型
    our_results, _, _ = evaluate_model(model_path, data)
    
    # 假设这些是基线模型的结果（实际应用中需要真实评估）
    baseline_results = {
        'LR': {
            'quality': {
                'accuracy': 0.36,
                'mcc': 0.12
            },
            'bookmark_removal': {
                'accuracy': 0.52,
                'recall': 0.70,
                'f1': 0.75
            }
        },
        'RNN': {
            'quality': {
                'accuracy': 0.36,
                'mcc': 0.09
            },
            'bookmark_removal': {
                'accuracy': 0.48,
                'recall': 0.52,
                'f1': 0.72
            }
        },
        'GRU': {
            'quality': {
                'accuracy': 0.32,
                'mcc': 0.04
            },
            'bookmark_removal': {
                'accuracy': 0.53,
                'recall': 0.70,
                'f1': 0.62
            }
        },
        'LSTM': {
            'quality': {
                'accuracy': 0.36,
                'mcc': 0.11
            },
            'bookmark_removal': {
                'accuracy': 0.66,
                'recall': 0.78,
                'f1': 0.72
            }
        },
        'Our Model': {
            'quality': {
                'accuracy': our_results['quality_level']['accuracy'],
                'mcc': our_results['quality_level']['mcc']
            },
            'bookmark_removal': {
                'accuracy': our_results['event_type']['accuracy'],  # 简化处理，实际应专门评估取消收藏
                'recall': our_results['event_type']['recall'],
                'f1': our_results['event_type']['f1']
            }
        }
    }
    
    # 创建结果目录
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 绘制直播质量预测比较
    models = list(baseline_results.keys())
    acc_scores = [baseline_results[m]['quality']['accuracy'] for m in models]
    mcc_scores = [baseline_results[m]['quality']['mcc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, acc_scores, width, label='准确率 (ACC)')
    plt.bar(x + width/2, mcc_scores, width, label='马修斯相关系数 (MCC)')
    
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.title('不同模型的直播质量预测性能比较')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/model_comparison_quality.png")
    plt.close()
    
    # 绘制取消收藏预测比较
    acc_scores = [baseline_results[m]['bookmark_removal']['accuracy'] for m in models]
    recall_scores = [baseline_results[m]['bookmark_removal']['recall'] for m in models]
    f1_scores = [baseline_results[m]['bookmark_removal']['f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, acc_scores, width, label='准确率 (ACC)')
    plt.bar(x, recall_scores, width, label='召回率 (Recall)')
    plt.bar(x + width, f1_scores, width, label='F1分数')
    
    plt.xlabel('模型')
    plt.ylabel('分数')
    plt.title('不同模型的取消收藏预测性能比较')
    plt.xticks(x, models)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/model_comparison_bookmark.png")
    plt.close()
    
    print("\n模型比较完成，比较图表已保存")
    return baseline_results

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估ODE-LSTM Hawkes模型')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='模型路径')
    parser.add_argument('--compare', action='store_true', help='是否与基线模型比较')
    
    args = parser.parse_args()
    
    # 评估模型
    results, _, _ = evaluate_model(args.model_path)
    
    # 如果需要，与基线模型比较
    if args.compare:
        compare_with_baselines(args.model_path)
