import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import torch

def visualize_hidden_states(hidden_states, labels, title, filename, n_samples=1000):
    """
    使用t-SNE可视化隐藏状态
    """
    # 创建结果目录
    vis_dir = 'visualizations'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 随机采样以加快可视化速度
    if len(hidden_states) > n_samples:
        indices = np.random.choice(len(hidden_states), n_samples, replace=False)
        hidden_states = hidden_states[indices]
        labels = labels[indices]
    
    # 使用最后一个时间步的隐藏状态
    if len(hidden_states.shape) == 3:  # (samples, seq_len, features)
        last_step_hidden = hidden_states[:, -1, :]
    else:
        last_step_hidden = hidden_states
    
    # 应用t-SNE进行降维
    print(f"正在对 {last_step_hidden.shape[0]} 个样本应用t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(last_step_hidden)
    
    # 创建可视化DataFrame
    df = pd.DataFrame({
        'tsne_1': tsne_results[:, 0],
        'tsne_2': tsne_results[:, 1],
        'label': labels
    })
    
    # 绘制t-SNE结果
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='tsne_1', y='tsne_2',
        hue='label',
        palette=sns.color_palette('hsv', len(np.unique(labels))),
        data=df,
        alpha=0.7
    )
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE 维度 1')
    plt.ylabel('t-SNE 维度 2')
    plt.legend(title='标签', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/{filename}")
    plt.close()
    
    print(f"t-SNE可视化已保存到 {vis_dir}/{filename}")
    return df

def visualize_results_comparison():
    """
    可视化不同模型的性能比较
    """
    # 假设这些是从评估结果中加载的数据
    # 实际应用中应该从evaluation_results.csv加载
    models = ['LR', 'RNN', 'GRU', 'LSTM', 'Our Model']
    
    # 直播质量预测结果
    quality_acc = [0.36, 0.36, 0.32, 0.36, 0.41]
    quality_mcc = [0.12, 0.09, 0.04, 0.11, 0.18]
    
    # 取消收藏预测结果
    bookmark_acc = [0.52, 0.48, 0.53, 0.66, 0.70]
    bookmark_recall = [0.70, 0.52, 0.70, 0.78, 0.90]
    bookmark_f1 = [0.75, 0.72, 0.62, 0.72, 0.85]
    
    # 创建结果目录
    vis_dir = 'visualizations'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 绘制直播质量预测比较
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, quality_acc, width, label='准确率 (ACC)')
    plt.bar(x + width/2, quality_mcc, width, label='马修斯相关系数 (MCC)')
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.title('直播质量预测性能比较', fontsize=14)
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/quality_comparison.png")
    plt.close()
    
    # 绘制取消收藏预测比较
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, bookmark_acc, width, label='准确率 (ACC)')
    plt.bar(x, bookmark_recall, width, label='召回率 (Recall)')
    plt.bar(x + width, bookmark_f1, width, label='F1分数')
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.title('取消收藏预测性能比较', fontsize=14)
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/bookmark_comparison.png")
    plt.close()
    
    print("模型比较可视化已保存")

def visualize_threshold_analysis():
    """
    可视化质量等级阈值分析
    """
    # 生成模拟的watch_to_order_ratio数据
    np.random.seed(42)
    n_samples = 10000
    
    # 生成混合分布数据
    low = np.random.beta(1, 15, int(n_samples * 0.8))
    medium = np.random.beta(2, 8, int(n_samples * 0.15))
    high = np.random.beta(5, 3, int(n_samples * 0.05))
    ratios = np.concatenate([low, medium, high])
    
    # 计算阈值
    thresholds = np.percentile(ratios, [20, 40, 60, 80])
    
    # 创建结果目录
    vis_dir = 'visualizations'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 绘制分布和阈值
    plt.figure(figsize=(12, 6))
    sns.histplot(ratios, kde=True, bins=50)
    
    # 添加阈值线
    colors = ['red', 'orange', 'green', 'blue']
    labels = ['20th 分位数', '40th 分位数', '60th 分位数', '80th 分位数']
    for threshold, color, label in zip(thresholds, colors, labels):
        plt.axvline(threshold, color=color, linestyle='--', linewidth=2, 
                   label=f'{label}: {threshold:.3f}')
    
    plt.title('观看-下单比率分布与质量等级阈值', fontsize=14)
    plt.xlabel('观看-下单比率', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/threshold_analysis.png")
    plt.close()
    
    # 绘制累积分布函数
    sorted_ratios = np.sort(ratios)
    cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_ratios, cdf)
    
    for threshold, q in zip(thresholds, [0.2, 0.4, 0.6, 0.8]):
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.7)
        plt.axhline(q, color='gray', linestyle=':', alpha=0.7)
        plt.scatter([threshold], [q], color='red', s=100, zorder=5)
        plt.text(threshold, q, f'  {q:.0%}', verticalalignment='bottom')
    
    plt.title('观看-下单比率的累积分布函数', fontsize=14)
    plt.xlabel('观看-下单比率', fontsize=12)
    plt.ylabel('累积概率', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{vis_dir}/ratio_cdf.png")
    plt.close()
    
    print("阈值分析可视化已保存")

if __name__ == "__main__":
    # 运行所有可视化
    print("生成可视化结果...")
    
    # 可视化阈值分析
    visualize_threshold_analysis()
    
    # 可视化模型比较
    visualize_results_comparison()
    
    # 如果有隐藏状态数据，可以可视化
    try:
        # 假设我们有保存的隐藏状态和标签
        if os.path.exists('results/hidden_states.npy') and os.path.exists('results/labels.npy'):
            hidden_states = np.load('results/hidden_states.npy')
            labels = np.load('results/labels.npy')
            
            # 按质量等级可视化
            visualize_hidden_states(
                hidden_states, 
                labels, 
                '按直播质量等级的隐藏状态t-SNE可视化', 
                'quality_hidden_states_tsne.png'
            )
    except Exception as e:
        print(f"可视化隐藏状态时出错: {e}")
        print("请先运行评估以生成隐藏状态数据")
    
    print("所有可视化结果已生成")
