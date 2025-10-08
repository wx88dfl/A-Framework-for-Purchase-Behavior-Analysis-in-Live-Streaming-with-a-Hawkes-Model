# 计算分位数阈值
quantiles = [0.20, 0.40, 0.60, 0.80]
thresholds = [ratio_df['watch_to_order_ratio'].quantile(q) for q in quantiles]

print("\n基于分位数分析的阈值确定:")
print("=" * 50)
for i, (q, threshold) in enumerate(zip(quantiles, thresholds)):
    print(f"分位数 {q:.0%}: {threshold:.3f}")

# 质量等级划分
quality_levels = []
for ratio in ratio_df['watch_to_order_ratio']:
    if ratio < thresholds[0]:
        quality_levels.append(0)  # 低质量
    elif ratio < thresholds[1]:
        quality_levels.append(1)  # 中等质量
    elif ratio < thresholds[2]:
        quality_levels.append(2)  # 良好质量
    elif ratio < thresholds[3]:
        quality_levels.append(3)  # 高质量
    else:
        quality_levels.append(4)  # 极高质量

ratio_df['quality_level'] = quality_levels

# 各质量等级的统计
quality_stats = ratio_df.groupby('quality_level')['watch_to_order_ratio'].agg(['count', 'mean', 'std'])
quality_stats['percentage'] = (quality_stats['count'] / len(ratio_df) * 100).round(2)

print("\n各质量等级分布:")
print("=" * 50)
print(quality_stats)