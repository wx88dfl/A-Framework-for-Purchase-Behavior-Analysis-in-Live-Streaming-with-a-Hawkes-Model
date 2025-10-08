# 正态性检验
shapiro_stat, shapiro_p = stats.shapiro(watch_to_order_ratios)
print(f"\n正态性检验 (Shapiro-Wilk):")
print(f"统计量: {shapiro_stat:.4f}, p-value: {shapiro_p:.4e}")

# 各质量等级间的方差分析
from scipy.stats import f_oneway
anova_groups = [ratio_df[ratio_df['quality_level'] == i]['watch_to_order_ratio'] for i in range(5)]
f_stat, p_value = f_oneway(*anova_groups)
print(f"\n方差分析 (ANOVA):")
print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4e}")

# Kruskal-Wallis 检验 (非参数替代)
h_stat, h_pvalue = stats.kruskal(*anova_groups)
print(f"\nKruskal-Wallis 检验:")
print(f"H-statistic: {h_stat:.4f}, p-value: {h_pvalue:.4e}")