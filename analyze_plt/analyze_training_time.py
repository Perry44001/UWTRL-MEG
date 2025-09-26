import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置matplotlib参数，使其符合科学论文要求
plt.rcParams.update({
    'font.family': 'times new roman',
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# 读取数据
df = pd.read_csv('train_time2.csv')

# 处理百分比数据
df['ACC(%)'] = df['ACC(%)'].str.rstrip('%').astype('float') / 100.0

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 设置颜色方案
colors = plt.cm.tab20(np.linspace(0, 1, len(df)+1))

# 绘制柱状图
bars = ax.bar(df['Model'], df['Training cost(min)'], color=colors, width=0.6)

# 在柱状图上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=16)

# 设置坐标轴标签
ax.set_xlabel('Network', fontweight='bold')
ax.set_ylabel('Training Time (min)', fontweight='bold')

# 设置x轴标签旋转
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 添加网格线
ax.grid(True, alpha=0.3, axis='y')

# 设置y轴范围，留出一定空间显示数值标签
y_max = df['Training cost(min)'].max()
ax.set_ylim(0, y_max * 1.1)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('training_time_comparison.pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

# 打印详细分析结果
print("\n训练时间分析结果：")
print("-" * 50)
for _, row in df.iterrows():
    print(f"{row['Model']}:")
    print(f"  训练时间: {int(row['Training cost(min)'])} 分钟")
    print(f"  准确率: {row['ACC(%)']:.2%}")
    # 只在误差为数字时才格式化输出
    try:
        mae_r = float(row['MAE-R(km)'])
        mae_z = float(row['MAE-Z(m)'])
        print(f"  径向误差: {mae_r:.4f} km")
        print(f"  深度误差: {mae_z:.2f} m")
    except (ValueError, TypeError):
        pass
    print("-" * 50) 