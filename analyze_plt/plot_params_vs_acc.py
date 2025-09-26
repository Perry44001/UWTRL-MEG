import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置全局字体
mpl.rcParams['font.family'] = 'Times New Roman'

# 读取数据
df = pd.read_csv('params.csv')

# 数据预处理
# 处理准确率（ACC，%）列，去除百分号并转换为数值（如果有的话）
# 但根据数据显示，这一列已经是数值格式
acc = df['准确率（ACC，%）']
params = df['参数规模（MB）']
models = df['网络']

# 创建图形和轴
plt.figure(figsize=(12, 8))

# 定义颜色轮换
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['*', 'o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

# 绘制星标图，使用轮换的颜色和标记
for i, model in enumerate(models):
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    plt.scatter(params.iloc[i], acc.iloc[i], s=150, marker=marker, color=color, edgecolors='black', linewidth=0.5)

# 为每个点添加模型名称标签（增大字体）
for i, model in enumerate(models):
    if i == 1:
        plt.annotate(model, (params.iloc[i]-35, acc.iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=22)
    elif i == 6:
        plt.annotate(model, (params.iloc[i]-45, acc.iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=22)
    else:
        plt.annotate(model, (params.iloc[i]+1, acc.iloc[i]-0.5), 
                 xytext=(5, 5), textcoords='offset points', fontsize=22)

# 设置图表属性（增大字体）
plt.xlabel('Parameter Scale (MB)', fontsize=22, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=22, fontweight='bold')
plt.title('Params vs Accuracy', fontsize=22, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim([60, 100])

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# 设置边框样式
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('#BBBBBB')
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('params_vs_acc.pdf', format='pdf', bbox_inches='tight')
plt.savefig('params_vs_acc.png', dpi=300, bbox_inches='tight')
plt.show()