import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

# 设置matplotlib参数，使其符合科学论文要求
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
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

# 定义要分析的网络列表
networks = [
    'meg_mix',
    'meg_blc',
    'meg(gfcc)',
    'meg(stft)',
    'mcl(gfcc)',
    'mcl(stft)'
]

# 定义不同的线型和标记样式
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'x']
colors = plt.cm.tab20(np.linspace(0, 1, len(networks)))

# 存储每个网络的定位误差数据
rs_errors = {}
ds_errors = {}

def smooth_curve(points, factor=0.8):
    """使用指数移动平均平滑曲线"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# 读取每个网络的数据
for network in networks:
    try:
        with open(f'../models/{network}/train_params.json', 'r') as f:
            data = json.load(f)
            rs_errors[network] = data['ABSE_Rs_val_list']
            ds_errors[network] = data['ABSE_Ds_val_list']
    except Exception as e:
        print(f"无法读取 {network} 的数据: {str(e)}")

# 绘制径向误差曲线
fig, ax = plt.subplots(figsize=(8, 6))
all_rs = np.concatenate([rs_errors[n] for n in rs_errors]) if rs_errors else np.array([0, 1])
rs_min, rs_max = np.min(all_rs), np.max(all_rs)
rs_pad = (rs_max - rs_min) * 0.1 if rs_max > rs_min else 0.05
for i, (network, errors) in enumerate(rs_errors.items()):
    smoothed_errors = smooth_curve(errors)
    line_style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i]
    ax.plot(errors, alpha=0.2, color=color)
    ax.plot(smoothed_errors, label=network, linestyle=line_style, marker=marker, markevery=5, color=color, linewidth=1.5, markersize=4)
ax.set_xlabel('Training Epochs', fontweight='bold')
ax.set_ylabel('Radial Error (m)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=False, edgecolor='black')
ax.set_ylim(rs_min - rs_pad, rs_max + rs_pad)
ax.set_xlim(0, len(rs_errors[list(rs_errors.keys())[0]]))
ax.tick_params(direction='out', length=4, width=1)
plt.savefig('radial_error_curves.pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

# 绘制深度误差曲线
fig, ax = plt.subplots(figsize=(8, 6))
all_ds = np.concatenate([ds_errors[n] for n in ds_errors]) if ds_errors else np.array([0, 1])
ds_min, ds_max = np.min(all_ds), np.max(all_ds)
ds_pad = (ds_max - ds_min) * 0.1 if ds_max > ds_min else 0.05
for i, (network, errors) in enumerate(ds_errors.items()):
    smoothed_errors = smooth_curve(errors)
    line_style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i]
    ax.plot(errors, alpha=0.2, color=color)
    ax.plot(smoothed_errors, label=network, linestyle=line_style, marker=marker, markevery=5, color=color, linewidth=1.5, markersize=4)
ax.set_xlabel('Training Epochs', fontweight='bold')
ax.set_ylabel('Depth Error (m)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=False, edgecolor='black')
ax.set_ylim(ds_min - ds_pad, ds_max + ds_pad)
ax.set_xlim(0, len(ds_errors[list(ds_errors.keys())[0]]))
ax.tick_params(direction='out', length=4, width=1)
plt.savefig('depth_error_curves.pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

# 计算最终性能对比
final_rs_errors = {network: errors[-1] for network, errors in rs_errors.items()}
final_ds_errors = {network: errors[-1] for network, errors in ds_errors.items()}

# 绘制最终性能对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 径向误差对比
networks = list(final_rs_errors.keys())
rs_values = list(final_rs_errors.values())
bars1 = ax1.bar(networks, rs_values, color=colors, width=0.6)
ax1.set_xlabel('Network Architecture', fontweight='bold')
ax1.set_ylabel('Final Radial Error (m)', fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom',
            fontsize=10)

# 深度误差对比
ds_values = list(final_ds_errors.values())
bars2 = ax2.bar(networks, ds_values, color=colors, width=0.6)
ax2.set_xlabel('Network Architecture', fontweight='bold')
ax2.set_ylabel('Final Depth Error (m)', fontweight='bold')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# 在柱状图上添加数值标签
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom',
            fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('final_localization_performance.pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

# 打印详细分析结果
print("\n定位性能分析结果：")
print("-" * 50)
for network in networks:
    if network in final_rs_errors:
        print(f"{network}:")
        print(f"  最终径向误差: {final_rs_errors[network]:.3f} m")
        print(f"  最终深度误差: {final_ds_errors[network]:.3f} m")
        print("-" * 50) 