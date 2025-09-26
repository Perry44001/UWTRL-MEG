import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和SCI绘图风格
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'savefig.dpi': 300,
    'figure.dpi': 300
})

# 创建输出目录（如果不存在）
output_dir = './loc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取JSON文件
json_path = r'e:/2.0_UWTR/models/meg(gfcc)/train_params.json'
with open(json_path, 'r') as f:
    train_params = json.load(f)

# 提取需要的参数列表
loss_train_list = train_params['loss_train_list']
acc_str_train_list = train_params['acc_str_train_list']
acc_val_list = train_params['acc_val_list']  # 添加测试准确率
pres_rc_train_list = train_params['pres_rc_train_list']
pres_lr_train_list = train_params['pres_lr_train_list']
pres_lz_train_list = train_params['pres_lz_train_list']
ABSE_Rs_train_list = train_params['ABSE_Rs_train_list']
ABSE_Ds_train_list = train_params.get('ABSE_Ds_train_list', [])
ABSE_Rs_val_list = train_params['ABSE_Rs_val_list']  # 添加测试ABSE-R
ABSE_Ds_val_list = train_params.get('ABSE_Ds_val_list', [])  # 添加测试ABSE-D

# 1. 绘制Loss曲线并保存为独立图像
plt.figure(figsize=(5, 4))
plt.plot(range(len(loss_train_list)), loss_train_list, color='#1f77b4')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
# plt.title('(a) Training Loss')
plt.tight_layout()
loss_path = os.path.join(output_dir, 'training_loss.png')
plt.savefig(loss_path, dpi=300, bbox_inches='tight')
loss_path = os.path.join(output_dir, 'training_loss.pdf')
plt.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.close()

# 2. 绘制多任务权重曲线并保存为独立图像
plt.figure(figsize=(5, 4))
plt.plot(range(len(pres_rc_train_list)), pres_rc_train_list, label='c-weight', color='#1f77b4')
plt.plot(range(len(pres_lr_train_list)), pres_lr_train_list, label='r-weight', color='#ff7f0e')
plt.plot(range(len(pres_lz_train_list)), pres_lz_train_list, label='d-weight', color='#2ca02c')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Weight')
# plt.title('(b) Task Weights')
plt.tight_layout()
weights_path = os.path.join(output_dir, 'task_weights.png')
plt.savefig(weights_path, dpi=300, bbox_inches='tight')
weights_path = os.path.join(output_dir, 'task_weights.pdf')
plt.savefig(weights_path, dpi=300, bbox_inches='tight')
plt.close()

# 3. 绘制准确率曲线并保存为独立图像
plt.figure(figsize=(5, 4))
plt.plot(range(len(acc_str_train_list)), acc_str_train_list, color='#1f77b4', label='Train')
plt.plot(range(len(acc_val_list)), acc_val_list, color='#ff7f0e', label='Test')  # 添加测试曲线
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')  # 添加图例
plt.tight_layout()
acc_path = os.path.join(output_dir, 'training_accuracy.png')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
acc_path = os.path.join(output_dir, 'training_accuracy.pdf')
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
plt.close()

# 4. 绘制ABSE曲线并保存为独立图像
plt.figure(figsize=(5, 4))
plt.plot(range(len(ABSE_Rs_train_list)), ABSE_Rs_train_list, label='r-train', color='#1f77b4', linestyle='--')
plt.plot(range(len(ABSE_Rs_val_list)), ABSE_Rs_val_list, label='r-test', color='#1f77b4', linestyle='-')  # 添加测试ABSE-R
if ABSE_Ds_train_list:
    plt.plot(range(len(ABSE_Ds_train_list)), ABSE_Ds_train_list, label='d-train', color='#ff7f0e', linestyle='--')
if ABSE_Ds_val_list:  # 添加测试ABSE-D
    plt.plot(range(len(ABSE_Ds_val_list)), ABSE_Ds_val_list, label='d-test', color='#ff7f0e', linestyle='-')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Normalized ABSE')
# plt.title('(d) Training ABSE')
plt.tight_layout()
abse_path = os.path.join(output_dir, 'training_abse.png')
plt.savefig(abse_path, dpi=300, bbox_inches='tight')
abse_path = os.path.join(output_dir, 'training_abse.pdf')
plt.savefig(abse_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'4个图像已保存至: {output_dir}')