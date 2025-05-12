import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from itertools import cycle
from sklearn.preprocessing import label_binarize

from nw_mtl import MultiTaskLossWrapper
from md_moe_rl import Dataset_audio

# 设置全局字体为 Times New Roman，字体大小为 24
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 24})

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch Model Testing')
    parser.add_argument('--model_path', type=str,
                        help='Path to the trained model checkpoint', default='E:/2.0_UWTR/models/2025-05-02-17-13-08epoch150_bs64_lr0.001/model.pth')
    parser.add_argument("--test_list_path", type=str, default='E:/MTQP/wjy_codes/shipsear_5s_16k_ocnwav_Pos/test_list.txt')
    parser.add_argument("--label_list_path", type=str, default='E:/MTQP/wjy_codes/shipsear_5s_16k_ocnwav/label_list.txt')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Test batch size')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--features', nargs='+', default=['stft'],
                        help='List of features to use: mel, welch, avg, gfcc, stft, cqt')
    parser.add_argument('--model', type=str, default='meg',
                        choices=['meg', 'meg_ori', 'meg_rsps', 'megx', 'meg_e', 'meg_mix', 'resnet18', 'resnet50', 'convnext', 'vgg16', 'vgg19',
                                 'mobilenetv2', 'densenet121', 'swin'],
                        help='Backbone network architecture')
    parser.add_argument('--task_type', type=str, default='mtl',
                        choices=['mtl', 'classification', 'localization'],
                        help='Task type: mtl, classification, or localization')
    return parser.parse_args()

def plot_confusion_matrix(cm, class_labels, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 8), dpi=300)
    size = len(class_labels)
    proportion = []
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(size, size) 
    pshow = np.array(pshow).reshape(size, size)
    #print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    print(proportion)

    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)  # 按照像素显示出矩阵
    
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, fontsize=24)
    plt.yticks(tick_marks, class_labels, fontsize=24)
    
    thresh = cm.max() / 2.
    iters = np.reshape([[[i, j] for j in range(size)] for i in range(size)], (cm.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=18, color='white', weight=5)  # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=18, color='white')
        else:
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=18)   # 显示对应的数字
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=18)
    
    plt.ylabel('True label', fontsize=24)
    plt.xlabel('Predict label', fontsize=24)
    plt.tight_layout()
    # plt.show()

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # PNG
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')

    
    plt.close('all')

def plot_roc_curves(y_true, y_score, class_labels, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(9, 8), dpi=300)
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # 将标签转换为one-hot编码
    n_classes = len(class_labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
    
    # 绘制所有类别的ROC曲线
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {class_labels[i]} (AUC = {roc_auc[i]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def test_model(model, test_loader, device, class_labels, output_dir):
    """测试模型并生成可视化结果"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    all_distances = []
    all_depths = []
    all_Rr = []  # 添加存储真实距离的列表
    all_Sz = []  # 添加存储真实深度的列表
    
    with torch.no_grad():
        for inputs_dict, labels, Rr, Sz in test_loader:
            # 将数据移到设备上
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            labels = labels.to(device)
            Rr = Rr.to(device)
            Sz = Sz.to(device)
            
            # 前向传播
            _, outputs, outtaskLocR, outtaskLocD, _ = model(inputs_dict, labels, Rr, Sz)
            
            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
            all_distances.extend(outtaskLocR.cpu().numpy())
            all_depths.extend(outtaskLocD.cpu().numpy())
            all_Rr.extend(Rr.cpu().numpy())  # 收集真实距离
            all_Sz.extend(Sz.cpu().numpy())  # 收集真实深度
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_distances = np.array(all_distances)
    all_depths = np.array(all_depths)
    all_Rr = np.array(all_Rr)  # 转换为numpy数组
    all_Sz = np.array(all_Sz)  # 转换为numpy数组
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plot_confusion_matrix(cm, class_labels, os.path.join(output_dir, 'confusion_matrix.pdf'))
    
    # 绘制ROC曲线
    plot_roc_curves(all_labels, all_scores, class_labels, os.path.join(output_dir, 'roc_curves.pdf'))
    
    # 计算并打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_labels))
    
    # 计算平均绝对误差
    mae_distance = np.mean(np.abs(all_distances - all_Rr))
    mae_depth = np.mean(np.abs(all_depths - all_Sz))
    print(f"\nMean Absolute Error - Distance: {mae_distance:.4f}")
    print(f"Mean Absolute Error - Depth: {mae_depth:.4f}")
    
    # 保存结果到JSON文件
    results = {
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(all_labels, all_preds, target_names=class_labels, output_dict=True),
        'mae_distance': float(mae_distance),
        'mae_depth': float(mae_depth)
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def main():
    args = get_args()
    
    # 获取模型文件所在目录作为输出目录
    output_dir = os.path.dirname(args.model_path)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载标签
    with open(args.label_list_path, 'r', encoding='utf-8') as f:
        class_labels = [line.strip() for line in f.readlines()]
    
    # 创建数据集和数据加载器
    test_dataset = Dataset_audio(args.test_list_path, sr=16000, chunk_duration=5, features=args.features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           num_workers=args.num_workers, pin_memory=True)
    
    # 创建模型
    feature_dim = {'mel': 200, 'stft': 513, 'cqt': 84, 'gfcc': 200, 'mfcc': 40}
    model = MultiTaskLossWrapper(
        model_name=args.model,
        feature_dim=feature_dim,
        channels=512,
        embd_dim=192,
        num_classes=len(class_labels),
        num_experts=5,
        k=3,
        task_type=args.task_type,
        features=args.features
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(args.model_path, weights_only=True))  # 添加 weights_only=True 参数
    model = model.to(device)
    
    # 测试模型
    test_model(model, test_loader, device, class_labels, output_dir)

if __name__ == '__main__':
    main() 