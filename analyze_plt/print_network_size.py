import torch
import torch.nn as nn
from nw_mtl import MultiTaskLossWrapper
from nw_class_moe import class_network, MultiFeatureClassMOENetwork


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_network_size(model_name, feature_dim, channels, embd_dim, num_classes, num_experts, k, features, task_type='mtl'):
    """打印网络大小"""
    # 获取输入特征大小
    input_size = feature_dim[features[0]]
    
    # 实例化完整的多任务模型
    model = MultiTaskLossWrapper(
        model_name=model_name,
        feature_dim=feature_dim,
        channels=channels,
        embd_dim=embd_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        k=k,
        task_type=task_type,
        features=features
    )
    
    # 分别计算总参数、分类网络参数和定位网络参数
    total_params = count_parameters(model)
    class_params = count_parameters(model.classn)
    local_params = count_parameters(model.localn)
    
    print(f"\n{model_name} 网络大小:")
    print(f"总参数量: {total_params:,}")
    print(f"分类网络参数量: {class_params:,}")
    print(f"定位网络参数量: {local_params:,}")
    print(f"总参数量 (MB): {total_params * 4 / (1024 * 1024):.2f}")

def main():
    # 基础配置
    feature_dim = {'mel': 200, 'stft': 513, 'cqt': 84, 'gfcc': 200, 'mfcc': 40}
    channels = 512
    embd_dim = 192
    num_classes = 5
    num_experts = 5
    k = 3
    features = ['gfcc']
    task_type = 'mtl'  # 多任务类型，与train_mtl.py保持一致

    # 要测试的网络列表
    networks = [
        'meg',
        'mcl',
        'meg_blc',
        'densenet121',
        'resnet18',
        'mobilenetv2',
        'resnet50',
        'swin'
    ]

    print("开始打印各网络大小...")
    for network in networks:
        print_network_size(
            model_name=network,
            feature_dim=feature_dim,
            channels=channels,
            embd_dim=embd_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            k=k,
            features=features,
            task_type=task_type
        )

if __name__ == '__main__':
    main()