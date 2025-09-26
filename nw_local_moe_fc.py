# 导入 PyTorch 库，用于深度学习相关操作
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的神经网络函数模块
import torch.nn.functional as F
# 导入 PyTorch 的初始化模块
from torch.nn import init

# 导入 PyTorch 卷积层的基类
from torch.nn.modules.conv import _ConvNd
# 导入 PyTorch 用于处理二维参数的工具
from torch.nn.modules.utils import _pair

# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 PyTorch 的自动求导变量模块（旧版本，现多使用张量直接处理）
from torch.autograd import Variable
# 导入 PyTorch 的参数模块
from torch.nn import Parameter

class local_head(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192, model_name='meg'):
        super().__init__()
        # 移除原有卷积层，仅保留输入尺寸参数
        self.input_size = input_size
        self.model_name = model_name

    def forward(self, x):
        # 第一层：按通道维度取平均（全局平均池化）
        out = F.adaptive_avg_pool1d(x, 1)  # 形状: (batch, input_size, 1)
        out = out.view(out.size(0), -1)  # 展平为1维特征: (batch, input_size)
        # 保持原输出结构（返回两个相同的特征用于r和z分支）
        return out, out

class local_network(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192, num_experts=5, k=3, model_name='meg'):
        super().__init__()
        self.local_head = local_head(input_size, channels, embd_dim)
        # 第二层：全连接层（输入为平均池化后的特征维度）
        self.fc1_r = nn.Linear(input_size, embd_dim)
        self.fc1_z = nn.Linear(input_size, embd_dim)
        # 第三层：全连接输出层
        self.fc2_r = nn.Linear(embd_dim, 1)
        self.fc2_z = nn.Linear(embd_dim, 1)
        self.model_name = model_name

        if model_name == 'meg_blc':
            # Xavier初始化全连接层权重
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # 获取平均池化后的1维特征
        out_r, out_z = self.local_head(x)
        # 第二层全连接 + ReLU激活
        out_r = F.relu(self.fc1_r(out_r))
        out_z = F.relu(self.fc1_z(out_z))

        # 第三层全连接输出
        final_output_r = F.relu(self.fc2_r(out_r))
        final_output_z = F.relu(self.fc2_z(out_z))
        return final_output_r.squeeze(-1), final_output_z.squeeze(-1), 0.0


class MultiFeatureLocalMOENetwork(nn.Module):
    def __init__(self, feature_configs, channels=512, embd_dim=192, dropout_rate=0.3):
        """
        多特征MOE定位网络
        
        Args:
            feature_configs: 特征配置列表，每个元素为字典，包含：
                - name: 特征名称 ('GFCC', 'STFT', 'MEL', 'CQT', 'MFCC')
                - input_size: 输入特征维度
            channels: 卷积通道数
            embd_dim: 嵌入维度
            dropout_rate: Dropout比率
        """
        super().__init__()
        self.num_features = len(feature_configs)
        
        # 为每个特征创建定位网络
        self.feature_nets = nn.ModuleDict({
            config['name']: local_head(config['input_size'], channels, embd_dim, model_name='mcl')
            for config in feature_configs
        })
        
        # 每个特征对应一个专家网络（r和z分别一个）
        self.experts_r = nn.ModuleDict({
            config['name']: nn.Sequential(
                nn.Linear(embd_dim, embd_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embd_dim // 2, 1)
            ) for config in feature_configs
        })
        
        self.experts_z = nn.ModuleDict({
            config['name']: nn.Sequential(
                nn.Linear(embd_dim, embd_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embd_dim // 2, 1)
            ) for config in feature_configs
        })
        
        # 为r和z分别创建独立的路由器
        self.router_r = nn.Sequential(
            nn.Linear(embd_dim * self.num_features, self.num_features),
            nn.Softmax(dim=-1)
        )
        
        self.router_z = nn.Sequential(
            nn.Linear(embd_dim * self.num_features, self.num_features),
            nn.Softmax(dim=-1)
        )
        
        # 为r和z分别添加温度参数
        self.temperature_r = nn.Parameter(torch.ones(1) * 1.0)
        self.temperature_z = nn.Parameter(torch.ones(1) * 1.0)
        
        # 添加L2正则化
        self.l2_lambda = 0.01
        
    def forward(self, feature_dict):
        """
        前向传播
        
        Args:
            feature_dict: 特征字典，键为特征名称，值为对应的特征张量
        """
        # 提取每个特征的特征向量
        feature_embeddings_r = []
        feature_embeddings_z = []
        for name, net in self.feature_nets.items():
            r, z = net(feature_dict[name])
            feature_embeddings_r.append(r)
            feature_embeddings_z.append(z)
        
        # 分别拼接r和z的特征向量
        combined_features_r = torch.cat(feature_embeddings_r, dim=1)
        combined_features_z = torch.cat(feature_embeddings_z, dim=1)
        
        # 分别计算r和z的权重
        router_weights_r = self.router_r(combined_features_r)
        router_weights_r = router_weights_r / self.temperature_r
        
        router_weights_z = self.router_z(combined_features_z)
        router_weights_z = router_weights_z / self.temperature_z
        
        # 计算每个专家的输出
        expert_outputs_r = []
        expert_outputs_z = []
        for name in self.experts_r.keys():
            idx = list(self.experts_r.keys()).index(name)
            expert_outputs_r.append(self.experts_r[name](feature_embeddings_r[idx]))
            expert_outputs_z.append(self.experts_z[name](feature_embeddings_z[idx]))
        
        expert_outputs_r = torch.stack(expert_outputs_r, dim=1)
        expert_outputs_z = torch.stack(expert_outputs_z, dim=1)
        
        # 分别使用加权平均
        final_output_r = torch.sum(expert_outputs_r * router_weights_r.unsqueeze(-1), dim=1)
        final_output_z = torch.sum(expert_outputs_z * router_weights_z.unsqueeze(-1), dim=1)
        
        # 分别计算r和z的负载均衡损失
        balance_loss_r = load_balancing_loss(self.router_r(combined_features_r), self.num_features)
        balance_loss_z = load_balancing_loss(self.router_z(combined_features_z), self.num_features)
        balance_loss = balance_loss_r + balance_loss_z
        
        return final_output_r.squeeze(-1), final_output_z.squeeze(-1), balance_loss

if __name__ == '__main__':
    # 测试参数
    input_size = 201
    channels = 512
    embd_dim = 192
    num_experts = 5
    k = 3
    
    # 创建模型实例
    model = local_network(input_size, channels, embd_dim, num_experts, k)
    
    # 生成随机输入数据
    batch_size = 64
    input_tensor = torch.randn(batch_size, input_size, 512)
    
    # 前向传播
    output_r, output_z, balance_loss = model(input_tensor)
    
    # 打印输出形状和负载均衡损失
    print(f"Output r shape: {output_r.shape}")
    print(f"Output z shape: {output_z.shape}")
    print(f"Load balancing loss: {balance_loss.item()}")
    
    # 测试多特征MOE定位网络
    feature_configs = [
        {'name': 'GFCC', 'input_size': 80},
        {'name': 'STFT', 'input_size': 128}
    ]
    
    input_tensor = {
        'GFCC': torch.randn(batch_size, 80, 512),
        'STFT': torch.randn(batch_size, 128, 512)
    }
    
    multi_feature_model = MultiFeatureLocalMOENetwork(
        feature_configs=feature_configs,
        channels=channels,
        embd_dim=embd_dim
    )
    
    # 前向传播
    output_r, output_z, balance_loss = multi_feature_model(input_tensor)
    
    # 打印输出形状和负载均衡损失
    print("\nMulti-feature MOE Local Network:")
    print(f"Output r shape: {output_r.shape}")
    print(f"Output z shape: {output_z.shape}")
    print(f"Load balancing loss: {balance_loss.item()}") 