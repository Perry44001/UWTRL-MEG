import torch
import torch.nn as nn
from nw_factory import create_class_network, create_local_network
from nw_moe import MOEMultiTaskNetwork, load_balancing_loss
from nw_class import class_network as class_network_ori
from nw_local import local_network as local_network_ori
from nw_class_moe import class_network
from nw_local_moe import local_network
from nw_drs import class_loc_network
from nw_class_moe_enhanced import class_network as class_network_e
from nw_local_moe_enhanced import local_network as local_network_e
from nw_moe_mix import MultiFeatureMOENetwork


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model_name, feature_dim, channels, embd_dim, num_classes, num_experts, k, task_type='mtl', features=None):
        super(MultiTaskLossWrapper, self).__init__()
        self.model_name = model_name
        self.task_type = task_type
        self.features = features
        self.feature_dim = feature_dim
        
        try:
            input_size = feature_dim[features[0]]
        except KeyError:
            raise ValueError(f'Unsupported feature type: {features[0]}')
        
        if model_name == 'mcl':
            self.classn = class_network_ori(input_size, channels, embd_dim, num_classes)
            self.localn = local_network_ori(input_size, channels, embd_dim)
        elif model_name == 'meg':
            # self.conv1 = nn.Conv1d(input_size, input_size, kernel_size=5, padding=2, dilation=1, bias=False)
            # self.bn1 = nn.BatchNorm1d(input_size)
            self.classn = class_network(input_size, channels, embd_dim, num_classes, num_experts, k)
            self.localn = local_network(input_size, channels, embd_dim, num_experts, k)
        elif model_name == 'meg_blc':
            self.classn = class_network_e(input_size, channels, embd_dim, num_classes, num_experts, k)
            self.localn = local_network_e(input_size, channels, embd_dim, num_experts, k)
        elif model_name == 'meg_mix':
            # 使用音频MOE网络进行分类
            feature_configs = [
                {'name': feature, 'input_size': feature_dim[feature]}
                for feature in features
            ]
            self.classn = MultiFeatureMOENetwork(
                feature_configs=feature_configs,
                channels=channels,
                embd_dim=embd_dim,
                num_classes=num_classes
            )
            # 使用local_network_e进行定位
            self.localn = local_network(input_size, channels, embd_dim, num_experts, k)
        else:
            # 使用常规网络 Resnet18 Resnet50 Mobilenetv2 Densenet121 Swin-Transformer
            self.classn = create_class_network(model_name, input_size, channels, embd_dim, num_classes)
            self.localn = create_class_network(model_name, input_size, channels, embd_dim, 2)
        
        # 任务权重参数
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.lossRec = torch.nn.CrossEntropyLoss()

    def forward(self, x, label, Rr, Sz):
        """
        前向传播函数

        参数:
        x (dict): 输入特征字典，键为特征名称，值为对应的特征张量
        label (Tensor): 分类标签
        Rr (Tensor): 定位距离的目标值
        Sz (Tensor): 定位深度的目标值

        返回:
        Tensor: 多任务损失
        """
        if self.model_name == 'megx':
            x = x[self.features[0]]
            # MOE网络前向传播
            class_output, (distance_output, depth_output), router_info = self.moe_network(x)
            
            # 计算分类损失
            lossRec = self.lossRec(class_output, label)
            
            # 计算定位损失
            lossLocR = torch.sum((Rr - distance_output.squeeze()) ** 2) 
            lossLocD = torch.sum((Sz - depth_output.squeeze()) ** 2) 
            
            # 计算负载均衡损失
            balance_loss = load_balancing_loss(router_info)
            
            # 计算任务权重
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2
            
            # 计算加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]
            
            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum()
            

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss + 0.01 * balance_loss
            elif self.task_type == 'classification':
                mtl_loss = lossRec
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum()
            
            return mtl_loss, class_output, distance_output.squeeze(), depth_output.squeeze(), c_tau_squared
        
        elif self.model_name == 'meg':
            x = x[self.features[0]]
            # x = self.conv1(x)
            # x = self.bn1(x)
            # 分类网络处理
            outtaskrcgC = self.classn(x)
            lossRec = self.lossRec(outtaskrcgC, label)
            
            # 定位网络处理
            outtaskLocR, outtaskLocD = self.localn(x)
            lossLocR = torch.sum((Rr - outtaskLocR) ** 2) 
            lossLocD = torch.sum((Sz - outtaskLocD) ** 2)

            # 计算 c_tau
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2

            # 计算每个任务的加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]
            
            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum()

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss
            elif self.task_type == 'classification':
                mtl_loss = lossRec
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum()

            return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared
        elif self.model_name == 'meg_ori':
            x = x[self.features[0]]

            # 分类网络处理
            outtaskrcgC = self.classn(x)
            lossRec = self.lossRec(outtaskrcgC, label)

            # 定位网络处理
            outtaskLocR, outtaskLocD = self.localn(x)
            lossLocR = torch.sum((Rr - outtaskLocR) ** 2) 
            lossLocD = torch.sum((Sz - outtaskLocD) ** 2)

            # 计算 c_tau
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2

            # 计算每个任务的加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]
            
            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum()

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss
            elif self.task_type == 'classification':
                mtl_loss = lossRec
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum()

            return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared
        elif self.model_name == 'meg_rsps':
            x = x[self.features[0]]

            # 分类网络处理
            outtaskrcgC, outtaskLocR, outtaskLocD = self.classlocn(x)
            lossRec = self.lossRec(outtaskrcgC, label)

            # 定位网络处理
            lossLocR = torch.sum((Rr - outtaskLocR) ** 2) 
            lossLocD = torch.sum((Sz - outtaskLocD) ** 2) 

            # 计算 c_tau
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2

            # 计算每个任务的加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]

            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum() 

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss
            elif self.task_type == 'classification':
                mtl_loss = lossRec
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum()

            return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared
        
        elif self.model_name == 'meg_e':
            x = x[self.features[0]]
            # 分类网络处理
            outtaskrcgC, balance_lossC = self.classn(x)
            lossRec = self.lossRec(outtaskrcgC, label)

            # 定位网络处理
            outtaskLocR, outtaskLocD, balance_lossL = self.localn(x)
            lossLocR = torch.sum((Rr - outtaskLocR) ** 2) 
            lossLocD = torch.sum((Sz - outtaskLocD) ** 2)

            # 计算 c_tau
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2

            # 计算每个任务的加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]
            
            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum()

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss + balance_lossC + balance_lossL
            elif self.task_type == 'classification':
                mtl_loss = lossRec + balance_lossC
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum() + balance_lossL

            return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared
        elif self.model_name == 'meg_mix':

            # 分类网络处理
            outtaskrcgC = self.classn(x)  # x 已经是特征字典
            lossRec = self.lossRec(outtaskrcgC, label)

            # 定位网络处理 - 使用第一个特征进行定位
            outtaskLocR, outtaskLocD = self.localn(x[self.features[0]])
            lossLocR = torch.sum((Rr - outtaskLocR) ** 2) 
            lossLocD = torch.sum((Sz - outtaskLocD) ** 2)

            # 计算 c_tau
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2

            # 计算每个任务的加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]
            
            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum()

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss
            elif self.task_type == 'classification':
                mtl_loss = lossRec
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum()

            return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared
        else:
            # 原有网络处理
            x = x[self.features[0]]
            x = x.unsqueeze(1)  # 添加额外的维度，形状变为 (batch_size, 1, input_size)
            # 复制通道维度以匹配ResNet18的输入要求
            x = x.repeat(1, 3, 1, 1)
                
            outtaskrcgC = self.classn(x)
            lossRec = self.lossRec(outtaskrcgC, label)

            outtaskLocRD = self.localn(x)
            outtaskLocR, outtaskLocD = outtaskLocRD[:, 0], outtaskLocRD[:, 1]
            lossLocR = torch.sum((Rr - outtaskLocR) ** 2) 
            lossLocD = torch.sum((Sz - outtaskLocD) ** 2) 

            # 计算 c_tau
            c_tau = torch.exp(self.log_vars)
            c_tau_squared = c_tau ** 2

            # 计算每个任务的加权损失
            weighted_lossRec = 0.5 * lossRec / c_tau_squared[0]
            weighted_lossLocR = 0.5 * lossLocR / c_tau_squared[1]
            weighted_lossLocD = 0.5 * lossLocD / c_tau_squared[2]

            # 计算正则化项
            reg_loss = torch.log(1 + c_tau_squared).sum()

            if self.task_type == 'mtl':
                mtl_loss = weighted_lossRec + weighted_lossLocR + weighted_lossLocD + reg_loss
            elif self.task_type == 'classification':
                mtl_loss = lossRec
            elif self.task_type == 'localization':
                mtl_loss = weighted_lossLocR + weighted_lossLocD + torch.log(1 + c_tau_squared[1:]).sum()

            return mtl_loss, outtaskrcgC, outtaskLocR, outtaskLocD, c_tau_squared

if __name__ == '__main__':
    # 测试参数
    batch_size = 5
    feature_dim = {'mel': 200, 'stft': 513, 'cqt': 84, 'gfcc': 200, 'mfcc': 40}
    channels = 64
    embd_dim = 64
    num_classes = 5
    num_experts = 2
    k = 1
    features = ['gfcc', 'stft']
    
    # 测试MOE网络
    print("Testing MOE network...")
    model_moe = MultiTaskLossWrapper(
        model_name='meg',
        feature_dim=feature_dim,
        channels=channels,
        embd_dim=embd_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        k=k
    )
    
    inputs = {
        'gfcc': torch.randn(batch_size, feature_dim['gfcc'], 100),
        'stft': torch.randn(batch_size, feature_dim['stft'], 100)
    }
    label = torch.randint(0, num_classes, (batch_size,))
    Rr = torch.rand(batch_size)
    Sz = torch.rand(batch_size)
    
    loss, class_out, dist_out, depth_out, weights = model_moe(inputs, label, Rr, Sz)
    print(f"MOE - Total loss: {loss.item()}")
    print(f"MOE - Classification output shape: {class_out.shape}")
    print(f"MOE - Distance output shape: {dist_out.shape}")
    print(f"MOE - Depth output shape: {depth_out.shape}")
    print(f"MOE - Task weights: {weights}")
    
    # 测试原有网络
    print("\nTesting original network...")
    model_orig = MultiTaskLossWrapper(
        model_name='resnet18',
        feature_dim=feature_dim,
        channels=channels,
        embd_dim=embd_dim,
        num_classes=num_classes
    )
    
    loss, class_out, dist_out, depth_out, weights = model_orig(inputs, label, Rr, Sz)
    print(f"Original - Total loss: {loss.item()}")
    print(f"Original - Classification output shape: {class_out.shape}")
    print(f"Original - Distance output shape: {dist_out.shape}")
    print(f"Original - Depth output shape: {depth_out.shape}")
    print(f"Original - Task weights: {weights}")

    # 测试meg_mix网络
    print("\nTesting meg_mix network...")
    model_mix = MultiTaskLossWrapper(
        model_name='meg_mix',
        feature_dim=feature_dim,
        channels=channels,
        embd_dim=embd_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        k=k,
        features=features
    )
    
    loss, class_out, dist_out, depth_out, weights = model_mix(inputs, label, Rr, Sz)
    print(f"meg_mix - Total loss: {loss.item()}")
    print(f"meg_mix - Classification output shape: {class_out.shape}")
    print(f"meg_mix - Distance output shape: {dist_out.shape}")
    print(f"meg_mix - Depth output shape: {depth_out.shape}")
    print(f"meg_mix - Task weights: {weights}")
