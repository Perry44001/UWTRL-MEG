import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 分类网络工厂
def create_network(model_name, input_size, channels, embd_dim, num_classes):
    if model_name == 'resnet18':
        backbone = models.resnet18(weights=None)
        # 添加两个stride=2的卷积层
        # backbone.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # 获取原始全连接层的输入特征数
        num_ftrs = backbone.fc.in_features
        # 替换原始的全连接层为适应五分类任务的新层
        backbone.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet50':
        backbone = models.resnet50(weights=None)
        # 添加两个stride=2的卷积层
        backbone.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'mobilenetv2':
        backbone = models.mobilenet_v2(weights=None)
        # 添加两个stride=2的卷积层
        backbone.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            *list(backbone.features.children())[1:]  # 保留原有的其他层
        )
        backbone.classifier[-1] = nn.Linear(1280, num_classes)
    elif model_name == 'densenet121':
        backbone = models.densenet121(weights=None)
        # 添加两个stride=2的卷积层
        backbone.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            *list(backbone.features.children())[1:]  # 保留原有的其他层
        )
        num_ftrs = backbone.classifier.in_features
        backbone.classifier = nn.Linear(num_ftrs, num_classes)
    # Swin Transformer实现示例
    elif model_name == 'swin':
        backbone = models.swin_t(weights=None)
        backbone.features[0][0] = nn.Sequential(    
            nn.Conv2d(3, 3, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        num_ftrs = backbone.head.in_features
        backbone.head = nn.Linear(num_ftrs, num_classes)
    
    # 新增ConvNeXt和VGG19支持
    elif model_name == 'convnext':
        backbone = models.convnext_tiny(weights=None)
        # 添加两个stride=2的卷积层
        backbone.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            *list(backbone.features.children())[1:]  # 保留原有的其他层
        )
        backbone.classifier[-1] = nn.Linear(768, num_classes)
    
    elif model_name == 'vgg16':
        backbone = models.vgg16(weights=None)
        # 添加两个stride=2的卷积层
        backbone.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            *list(backbone.features.children())[2:]  # 保留原有的其他层
        )
        # 使用全局平均池化和更小的全连接层
        backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),                  # 展平
            nn.Linear(512, num_classes)
        )
    
    elif model_name == 'vgg19':
        backbone = models.vgg19(weights=None)
        # 添加两个stride=2的卷积层
        backbone.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            *list(backbone.features.children())[2:]  # 保留原有的其他层
        )
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 使用全局平均池化和更小的全连接层
        backbone.classifier = nn.Sequential(
            nn.Flatten(),                  # 展平
            nn.Linear(512, num_classes)   # 1024 -> num_classes
        )
    
    elif model_name == 'new':

        backbone = ResidualFrequencyClassifier()


    return backbone


class new_network(nn.Module):
    def __init__(self, input_size, channels, embd_dim=192, model_name='meg'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7,3), stride=(1,1), padding=(3,1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7,3), stride=(1,1), padding=(3,1), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        
        # 初始化频率注意力模块
        self.freq_gate = FrequencyGate(gate_channels=32, reduction_ratio=16)

    def forward(self, input):
        x0 = input
        x = self.conv1(x0)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x0 = self.conv3(x0)
        x = x + x0
        
        # 应用频率注意力
        x = self.freq_gate(x)
        
        # 将x按照第3个维度（时间）平均
        fattention = x.mean(dim=3)
        
        # 后续处理保持不变
        fattention = fattention.squeeze(-1)
        fattention = nn.softmax(fattention, dim=-1)
        x = x * fattention.unsqueeze(-1).unsqueeze(-1)
        
        return x


class ResidualFrequencyClassifier(nn.Module):
    def __init__(self, input_channels=3, num_classes=5, base_channels=16):
        super(ResidualFrequencyClassifier, self).__init__()
        
        # 调用5个封装模块，前3个使用池化，后2个不使用以保持特征图尺寸
        self.block1 = ResidualFrequencyBlock(input_channels, base_channels, use_pooling=True)
        self.block2 = ResidualFrequencyBlock(base_channels, base_channels*2, use_pooling=True)
        self.block3 = ResidualFrequencyBlock(base_channels*2, base_channels*4, use_pooling=True)
        self.block4 = ResidualFrequencyBlock(base_channels*4, base_channels*4, use_pooling=False)
        self.block5 = ResidualFrequencyBlock(base_channels*4, base_channels*4, use_pooling=False)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels*4, num_classes)
        
    def forward(self, x):
        # 依次通过5个残差频率模块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # 分类头
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class ResidualFrequencyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(7,3), stride=1, 
                 reduction_ratio=16, use_pooling=True):
        super(ResidualFrequencyBlock, self).__init__()
        self.stride = stride
        self.use_pooling = use_pooling
        
        # 卷积层 (保持每个模块3个卷积层)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=(kernel_size[0]//2, kernel_size[1]//2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                              padding=(kernel_size[0]//2, kernel_size[1]//2))
        
        # 频率注意力机制
        self.freq_attention = FrequencyGate(out_channels, reduction_ratio)
        
        # 残差连接适配
        if stride != 1 or in_channels != out_channels:
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.bn_res = nn.BatchNorm2d(out_channels)
        else:
            self.conv_res = None
            
        # 平均池化层 (可选)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2) if use_pooling else None
        
    def forward(self, x):
        residual = x
        
        # 卷积块
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # 残差连接
        if self.conv_res is not None:
            residual = self.bn_res(self.conv_res(residual))
        out += residual
        out = F.relu(out)
        # 频率注意力
        out = self.freq_attention(out)
        
        # 可选平均池化
        if self.avg_pool is not None:
            out = self.avg_pool(out)
        
        return out





class FrequencyGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(FrequencyGate, self).__init__()
        self.gate_channels = gate_channels
        self.pool_types = pool_types
        
        # 使用1x1卷积替代全连接层，适应频率维度动态变化
        self.conv_reduce = nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1)
        self.conv_expand = nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, channels, frequency, time)
        freq_att_sum = None
        
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # 在频率维度(第2维)上进行平均池化
                avg_pool = F.avg_pool2d(x, kernel_size=(1, x.size(3)), stride=(1, x.size(3)))  # (B,C,F,1)

                freq_att = self.conv_expand(self.relu(self.conv_reduce(avg_pool)))
            elif pool_type == 'max':
                # 在频率维度上进行最大池化
                max_pool = F.max_pool2d(x, kernel_size=(1, x.size(3)), stride=(1, x.size(3)))  # (B,C,F,1)
                freq_att = self.conv_expand(self.relu(self.conv_reduce(max_pool)))
            
            # 将注意力权重扩展到原始频率维度
            freq_att = freq_att.expand(-1, -1, -1, x.size(3))  # (B,C,F,T)
            # print('freq_att', freq_att)
            
            if freq_att_sum is None:
                freq_att_sum = freq_att
            else:
                freq_att_sum = freq_att_sum + freq_att
        
        # 应用sigmoid激活并加权
        scale = F.sigmoid(freq_att_sum)
        return x * scale