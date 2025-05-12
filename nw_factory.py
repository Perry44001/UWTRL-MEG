import torch.nn as nn
from torchvision import models

# 分类网络工厂
def create_class_network(model_name, input_size, channels, embd_dim, num_classes):
    if model_name == 'resnet18':
        backbone = models.resnet18(weights=None)
        # 添加两个stride=2的卷积层
        backbone.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
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
    
    return backbone

class LocalNetwork(nn.Module):
    def __init__(self, backbone, embd_dim):
        super().__init__()
        self.backbone = backbone
        self.relu = nn.ReLU()
        self.distance_head = nn.Linear(embd_dim, 1)
        self.depth_head = nn.Linear(embd_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        distance = self.distance_head(x)
        depth = self.depth_head(x)
        return distance, depth

# 定位网络工厂
def create_local_network(model_name, input_size, channels, embd_dim):
    if model_name == 'resnet18':
        backbone = models.resnet18(weights=None)
        # 恢复ResNet原始Conv2d结构
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Linear(512, embd_dim)
    elif model_name == 'resnet50':
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Linear(2048, embd_dim)
    elif model_name == 'vgg16':
        backbone = models.vgg16(weights=None)
        backbone.features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)        
        backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embd_dim)
        )
    elif model_name == 'mobilenetv2':
        backbone = models.mobilenet_v2(weights=None)
        backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        backbone.classifier[-1] = nn.Linear(1280, embd_dim)
    elif model_name == 'densenet121':
        backbone = models.densenet121(weights=None)
        backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = backbone.classifier.in_features
        backbone.classifier = nn.Linear(num_ftrs, 1)
    # Swin Transformer实现示例
    elif model_name == 'swin':
        backbone = models.swin_t(weights=None)
        backbone.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        backbone.head = nn.Linear(768, embd_dim)
    # 新增ConvNeXt和VGG19支持
    elif model_name == 'convnext':
        backbone = models.convnext_tiny(weights=None)
        backbone.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        backbone.classifier[-1] = nn.Linear(768, embd_dim)
    elif model_name == 'vgg19':
        backbone = models.vgg19(weights=None)
        # 使用全局平均池化和更小的全连接层
        backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),                  # 展平
            nn.Linear(512, 1024),          # 512 -> 1024
            nn.ReLU(True),
            nn.Dropout(0.5),              # 增加dropout比例
            nn.Linear(1024, embd_dim)       # 1024 -> embd_dim
        )
    return LocalNetwork(backbone, embd_dim)