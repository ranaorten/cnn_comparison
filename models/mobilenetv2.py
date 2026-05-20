import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        self.conv = nn.Sequential(
            # 1. Pointwise conv - genişlet
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # 2. Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # 3. Pointwise conv - daralt
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)        

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # (expand_ratio, out_channels, num_blocks, stride)
        self.cfgs = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        layers = []
        in_channels = 32
        for expand_ratio, out_channels, num_blocks, stride in self.cfgs:
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(InvertedResidual(in_channels, out_channels, s, expand_ratio))
                in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MobileNetV2(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Çıktı shape:", out.shape)
    print("Parametre sayısı:", sum(p.numel() for p in model.parameters()))
