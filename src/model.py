# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    # Based on https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

# Define Convolutional Block Attention Module (CBAM)
class CBAM(nn.Module):
    # Based on https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        # Channel Attention Module
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention Module
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_avg_pool(x)
        max_out = self.channel_max_pool(x)
        channel_out = avg_out + max_out
        channel_out = self.channel_fc(channel_out)
        channel_out = self.channel_sigmoid(channel_out)
        x = x * channel_out
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_conv(spatial_out)
        spatial_out = self.spatial_sigmoid(spatial_out)
        x = x * spatial_out
        
        return x

# Define Focal Loss
class FocalLoss(nn.Module):
    # Based on https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    def __init__(self, alpha=1, gamma=2, logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Balancing factor
        self.gamma = gamma  # Focusing parameter
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        p = torch.sigmoid(inputs)
        pt = p * targets - (1 - p) * (1 - targets)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss.sum()

# Define the WildfireSegmentation Model
class WildfireSegmentation(nn.Module):
    # Extracted from the course slides (U-model)
    def __init__(self, num_classes=1, input_channels=3, use_se=False, use_cbam=False):
        super(WildfireSegmentation, self).__init__()
        self.use_se = use_se
        self.use_cbam = use_cbam

        # Encoder path
        self.encoder1 = self.conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if self.use_se:
            layers.append(SEBlock(out_channels))

        if self.use_cbam:
            layers.append(CBAM(out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder path
        u4 = self.upconv4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.decoder4(u4)
        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.decoder3(u3)
        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.decoder2(u2)
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.decoder1(u1)

        # Final Convolution
        out = self.final_conv(d1)

        return out