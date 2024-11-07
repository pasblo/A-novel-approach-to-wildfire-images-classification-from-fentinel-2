# Import libraries
import torch
from torch import nn

class WildfireSegmentation(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the parent class

        # Encoder path
        self.encoder0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),  # Input channels: 3, output channels: 32
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.pool0 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),  # Input channels: 32, output channels: 64
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),  # Output channels: 128
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),  # Output channels: 256
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),  # Output channels: 512
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, padding = 1),  # Output channels: 1024
            nn.ReLU(inplace = True),
            nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        # Decoder path (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 3, padding = 1),  # Concatenate channels
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, padding = 1),  # Concatenate channels
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1),  # Concatenate channels
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1),  # Concatenate channels
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
        self.decoder0 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 3, padding = 1),  # Concatenate channels
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

        # Final convolution layer to map to the number of classes (1 in this case)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) # MOD

    def forward(self, x):
        # Encoder path
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)

        e1 = self.encoder1(p0)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder path (Upsampling)
        u4 = self.upconv4(b)
        c4 = torch.cat([u4, e4], dim = 1)
        d4 = self.decoder4(c4)

        u3 = self.upconv3(d4)
        c3 = torch.cat([u3, e3], dim = 1)
        d3 = self.decoder3(c3)

        u2 = self.upconv2(d3)
        c2 = torch.cat([u2, e2], dim = 1)
        d2 = self.decoder2(c2)

        u1 = self.upconv1(d2)
        c1 = torch.cat([u1, e1], dim = 1)
        d1 = self.decoder1(c1)

        u0 = self.upconv0(d1)
        c0 = torch.cat([u0, e0], dim = 1)
        d0 = self.decoder0(c0)

        # Final convolution layer
        out = self.final_conv(d0)
        
        # No sigmoid
        return out