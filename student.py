import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        # A smaller version of VGG-like backbone with fewer layers and filters
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.slice2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.slice3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.slice4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.slice5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]
class SimpleCRAFT(nn.Module):
    def __init__(self):
        super(SimpleCRAFT, self).__init__()
        # Backbone
        self.basenet = SimpleVGG()

        # U network
        self.upconv1 = double_conv(64 + 32, 32, 32)  # Reduced channels
        self.upconv2 = double_conv(48 + 24, 24, 8)
        self.upconv3 = double_conv(20 + 10, 10, 16)
        self.upconv4 = double_conv(8 + 4, 4, 4)

        # Classification head
        self.conv_cls = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, kernel_size=1)  # Output 2 channels: region and affinity maps
        )

    def forward(self, x):
        sources = self.basenet(x)

        # print(sources[0].size())
        # print(sources[1].size())
        # print(sources[2].size())
        # print(sources[3].size())

        y = self.upconv1(sources[3])
        # print("y", y.size())

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, sources[2]], dim=1)

        # print("y", y.size())
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, sources[1]], dim=1)

        # print("y", y.size())
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources[0].size()[2:], mode='bilinear', align_corners=True)
        feature = self.upconv4(y)



        y = self.conv_cls(feature)
        # print("Final", y.size())
        return y.permute(0, 2, 3, 1), feature


import torch
import torch.nn as nn


# Simpler double convolution layer with reduced channels
class simple_double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(simple_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Simplified CRAFT model
class mediumCRAFT(nn.Module):
    def __init__(self):
        super(mediumCRAFT, self).__init__()

        # Base VGG-like architecture (slice1 to slice4) with reduced channels
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.slice2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.slice3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.slice4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Reduced number of channels and layers in slice5
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Fewer filters than before
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, stride=1)  # Fewer parameters overall
        )

        # Simpler upsampling layers (double_conv layers) with reduced channels
        self.upconv1 = simple_double_conv(768, 256)  # Combining slice5 (512) and slice4 (256)
        self.upconv2 = simple_double_conv(384, 128)  # Combining upconv1 (256) and slice3 (128)
        self.upconv3 = simple_double_conv(192, 64)  # Combining upconv2 (128) and slice2 (64)
        self.upconv4 = simple_double_conv(96, 64)  # Combining upconv3 (64) and slice1 (32)

        # Final classification layer with reduced parameters
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1, stride=1)    ,
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # Encoder path (down-sampling)
        y1 = self.slice1(x)
        y2 = self.slice2(y1)
        y3 = self.slice3(y2)
        y4 = self.slice4(y3)
        y5 = self.slice5(y4)
        # print(y1.size(), y2.size(), y3.size(), y4.size(), y5.size(),)
        # Decoder path (up-sampling)
        up1 = self.upconv1(torch.cat([y5, y4], dim=1))

        up2 = self.upconv2(torch.cat([up1, y4], dim=1))
        up2 = F.interpolate(up2, size=y3.size()[2:], mode='bilinear', align_corners=False)
        up3 = self.upconv3(torch.cat([up2, y3], dim=1))
        up3 = F.interpolate(up3, size=y2.size()[2:], mode='bilinear', align_corners=False)
        up4 = self.upconv4(torch.cat([up3, y2], dim=1))
        up4 = F.interpolate(up4, size=y1.size()[2:], mode='bilinear', align_corners=False)
        # print(up4.size(), y1.size())
        # print(torch.cat([up4, y1]).size())
        # Final classification layer
        output = self.conv_cls(up4)

        return output.permute(0, 2, 3, 1), up4
