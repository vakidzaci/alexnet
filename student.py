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
            nn.Conv2d(4, 4, kernel_size=3, padding=1), nn.ReLU(inplace=True),
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
