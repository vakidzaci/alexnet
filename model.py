import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Conv Layer 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max Pooling Layer 1

            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # Conv Layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max Pooling Layer 2

            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # Conv Layer 3
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Conv Layer 4
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Conv Layer 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max Pooling Layer 3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # FC Layer 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # FC Layer 2
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # Output Layer
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output from conv layers
        x = self.classifier(x)
        return x


# Example of model creation
model = AlexNet(num_classes=1000)  # 1000 is the default number of classes in ImageNet
