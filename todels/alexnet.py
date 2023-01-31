import torch
import torch.nn as nn

class Alexnet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.5, local_norm_size=5):
        super(self.__class__, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=local_norm_size, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=local_norm_size, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.LazyLinear(4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.network(x)
        out = self.fc(out)


class LigthAlexnet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.5, local_norm_size=5):
        super(self.__class__, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=local_norm_size, alpha=0.0001, beta=0.75, k=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.LazyLinear(120),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        out = self.network(x)
        out = self.fc(out)
        
