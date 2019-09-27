import sys
import torch
from torch import nn


class ResNetv2_20(nn.Module):
    """ResNetv2 Model (same architecture as logit pairing for CIFAR10)
    """
    def __init__(self, num_classes):
        super(ResNetv2_20, self).__init__()
        self.filters = 64

        self.conv = nn.Conv2d(3, self.filters, 3, padding=1, bias=False)
        self.resblock1 = ResBlock(self.filters, bi_channel=False)
        self.resblock2 = ResBlock(self.filters*2)
        self.resblock3 = ResBlock(self.filters*4)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            View(-1, self.filters*4),
            nn.Linear(self.filters*4, num_classes)
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.classifier(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, base_c, bi_channel=True):
        super(ResBlock, self).__init__()
        if bi_channel:
            in_c = base_c // 2
            stride = 2
        else:
            in_c = base_c
            stride = 1

        self.conv = nn.Conv2d(in_c, base_c, 1, stride, 0, bias=False)
        self.block1 = nn.Sequential(
            *convblock(in_c=in_c, out_c=base_c, kernel_size=3,
                       stride=stride, padding=1, bias=False),
            *convblock(in_c=base_c, out_c=base_c,
                       kernel_size=3, padding=1, bias=False)
            )
        self.block2 = nn.Sequential(
            *convblock(in_c=base_c, out_c=base_c,
                       kernel_size=3, padding=1, bias=False),
            *convblock(in_c=base_c, out_c=base_c,
                       kernel_size=3, padding=1, bias=False)
            )
        self.block3 = nn.Sequential(
            *convblock(in_c=base_c, out_c=base_c,
                       kernel_size=3, padding=1, bias=False),
            *convblock(in_c=base_c, out_c=base_c,
                       kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        fmap_1 = self.block1(x) + self.conv(x)
        fmap_2 = self.block2(fmap_1) + fmap_1
        fmap_3 = self.block3(fmap_2) + fmap_2 
        return fmap_3



def convblock(in_c, out_c, kernel_size, stride=1, padding=0, use_bn=True, bias=True):
    """
    Returns convolution block
    """
    if use_bn:
        return [
            nn.BatchNorm2d(in_c, momentum=0.05),
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias),
            nn.ReLU(True)
        ]
    else:
        return [
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias),
            nn.ReLU(True)
        ]


class View(nn.Module):
    """Basic reshape module.
    """
    def __init__(self, *shape):
        """
        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """Reshapes tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: Flattened tensor.
        """
        return x.view(*self.shape)


def init_he(m):
    if type(m) in [nn.Linear or nn.Conv2d]:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

