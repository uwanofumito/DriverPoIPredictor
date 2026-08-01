"""ResNet18 gaze-regression architecture, vendored (trimmed to resnet18 only)
from yakhyo/gaze-estimation (MIT license), https://github.com/yakhyo/gaze-estimation

Outputs two logit vectors (yaw, pitch) over `num_bins` angle bins rather than
a single regressed angle — decode with decode_bins() below, matching that
project's training scheme on Gaze360 (bins=90, binwidth=4, angle=180).
"""

from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class GazeResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], layers: List[int], num_bins: int):
        super().__init__()
        self.in_channels = 64
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample, norm_layer)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc_yaw(x), self.fc_pitch(x)


def gaze_resnet18(num_bins=90) -> GazeResNet:
    return GazeResNet(BasicBlock, [2, 2, 2, 2], num_bins)


# Gaze360 training config from yakhyo/gaze-estimation (config.py)
GAZE360_BINS = 90
GAZE360_BINWIDTH = 4
GAZE360_ANGLE = 180


def decode_bins(yaw_logits: Tensor, pitch_logits: Tensor):
    """Bin logits -> continuous (yaw, pitch) in degrees, via softmax-weighted
    expectation over bin centers (same decode as the reference implementation)."""
    idx = torch.arange(GAZE360_BINS, device=yaw_logits.device, dtype=torch.float32)
    yaw = (torch.softmax(yaw_logits, dim=1) * idx).sum(dim=1) * GAZE360_BINWIDTH - GAZE360_ANGLE
    pitch = (torch.softmax(pitch_logits, dim=1) * idx).sum(dim=1) * GAZE360_BINWIDTH - GAZE360_ANGLE
    return yaw, pitch
