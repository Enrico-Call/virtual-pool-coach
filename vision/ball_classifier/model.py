from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

from game_model import BallStateType

LABELS: Tuple[BallStateType, ...] = ('solid', 'stripes', 'eight', 'cue')
LABEL_INDEX_MAP: Dict[BallStateType, int] = {label: index for index, label in enumerate(LABELS)}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# class LargeModel(nn.Module):
#     def __init__(self, class_count=len(LABELS)):
#         super().__init__()
#         self.class_count = class_count
#
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, self.class_count)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     @classmethod
#     def load(cls, file: Path) -> LargeModel:
#         instance = cls()
#         instance.load_state_dict(torch.load(file))
#         return instance
#
#     @classmethod
#     def load_as_base(cls, file: Path, base_class_count: int, new_class_count=len(LABELS)) -> LargeModel:
#         instance = cls(class_count=base_class_count)
#         instance.load_state_dict(torch.load(file))
#         instance.fc3 = nn.Linear(84, new_class_count)
#         instance.class_count = new_class_count
#         return instance


class SmallModel(nn.Module):
    def __init__(self, class_count=len(LABELS)):
        super().__init__()
        self.class_count = class_count

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5)
        self.fc1 = nn.Linear(6 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, self.class_count)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def load(cls, file: Path) -> SmallModel:
        instance = cls()
        instance.load_state_dict(torch.load(file))
        return instance
