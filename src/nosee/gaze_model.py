"""Modified GazeMap gaze-network architecture, kept separate from its weights.

Copyright 2026 Fuad Mammadov. Based on
https://github.com/fammad/GazeMap; see third_party/GazeMap/LICENSE and NOTICE.
"""

from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation
from torch import nn


class GazeNet(nn.Module):
    def __init__(self, conv1: int = 20, conv2: int = 50, hidden: int = 500, use_head_pose: bool = True):
        super().__init__()
        self.use_head_pose = use_head_pose
        self.conv1 = nn.Conv2d(1, conv1, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1, conv2, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(conv2 * 6 * 12, hidden)
        self.fc2 = nn.Linear(hidden + (2 if use_head_pose else 0), 2)

    def forward(self, image: torch.Tensor, head_pose: torch.Tensor | None = None) -> torch.Tensor:
        value = self.pool(torch.relu(self.conv1(image)))
        value = self.pool(torch.relu(self.conv2(value)))
        value = value.flatten(start_dim=1)
        value = torch.relu(self.fc1(value))
        if self.use_head_pose:
            if head_pose is None:
                raise ValueError("This gaze model requires the head pose vector.")
            value = torch.cat([value, head_pose], dim=1)
        return self.fc2(value)


def pose_to_angles(rotation_vector):
    forward = Rotation.from_rotvec(rotation_vector).as_matrix()[:, :, 2]
    # Keep NumPy output, matching the upstream geometry convention.
    import numpy as np

    pitch = np.arcsin(np.clip(forward[:, 1], -1.0, 1.0))
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    return np.stack([pitch, yaw], axis=1)
