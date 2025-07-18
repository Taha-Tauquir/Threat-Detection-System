# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 13:19:12 2025

@author: Hp
"""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

# === Model ===
class RadarBBoxWeaponNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = r3d_18(pretrained=False)
        self.backbone.stem[0] = nn.Conv3d(1, 64, (3, 7, 7), (1, 2, 2), (1, 3, 3), bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.bbox_head = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.Sigmoid()  # ⬅️ keeps bbox in [0,1]
        )
        self.weapon_head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        bbox = self.bbox_head(features)                     #  Already has sigmoid
        weapon = torch.sigmoid(self.weapon_head(features))  #  sigmoid needed here
        return bbox, weapon
