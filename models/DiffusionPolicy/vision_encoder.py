import torch
import torch.nn as nn
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, feature_dim=512, pretrained=False, weights_path=""):
        super(VisionEncoder, self).__init__()
        self.resnet = self._build_resnet(pretrained=pretrained, weights_path=weights_path)
        self.resnet.fc = nn.Identity()
        self.feature_dim = feature_dim
        self.projection = nn.Linear(512, feature_dim)

    @staticmethod
    def _build_resnet(pretrained=False, weights_path=""):
        if weights_path:
            resnet = models.resnet18(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            resnet.load_state_dict(state)
            print(f"[VisionEncoder] loaded local ResNet18 weights: {weights_path}")
            return resnet

        if pretrained:
            try:
                return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception as exc:
                print(f"[VisionEncoder] failed to load pretrained weights, using random init: {exc}")

        return models.resnet18(weights=None)

    def forward(self, x):
        """
        x: [batch, 3, height, width]
        """
        features = self.resnet(x) # [batch, 512]
        features = self.projection(features) # [batch, feature_dim]
        return features

class MultiViewEncoder(nn.Module):
    def __init__(self, feature_dim=224, pretrained=False, weights_path=""):
        super(MultiViewEncoder, self).__init__()
        # 两个视图共用一个编码器
        self.encoder = VisionEncoder(
            feature_dim=feature_dim,
            pretrained=pretrained,
            weights_path=weights_path,
        )
        self.total_feature_dim = feature_dim * 2

    def forward(self, front, hand):
        """
        front: [B, 3, H, W]
        hand: [B, 3, H, W]
        """
        f_feat = self.encoder(front)
        h_feat = self.encoder(hand)
        
        # 拼接特征: [batch, feature_dim * 2] = 192
        combined = torch.cat([f_feat, h_feat], dim=1)
        return combined
