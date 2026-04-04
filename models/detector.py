import torch
import torch.nn as nn
from .components import CViTFeatureExtractor, SpatialTransformerBlock, Conv1DClassificationHead

class HierarchicalDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = CViTFeatureExtractor()
        self.attention_block = SpatialTransformerBlock(embed_dim=512, num_heads=8, depth=2)
        self.classifier_head = Conv1DClassificationHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        cnn_features = self.feature_extractor(x)
        attention_features = self.attention_block(cnn_features)
        logits, features = self.classifier_head(attention_features)
        return logits, features