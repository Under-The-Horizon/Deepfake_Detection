import torch
import torch.nn as nn
import torchvision.models as models

class CViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet34(weights='DEFAULT')
        

        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):

        return self.features(x)

class SpatialTransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, dropout=0.1,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        return self.transformer(x)

class Conv1DClassificationHead(nn.Module):
    def __init__(self, in_channels=512, num_classes=2):
        super().__init__()
        self.conv_sequence = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.conv_sequence(x) 
        features = torch.flatten(x, 1)
        logits = self.fc(features)            
        return logits, features