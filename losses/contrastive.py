import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        dot_product = torch.mm(features, features.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0) 
        distances = torch.sqrt(distances + 1e-8)    

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float()
        
        match_loss = mask * distances.pow(2)
        non_match_loss = (1.0 - mask) * F.relu(self.margin - distances).pow(2)
        
        loss = match_loss.sum() + non_match_loss.sum()
        return loss / (features.size(0) * features.size(0))