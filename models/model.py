import torch
import torch.nn as nn
import torch.nn.functional as F

"""
input: (paste cxr img, cur cxr img)
output: improve / deteriorate (Binary Classification)

Baseline Architecture:
Feed 2 images into Encoder
Concat Each Encoders output feature vector
Pass through Fully Connected Layer
"""

class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=256, k_size=3, d_model=256, nhead=4, num_layers=6):
        super(ImageEncoder, self).__init__()
        
        self.embed = nn.Conv2d(in_channels=1, out_channels=emb_dim, kernel_size=k_size, stride=1, padding=1)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.encoder(x)
        pass

class DownStreamTaskModel(nn.Module):
    def __init__(self, enc_dims, output_dim):
        super(DownStreamTaskModel, self).__init__()
        
        # enc_l1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # enc2_l2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        
        self.enc1 = ImageEncoder()
        self.enc2 = ImageEncoder()
        # self.enc1 = nn.TransformerEncoder(enc_l1, num_layers=6)
        # self.enc2 = nn.TransformerEncoder(enc2_l2, num_layers=6)
        
        
        self.fc = nn.Linear(1024, output_dim)
        
        
    def forward(self, x1, x2):
        # Extract features from both encoders
        f1 = self.enc1(x1)
        f2 = self.enc2(x2)
        
        # Concatenate features
        concated_x = torch.concat([f1, f2], dim=1)
        
        # FC Layer
        out = self.fc(concated_x)
        
        return out