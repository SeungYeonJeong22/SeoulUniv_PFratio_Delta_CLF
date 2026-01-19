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
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        d_model=256,
        nhead=4,
        num_layers=6,
        dropout=0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size

        # 1) Patch Embedding (ViT의 patchify + linear projection 역할)
        # (B,1,H,W) -> (B,d_model,H/P,W/P)
        self.patch_embed = nn.Conv2d(
            in_channels=in_chans,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # 2) CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3) Positional Embedding
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, d_model))
        self.pos_drop = nn.Dropout(p=dropout)

        # 4) Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # (선택) LayerNorm (ViT에서 흔히 사용)
        self.norm = nn.LayerNorm(d_model)

        # init (간단)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        x: (B, 1, H, W)  where H=W=img_size
        return:
          cls_feat: (B, d_model)  (분류/회귀용 대표 feature)
          tokens:   (B, 1+num_patches, d_model) (원하면 활용)
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)                 # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, d_model)

        # CLS token 붙이기
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, 1+num_patches, d_model)

        # Positional embedding
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        # Transformer
        x = self.encoder(x)                     # (B, 1+num_patches, d_model)
        x = self.norm(x)

        cls_feat = x[:, 0]                      # (B, d_model)
        return cls_feat, x


class DownStreamTaskModel(nn.Module):
    def __init__(self, enc_dims, output_dim):
        super(DownStreamTaskModel, self).__init__()
        
        # enc_l1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        # enc2_l2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        
        self.enc1 = ImageEncoder()
        self.enc2 = ImageEncoder()
        # self.enc1 = nn.TransformerEncoder(enc_l1, num_layers=6)
        # self.enc2 = nn.TransformerEncoder(enc2_l2, num_layers=6)
        
        
        self.fc = nn.Linear(512, output_dim)
        
        
    def forward(self, x1, x2):
        # Extract features from both encoders
        f1, _ = self.enc1(x1)
        f2, _ = self.enc2(x2)
        
        # Concatenate features
        concated_x = torch.concat([f1, f2], dim=1)
        
        # FC Layer
        out = self.fc(concated_x)
        
        return out