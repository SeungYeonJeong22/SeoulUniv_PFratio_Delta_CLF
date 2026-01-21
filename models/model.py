import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

"""
input: (paste cxr img, cur cxr img)
output: improve / deteriorate (Binary Classification)

Baseline Architecture:
Feed 2 images into Encoder which based on patch(ViT) 
Concat Each Encoders output feature vector
Pass through Fully Connected Layer
"""

# class PatchBasedTransformerEncoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
        
#         img_size=cfg.img_size
#         patch_size=cfg.patch_size
#         in_chans=cfg.in_chans
#         d_model=cfg.d_model
#         nhead=cfg.nhead
#         num_layers=cfg.num_layers
#         dropout=cfg.dropout     
        
#         assert img_size % patch_size == 0
#         self.img_size = img_size
#         self.patch_size = patch_size

#         # Patch Embedding
#         # (B,1,H,W) -> (B,d_model,H/P,W/P)
#         self.patch_embed = nn.Conv2d(
#             in_channels=in_chans,
#             out_channels=d_model,
#             kernel_size=patch_size,
#             stride=patch_size,
#             padding=0
#         )

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

#         # Positional Embedding
#         num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, d_model))
#         self.pos_drop = nn.Dropout(p=dropout)

#         # Transformer Encoder
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             batch_first=True,
#             dropout=dropout
#         )
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

#         self.norm = nn.LayerNorm(d_model)

#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         nn.init.trunc_normal_(self.cls_token, std=0.02)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         x = self.patch_embed(x)                 # (B, d_model, H/P, W/P)
#         x = x.flatten(2).transpose(1, 2)        # (B, (H/P*W/P), d_model)

#         cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
#         x = torch.cat([cls, x], dim=1)          # (B, 1+num_patches, d_model)

#         x = x + self.pos_embed[:, : x.size(1), :]
#         x = self.pos_drop(x)

#         x = self.encoder(x)                     # (B, 1+num_patches, d_model)
#         x = self.norm(x)

#         cls_feat = x[:, 0]                      # (B, d_model)
#         return cls_feat, x


def build_resnet50_encoder_from_moco(ckpt_path, in_chans=1, device="cpu"):
    model = models.resnet50(weights=None)

    if in_chans == 1:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            1, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # encoder_q만 뽑고 prefix 제거 + fc 제거
    new_state = {}
    for k, v in state.items():
        if k.startswith("module.encoder_q."):
            kk = k.replace("module.encoder_q.", "")
            # MoCo projection head는 보통 model.fc가 아니라 "fc.0, fc.2" 같은 MLP
            if kk.startswith("fc."):
                continue
            new_state[kk] = v
            
    if "conv1.weight" in new_state and in_chans == 1:
        w = new_state["conv1.weight"]
        if w.shape[1] == 3:
            new_state["conv1.weight"] = w.mean(dim=1, keepdim=True)            

    # encoder로만 쓰려면 fc 제거/무시
    model.fc = nn.Identity()

    return model


class DownStreamTaskModel(nn.Module):
    def __init__(self, cfg):
        super(DownStreamTaskModel, self).__init__()
        # self.enc = timm.create_model(
        #     "resnet50",
        #     pretrained=True,
        #     features_only=True,   
        #     in_chans=1            
        # )
        
        self.enc = build_resnet50_encoder_from_moco("./models/CheSS.pth.tar", in_chans=1)
        
        # enc = build_resnet50_encoder_from_moco("./models/CheSS.pth.tar", in_chans=1)
        
        d_model = cfg.d_model
        output_dim = cfg.output_dim
        
        # self.enc1 = PatchBasedTransformerEncoder(cfg)
        # self.enc2 = PatchBasedTransformerEncoder(cfg)
        
        self.fc = nn.Linear(4096, output_dim)
        
        
    def forward(self, x1, x2):
        # Extract features from both encoders
        # f1, _ = self.enc(x1)
        f1= self.enc(x1)
        # f2, _ = self.enc(x2)
        f2= self.enc(x2)
        
        # Concatenate features
        concated_x = torch.concat([f1, f2], dim=1)
        
        # FC Layer
        out = self.fc(concated_x)
        
        return out