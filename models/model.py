import torch
import torch.nn as nn
import torchvision.models as models
"""
input: (paste cxr img, cur cxr img)
output: improve / deteriorate (Binary Classification)

Baseline Architecture:
Feed 2 images into Encoder which based on patch
Concat Each Encoders output feature vector
Pass through Fully Connected Layer
"""

def build_encoder(ckpt_path, in_chans=1, device="cpu"):
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

    if "CheSS" in ckpt_path:
        new_state = {}
        for k, v in state.items():
            if k.startswith("module.encoder_q."):
                kk = k.replace("module.encoder_q.", "")
                if kk.startswith("fc."):
                    continue
                new_state[kk] = v
                
        if "conv1.weight" in new_state and in_chans == 1:
            w = new_state["conv1.weight"]
            if w.shape[1] == 3:
                new_state["conv1.weight"] = w.mean(dim=1, keepdim=True)            

    else:    
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}

        if "conv1.weight" in state and in_chans == 1:
            w = state["conv1.weight"]
            if w.shape[1] == 3:
                state["conv1.weight"] = w.mean(dim=1, keepdim=True)
                
        new_state = state
    
    # encoder로만 쓰려면 fc 제거/무시
    model.fc = nn.Identity()
    
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    model.to(device)

    return model


class DownStreamTaskModel(nn.Module):
    def __init__(self, cfg):
        super(DownStreamTaskModel, self).__init__()
        pretrained_model_path = getattr(cfg, "pretrained_model_path", None)
        self.enc = build_encoder(pretrained_model_path, in_chans=1)
        self.freeze_encoder_weights()
        
        output_dim = cfg.model.output_dim
        self.fc = nn.Linear(4096, output_dim)
        
    
    def freeze_encoder_weights(self):
        for param in self.enc.parameters():
            param.requires_grad = False
        
        
    def forward(self, x1, x2):
        # Extract features from both encoders
        f1 = self.enc(x1)
        f2 = self.enc(x2)
        
        # Concatenate features
        concated_x = torch.concat([f1, f2], dim=1)
        
        # FC Layer
        out = self.fc(concated_x)
        
        return out