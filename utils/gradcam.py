import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer1, target_layer2):
        self.model = model
        self.t1 = target_layer1
        self.t2 = target_layer2

        self.acts = {"x1": None, "x2": None}
        self.grads = {"x1": None, "x2": None}
        self._register_hooks()

    def _register_hooks(self):
        def fwd_x1(m, i, o): self.acts["x1"] = o
        def bwd_x1(m, gi, go): self.grads["x1"] = go[0]
        def fwd_x2(m, i, o): self.acts["x2"] = o
        def bwd_x2(m, gi, go): self.grads["x2"] = go[0]

        self.t1.register_forward_hook(fwd_x1)
        self.t1.register_full_backward_hook(bwd_x1)
        self.t2.register_forward_hook(fwd_x2)
        self.t2.register_full_backward_hook(bwd_x2)

    def _make_cam(self, acts, grads, out_size):
        # GAP over spatial dims -> channel weights
        w = grads.mean(dim=(2, 3), keepdim=True)          # (B,C,1,1)
        cam = (w * acts).sum(dim=1, keepdim=True)         # (B,1,H,W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
        cam = cam[:, 0]                                   # (B,H,W)

        # normalize per-sample
        b = cam.shape[0]
        flat = cam.view(b, -1)
        mn = flat.min(dim=1)[0].view(b, 1, 1)
        mx = flat.max(dim=1)[0].view(b, 1, 1)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.detach()

    @torch.enable_grad()
    def generate(self, x1, x2):
        """
        반환:
          cam1: (H,W) [0,1]
          cam2: (H,W) [0,1]
          dcam: (H,W) [-1,1]  (cam2 - cam1 정규화)
          prob, pred
        """
        self.model.eval()

        x1 = x1.requires_grad_(True)
        x2 = x2.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        self.acts["x1"]=self.acts["x2"]=None
        self.grads["x1"]=self.grads["x2"]=None

        out = self.model(x1, x2)       # (B,1)
        logit = out[:, 0]              # (B,)
        score = logit                  # 클래스 구분 없이 그냥 모델 score로 CAM

        score.sum().backward()

        cam1 = self._make_cam(self.acts["x1"], self.grads["x1"], out_size=x1.shape[2:])[0]
        cam2 = self._make_cam(self.acts["x2"], self.grads["x2"], out_size=x2.shape[2:])[0]

        prob = torch.sigmoid(logit.detach())[0].item()
        pred = int(prob >= 0.5)

        return cam1, cam2, prob, pred
    

# cam overlay
def overlay_cam_on_gray(gray_img, cam, alpha=0.4):
    """
    gray_img: (H,W) uint8 or float
    cam: (H,W) torch [0,1]
    """
    if gray_img.dtype != np.uint8:
        g = (255 * (gray_img / (gray_img.max() + 1e-8))).astype(np.uint8)
    else:
        g = gray_img

    cam_np = cam.detach().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)

    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(heatmap, alpha, g3, 1 - alpha, 0)
    return overlay 