
import numpy as np
import cv2

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = []  # -> x1, x2를 따로 보기 위해서 리스트로 저장
        self.gradients = []  # -> x1, x2를 따로 보기 위해서 리스트로 저장

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)


    def generate(self, x1, x2):
        self.activations.clear()
        self.gradients.clear()
        self.model.zero_grad(set_to_none=True)

        out = self.model(x1, x2)
        logit = out[:, 0]
        prob = torch.sigmoid(logit)[0].item()
        pred = int(prob >= 0.5)

        logit.backward()

        cams = []
        for acts, grads in zip(self.activations, self.gradients):
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x1.shape[2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0]
            cam = (cam - cam.min()) / (cam.max() + 1e-8)
            cams.append(cam.detach())

        return cams[0], cams[1], prob, pred

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
    
    