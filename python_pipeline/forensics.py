"""Supporting forensic analyzers (2D FFT, ELA, Laplacian noise) and Grad-CAM."""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def encode_png_b64(rgb_or_bgr: np.ndarray, is_bgr: bool = False) -> str:
    arr = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_BGR2RGB) if is_bgr else rgb_or_bgr
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def compute_gradcam_heatmap(
    model: nn.Module, input_tensor: torch.Tensor, target_layer: nn.Module, out_size: int = 224
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    fwd_hook = target_layer.register_forward_hook(lambda _m, _i, out: activations.append(out.detach()))
    bwd_hook = target_layer.register_full_backward_hook(
        lambda _m, _gi, grad_out: gradients.append(grad_out[0].detach())
    )

    try:
        model.zero_grad()
        logit = model(input_tensor).squeeze()
        logit.backward()
        act = activations[0][0]
        grad = gradients[0][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0)).cpu().numpy()
        cam = cv2.resize(cam, (out_size, out_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)
        return heatmap_bgr
    finally:
        fwd_hook.remove()
        bwd_hook.remove()


def compute_supporting_forensics(image_rgb: np.ndarray) -> Dict:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. 2D FFT Frequency Spectrum
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log1p(np.abs(f))
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fft_bgr = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)

    # 2. Error Level Analysis (JPEG quality 90 residual)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr, encode_param)
    recomp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela_diff = cv2.absdiff(bgr, recomp).astype(np.float32) * 12.0
    ela_gray = np.clip(ela_diff.mean(axis=2), 0, 255).astype(np.uint8)
    ela_bgr = cv2.applyColorMap(ela_gray, cv2.COLORMAP_INFERNO)

    # 3. Laplacian local texture/noise inconsistency
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    noise_bgr = cv2.applyColorMap(lap_norm, cv2.COLORMAP_INFERNO)

    return {
        "fft_spectrum_data_url": encode_png_b64(fft_bgr, is_bgr=True),
        "ela_heatmap_data_url": encode_png_b64(ela_bgr, is_bgr=True),
        "noise_map_data_url": encode_png_b64(noise_bgr, is_bgr=True),
        "laplacian_variance": round(float(np.var(lap)), 3),
        "ela_mean_residual": round(float(ela_diff.mean() / 12.0), 3),
    }
