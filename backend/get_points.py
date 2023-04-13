import urllib.request
import numpy as np
import torch
import torch.nn.functional as F

import os
from os.path import isfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import clip
from PIL import Image
from scipy import ndimage
from torch import nn
from skimage.feature import blob_log


def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def format_image(image, resize=None):
    """ image is a PIL image """
    image = image.convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam


def resize_peaks(peaks, model, image):
    """ Peaks contain coordinates in image of size model.visual.input_resolution.
        Resize them to the actual image size. """
    resized_peaks = []
    width, height = image.size
    for y, x, size in peaks:
        x = int(x / model.visual.input_resolution * width)
        y = int(y / model.visual.input_resolution * height)
        resized_peaks.append((y, x, size))
    return resized_peaks


def get_points(text_query, image, num_detections):
    """ Find (x, y) coordinates based on the text query and the image """
    clip_model = "RN50" #@param ["RN50", "RN101", "RN50x4", "RN50x16"]
    saliency_layer = "layer4" #@param ["layer4", "layer3", "layer2", "layer1"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH")
    if MODEL_WEIGHTS_PATH and isfile(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device(device)))

    image_input = preprocess(image).unsqueeze(0).to(device)
    image_np = format_image(image, model.visual.input_resolution)
    text_input = clip.tokenize([text_query]).to(device)
    
    attn_map = gradCAM(
        model.visual,
        image_input,
        model.encode_text(text_input).float(),
        getattr(model.visual, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    attn_map_log = attn_map.copy()
    attn_map = ndimage.gaussian_filter(attn_map, 0.02*max(image_np.shape[:2]))
    attn_map = normalize(attn_map)

    peaks = blob_log(attn_map)
    # sort peaks by the value of the attention map
    peaks = np.array(sorted(peaks.tolist(), key=lambda x: attn_map[int(x[0]), int(x[1])], reverse=True))
    if num_detections is not None:
        peaks = peaks[:num_detections]
    resized_peaks = resize_peaks(peaks, model, image)
    return [{"x": int(x), "y": int(y)} for y, x, _ in resized_peaks]
