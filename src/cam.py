from model import get_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from logger import setup_logger
from chestxray_dataset import ALL_DISEASES

logging = setup_logger(__file__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def generate_cam(model, image_tensor, target_class_idx, device):
    captured_outputs = {}

    def hook(module, input, output):
        captured_outputs['last_conv'] = output.detach()

    hook_handle = model.layer1.register_forward_hook(hook())

    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

    print("Interposed Layer 1 Activations:\n", captured_outputs["layer1"])

    hook_handle.remove()

    feature_map = captured_outputs["last_conv"].squeeze(0)  # (1024, 7, 7)
    feature_map = feature_map.cpu().numpy()   

    classifier_weights = model.model.classifier.weight.data   # (14, 1024)
    weights = classifier_weights[target_class_idx].cpu().numpy()

    cam_map = np.dot(
        weights,                              # (1024,)
        feature_map.reshape(1024, -1)         # (1024, 49)
    ).reshape(7, 7)                           # → (7, 7)

    # The predicted probability for the target disease
    pred_prob = output[0, target_class_idx].item()

    disease_name = ALL_DISEASES[target_class_idx]
    logging.info(f"CAM generated for '{disease_name}' | pred prob: {pred_prob:.4f}")

    return cam_map, pred_prob

def overlay_cam_image(image_tensor, cam_map, alpha=0.4):

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    img_np = image_tensor.cpu().numpy()          # (3, 224, 224)
    img_np = img_np.transpose(1, 2, 0)           # (224, 224, 3) — HWC format for display
    img_np = (img_np * std + mean)               # undo normalisation
    img_np = np.clip(img_np, 0, 1)              # clamp to valid range
    img_np = (img_np * 255).astype(np.uint8)    # float → uint8

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # converting normal rgb to tensor bgr image format

    cam_resized = cv2.resize(cam_map, (224, 224))

    # getting the max and min value of intensity of colour to get every pixel coloured
    cam_min, cam_max = cam_resized.min(), cam_resized.max()

    #  to avoid division by zero
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_resized)