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

logging = setup_logger()

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
        

    hook_handle = model.model.features.register_forward_hook(hook)
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
    output.sigmoid_()
    print(output)
    # print("Interposed Layer 1 Activations:\n", captured_outputs["last_conv"])
    

    hook_handle.remove()

    feature_map = captured_outputs["last_conv"].squeeze(0)  # (1024, 7, 7)
    feature_map = feature_map.cpu().numpy()   

    classifier_weights = model.model.classifier.weight.data   # (14, 1024)
    weights = classifier_weights[target_class_idx].cpu().numpy()

    # print(feature_map.shape)
    # print(feature_map.size)

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

    cam_uint8 = (cam_norm * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # alpha = 0.4 means 60% original Xray and 40% colour map
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    # Convert back to RGB for matplotlib display
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb, heatmap_rgb

def visualize_prediction(model, image_tensor, true_labels,
                          device, target_class_idx=6, # for Pnemonia
                          save_path=None):
    batched = image_tensor.unsqueeze(0)

    cam_map, target_prob = generate_cam(model, batched, target_class_idx, device)

    # Get all 14 predictions in one forward pass
    # model.eval()
    # with torch.no_grad():
    #     all_probs = model(batched.to(device))[0].cpu().numpy()  # (14,)

    # Generate the overlay image
    overlay_img, _ = overlay_cam_image(image_tensor, cam_map, alpha=0.4)

    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    orig_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np * std + mean, 0, 1)


    fig = plt.figure(figsize=(18, 7))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.3)

    # ── Panel 1: Original X-ray ───────────────────────────────────────────────

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(orig_np, cmap="gray")
    ax1.set_title("Original X-ray", fontsize=13, fontweight="bold", pad=10)
    ax1.axis("off")

    # ── Panel 2: CAM overlay ─────────────────────────────────────────────────

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(overlay_img)

    disease_name = ALL_DISEASES[target_class_idx]
    true_val     = int(true_labels[target_class_idx])
    true_str     = "Present ✓" if true_val == 1 else "Absent ✗"
    true_colour  = "#2E7D32" if true_val == 1 else "#C62828"

    ax2.set_title(
        f"CAM — {disease_name}\nPred: {target_prob:.2%}  |  Ground truth: ",
        fontsize=12, fontweight="bold", pad=10
    )
    # Add coloured ground truth annotation
    ax2.text(0.5, -0.05, true_str,
             transform=ax2.transAxes,
             ha="center", fontsize=11,
             color=true_colour, fontweight="bold")
    ax2.axis("off")

    # Add a colourbar so the reader knows what red/blue means
    sm = plt.cm.ScalarMappable(cmap="jet",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("Attention", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    # ── Panel 3: Prediction bar chart ────────────────────────────────────────

    # ax3 = fig.add_subplot(gs[2])

    # # Sort diseases by predicted probability (highest at top)
    # sorted_idx   = np.argsort(all_probs)[::-1]
    # sorted_probs = all_probs[sorted_idx]
    # sorted_names = [ALL_DISEASES[i] for i in sorted_idx]
    # sorted_truth = [int(true_labels[i]) for i in sorted_idx]

    # # Colour each bar:
    # #   green  = low probability (< 0.3)
    # #   orange = medium          (0.3 – 0.6)
    # #   red    = high            (> 0.6)
    # bar_colours = []
    # for p in sorted_probs:
    #     if p >= 0.6:
    #         bar_colours.append("#C62828")   # red — high risk
    #     elif p >= 0.3:
    #         bar_colours.append("#E65100")   # orange — moderate
    #     else:
    #         bar_colours.append("#2E7D32")   # green — low risk

    # y_pos = np.arange(len(ALL_DISEASES))
    # bars = ax3.barh(y_pos, sorted_probs, color=bar_colours,
    #                 height=0.65, edgecolor="none", alpha=0.85)

    # # Decision threshold line at 0.5
    # ax3.axvline(x=0.5, color="black", linestyle="--",
    #             linewidth=1.2, alpha=0.5, label="Threshold (0.5)")

    # # Star marker for ground truth positive diseases
    # for i, (truth, prob) in enumerate(zip(sorted_truth, sorted_probs)):
    #     if truth == 1:
    #         ax3.text(min(prob + 0.02, 0.95), i, "★",
    #                  va="center", fontsize=10, color="#1565C0",
    #                  fontweight="bold")

    # # Probability values at end of each bar
    # for i, (prob, bar) in enumerate(zip(sorted_probs, bars)):
    #     ax3.text(prob + 0.01, i, f"{prob:.2f}",
    #              va="center", ha="left", fontsize=8,
    #              color="dimgray")

    # ax3.set_yticks(y_pos)
    # ax3.set_yticklabels(sorted_names, fontsize=9)
    # ax3.set_xlim(0, 1.12)
    # ax3.set_xlabel("Predicted probability", fontsize=10)
    # ax3.set_title("All 14 Disease Predictions\n★ = ground truth positive",
    #               fontsize=12, fontweight="bold", pad=10)
    # ax3.legend(fontsize=8, loc="lower right")
    # ax3.spines["top"].set_visible(False)
    # ax3.spines["right"].set_visible(False)
    # ax3.invert_yaxis()   # highest probability at top

    # ── Overall figure title ─────────────────────────────────────────────────

    fig.suptitle(
        "CheXNet — Prediction Visualisation with Class Activation Map",
        fontsize=14, fontweight="bold"
    )

    # ── Save if path provided ─────────────────────────────────────────────────

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        logging.info(f"Visualisation saved → {save_path}")

    return fig

def load_single_image(image_path):
    """
    Load one X-ray from disk, apply inference transforms,
    and return a tensor ready to pass to generate_cam().

    Returns:
        image_tensor : torch.Tensor (3, 224, 224) — NOT batched
    """
    img = Image.open(image_path).convert("RGB")
    return INFERENCE_TRANSFORM(img)
