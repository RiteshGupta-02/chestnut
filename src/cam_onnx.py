"""
cam_onnx.py — CAM generation using ONNX Runtime
No PyTorch needed at runtime. No hooks needed.
"""

import cv2
import numpy as np
from PIL import Image
import io
import matplotlib
matplotlib.use("Agg")          # no display needed on server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ALL_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → numpy array ready for ONNX Runtime.

    ONNX Runtime expects numpy arrays, not PyTorch tensors.
    Shape must be (1, 3, 224, 224) float32.

    Args:
        image_bytes: raw bytes from file upload

    Returns:
        numpy array (1, 3, 224, 224) float32
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)

    # Centre crop to 224×224
    left = (256 - 224) // 2   # = 16
    img  = img.crop((left, left, left + 224, left + 224))

    # → numpy, normalise
    arr  = np.array(img, dtype=np.float32) / 255.0          # (224, 224, 3)
    arr  = (arr - IMAGENET_MEAN) / IMAGENET_STD             # normalise
    arr  = arr.transpose(2, 0, 1)                           # → (3, 224, 224)
    arr  = np.expand_dims(arr, axis=0)                      # → (1, 3, 224, 224)
    return arr.astype(np.float32)


def generate_cam_onnx(feature_session, classifier_weights,
                       image_array, target_class_idx):
    """
    Generate raw (7,7) CAM using ONNX feature extractor.

    This replaces the hook-based generate_cam() from cam.py.
    Instead of intercepting the forward pass, we run a separate
    ONNX model that stops at the feature map and returns it directly.

    Args:
        feature_session     : onnxruntime.InferenceSession for features model
        classifier_weights  : numpy (14, 1024) — loaded from .npy file
        image_array         : numpy (1, 3, 224, 224) float32
        target_class_idx    : int 0-13

    Returns:
        cam_map  : numpy (7, 7) — raw importance values
        pred_prob: float — predicted probability for target disease
    """

    # Run feature extractor → (1, 1024, 7, 7)
    feature_map = feature_session.run(
        ["feature_map"],
        {"image": image_array}
    )[0]

    # Remove batch dimension → (1024, 7, 7)
    feature_map = feature_map.squeeze(0)

    # Get weights for target disease → (1024,)
    weights = classifier_weights[target_class_idx]

    # CAM = weighted sum over channels at each spatial position
    # Same math as before — just numpy instead of torch
    cam_map = np.dot(
        weights,                              # (1024,)
        feature_map.reshape(1024, -1)         # (1024, 49)
    ).reshape(7, 7)                           # → (7, 7)

    # We need pred_prob too — run the full model
    # Note: feature_session only gives feature map, not predictions
    # pred_prob is passed in from the /cam endpoint which already has it
    # So we return cam_map and compute prob outside
    return cam_map


def overlay_cam_on_image(image_array, cam_map, alpha=0.4):
    """
    Same logic as before — now takes numpy array instead of torch tensor.

    Args:
        image_array : numpy (1, 3, 224, 224) float32 — normalised
        cam_map     : numpy (7, 7) — raw CAM
        alpha       : heatmap opacity

    Returns:
        overlay_rgb : numpy (224, 224, 3) uint8
    """

    # Undo normalisation → displayable image
    img = image_array.squeeze(0)              # (3, 224, 224)
    img = img.transpose(1, 2, 0)             # (224, 224, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize 7×7 → 224×224
    cam_resized = cv2.resize(cam_map, (224, 224))

    # Normalise to 0–255
    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam_resized - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = np.zeros_like(cam_resized)
    cam_uint8 = (cam_norm * 255).astype(np.uint8)

    # Apply colormap + blend
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def build_cam_response(image_array, cam_map, predictions, target_disease):
    """
    Build a clean three-panel figure and return it as PNG bytes.
    Suitable for returning directly from a FastAPI StreamingResponse.

    Args:
        image_array    : numpy (1, 3, 224, 224)
        cam_map        : numpy (7, 7)
        predictions    : dict {disease: probability}
        target_disease : str — which disease is being explained

    Returns:
        bytes — PNG image
    """
    overlay = overlay_cam_on_image(image_array, cam_map, alpha=0.4)

    # Original image for display
    orig = image_array.squeeze(0).transpose(1, 2, 0)
    orig = np.clip(orig * IMAGENET_STD + IMAGENET_MEAN, 0, 1)

    # ── Figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6), facecolor="#0e1320")
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)

    text_color = "#e8edf5"
    grid_color = "#1e2d47"

    # Panel 1 — original
    # ax1 = fig.add_subplot(gs[0])
    # ax1.imshow(orig, cmap="gray")
    # ax1.set_title("Original X-Ray", color=text_color,
    #               fontsize=11, fontweight="bold", pad=8)
    # ax1.axis("off")
    # ax1.set_facecolor("#080b12")

    # Panel 2 — CAM overlay
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(overlay)
    # ax2.set_title(f"CAM — {target_disease}", color=text_color,
    #               fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")
    # ax2.set_facecolor("#080b12")

    # Colourbar
    # sm = plt.cm.ScalarMappable(cmap="jet",
    #                             norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
    # cbar.set_label("Attention", color=text_color, fontsize=8)
    # cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)
    # cbar.set_ticks([0, 0.5, 1])
    # cbar.set_ticklabels(["Low", "Mid", "High"])

    # # Panel 3 — bar chart
    # ax3 = fig.add_subplot(gs[2])
    # ax3.set_facecolor("#080b12")

    # sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    # diseases     = [d.replace("_", " ") for d, _ in sorted_items]
    # probs        = [p for _, p in sorted_items]
    # y_pos        = np.arange(len(diseases))

    # colours = []
    # for p in probs:
    #     if p >= 0.6:   colours.append("#ef4444")
    #     elif p >= 0.3: colours.append("#f59e0b")
    #     else:          colours.append("#22c55e")

    # ax3.barh(y_pos, probs, color=colours, height=0.65, alpha=0.85)
    # ax3.axvline(x=0.5, color="#7a8ba8", linestyle="--",
    #             linewidth=1, alpha=0.6)
    # ax3.set_yticks(y_pos)
    # ax3.set_yticklabels(diseases, fontsize=8, color=text_color)
    # ax3.set_xlim(0, 1.1)
    # ax3.set_xlabel("Probability", color=text_color, fontsize=9)
    # ax3.tick_params(colors=text_color)
    # ax3.spines["top"].set_visible(False)
    # ax3.spines["right"].set_visible(False)
    # for spine in ["bottom", "left"]:
    #     ax3.spines[spine].set_color(grid_color)
    # ax3.set_title("All Predictions", color=text_color,
    #               fontsize=11, fontweight="bold", pad=8)
    # ax3.invert_yaxis()

    # fig.patch.set_facecolor("#0e1320")

    # ── Serialise to PNG bytes ─────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def stable_sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )