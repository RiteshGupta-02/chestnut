import numpy as np
from PIL import Image

# 1. Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image_tensor(img):
    # img = Image.open("../dataset/images/00000001_000.png").convert("RGB")

    # 3. Step 1: transforms.Resize(256)
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * (256 / w))
    else:
        new_h = 256
        new_w = int(w * (256 / h))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 4. Step 2: transforms.CenterCrop(224)
    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = (new_w + 224) / 2
    bottom = (new_h + 224) / 2
    img_cropped = img_resized.crop((left, top, right, bottom))

    # 5. Step 3: transforms.ToTensor() -> Scale & Transpose to (C, H, W)
    img_array = np.array(img_cropped, dtype=np.float32) / 255.0
    tensor_like = np.transpose(img_array, (2, 0, 1)) # Shape becomes (3, 224, 224)

    # 6. Step 4: transforms.Normalize()
    # Reshape means/stds to (3, 1, 1) for proper broadcast alignment
    mean = IMAGENET_MEAN[:, np.newaxis, np.newaxis]
    std = IMAGENET_STD[:, np.newaxis, np.newaxis]
    tensor_normalized = (tensor_like - mean) / std

    # 7. Add Batch Dimension -> (1, 3, 224, 224)
    final_tensor = np.expand_dims(tensor_normalized, axis=0)
    return final_tensor

def stable_sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )

