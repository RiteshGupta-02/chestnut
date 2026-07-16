"""
NIH ChestX-ray14 Dataset — PyTorch Dataset class
Multi-label classification with augmentation for CheXNet replication

Expected directory structure:
    data/
    ├── images/                   # all unzipped .png X-ray images
    ├── Data_Entry_2017.csv       # main labels file
    ├── train_val_list.txt        # official train split
    └── test_list.txt             # official test split

Usage:
    from chestxray_dataset import ChestXray14Dataset, get_transforms, get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders('data/')
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─── Label definitions ──────────────────────────────────────────────────────

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
# NOTE: "No Finding" is intentionally excluded from the 14 output labels.
# When all 14 probabilities are low, that implicitly means a healthy/normal scan.
NUM_CLASSES = len(ALL_DISEASES)  # 14


# ─── Dataset class ───────────────────────────────────────────────────────────

class ChestXray14Dataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.

    Args:
        image_dir   (str)               : path to folder containing all .png images
        df          (pd.DataFrame)      : rows for this split, with binary label columns
        transform   (callable, optional): torchvision transform pipeline
        return_info (bool)              : if True, __getitem__ also returns image filename
                                          (useful for debugging and CAM visualization)
    """

    def __init__(self, image_dir: str, df: pd.DataFrame,
                 transform=None, return_info: bool = False):
        self.image_dir   = image_dir
        self.df          = df.reset_index(drop=True)
        self.transform   = transform
        self.return_info = return_info

        # Pre-extract label matrix as float32 tensor for speed
        self.labels = torch.tensor(
            self.df[ALL_DISEASES].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        img_path  = os.path.join(self.image_dir, row["Image Index"])

        # Load as grayscale then convert to RGB
        # DenseNet-121 expects 3-channel input (ImageNet pretrained)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]  # shape: (14,)  float32

        if self.return_info:
            return image, label, row["Image Index"]
        return image, label


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_transforms(split: str = "train") -> T.Compose:
    """
    Returns the appropriate torchvision transform pipeline.

    Training augmentations follow the CheXNet paper:
      - Random horizontal flip
      - Random rotation (±10°)
      - Color jitter for slight brightness/contrast variation
      - Resize to 224x224, normalize with ImageNet stats

    Validation/test: deterministic resize + center crop + normalize only.

    Args:
        split (str): one of "train", "val", "test"

    Returns:
        torchvision.transforms.Compose
    """

    # ImageNet mean and std — used because DenseNet-121 is pretrained on ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]  #------------------------------------------------------ attention here

    if split == "train":
        return T.Compose([
            T.Resize(256),                          # slightly larger before crop
            T.RandomCrop(224),                      # random crop to 224×224
            T.RandomHorizontalFlip(p=0.5),          # L/R symmetric for chests
            T.RandomRotation(degrees=10),           # minor rotation ±10°
            T.ColorJitter(                          # subtle brightness/contrast
                brightness=0.2,
                contrast=0.2,
            ),
            T.ToTensor(),                           # → [0,1] float tensor (C,H,W)
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # val and test: deterministic, no randomness
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─── Label parsing helpers ────────────────────────────────────────────────────

def parse_labels(csv_path: str) -> pd.DataFrame:
    """
    Load Data_Entry_2017.csv and expand the pipe-separated 'Finding Labels'
    column into one binary column per disease.

    Args:
        csv_path (str): path to Data_Entry_2017.csv

    Returns:
        pd.DataFrame with columns: Image Index, Patient ID, Finding Labels,
                                   + one binary column per disease in ALL_DISEASES
    """
    df = pd.read_csv(csv_path)

    for disease in ALL_DISEASES:
        df[disease] = df["Finding Labels"].apply(
            lambda x: 1 if disease in x else 0
        )

    # Convenience flag: 1 if the report says "No Finding" (healthy scan)
    df["No Finding"] = df["Finding Labels"].apply(
        lambda x: 1 if "No Finding" in x else 0
    )

    print(f"[Dataset] Loaded {len(df):,} rows from {csv_path}")
    print(f"[Dataset] Class distribution:\n{df[ALL_DISEASES].sum().sort_values(ascending=False)}\n")

    return df


def apply_official_split(df: pd.DataFrame,
                         train_val_txt: str,
                         test_txt: str,
                         val_fraction: float = 0.1,
                         random_seed: int = 42):
    """
    Apply the official NIH train/test split (avoids patient data leakage),
    then carve a validation set out of the training portion.

    Args:
        df             : full DataFrame from parse_labels()
        train_val_txt  : path to train_val_list.txt
        test_txt       : path to test_list.txt
        val_fraction   : fraction of train set to use for validation (default 0.1)
        random_seed    : for reproducibility

    Returns:
        train_df, val_df, test_df  — three DataFrames
    """
    with open(train_val_txt) as f:
        train_val_files = set(f.read().splitlines())
    with open(test_txt) as f:
        test_files = set(f.read().splitlines())

    train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
    test_df      = df[df["Image Index"].isin(test_files)].copy()

    # Split train → train + val at patient level to prevent leakage
    patient_ids  = train_val_df["Patient ID"].unique()
    train_pids, val_pids = train_test_split(
        patient_ids,
        test_size=val_fraction,
        random_state=random_seed,
    )

    train_df = train_val_df[train_val_df["Patient ID"].isin(train_pids)]
    val_df   = train_val_df[train_val_df["Patient ID"].isin(val_pids)]

    print(f"[Split] Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    return train_df, val_df, test_df


# ─── Class weights (for handling imbalance) ──────────────────────────────────

def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss to handle
    the severe class imbalance in ChestX-ray14.

    Formula: weight_i = (N - n_i) / n_i
    where N = total samples, n_i = positive samples for class i.

    Returns:
        torch.Tensor of shape (14,) — pass as pos_weight to BCEWithLogitsLoss
    """
    N = len(train_df)
    pos_counts  = train_df[ALL_DISEASES].sum()
    pos_weights = (N - pos_counts) / (pos_counts + 1e-6)  # avoid div by zero
    pos_weights = pos_weights.clip(upper=50.0)             # cap extreme weights

    print("[Weights] Positive class weights:")
    for name, w in zip(ALL_DISEASES, pos_weights):
        print(f"  {name:<22}: {w:.1f}")

    return torch.tensor(pos_weights.values, dtype=torch.float32)


# ─── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(data_dir: str,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    val_fraction: float = 0.1,
                    return_weights: bool = True,
                    pin_memory: bool = True):
    """
    One-call factory: parses labels, splits data, builds Datasets and DataLoaders.

    Args:
        data_dir      : root directory containing images/, Data_Entry_2017.csv, etc.
        batch_size    : images per batch (32 works for 8GB GPU; lower if OOM)
        num_workers   : parallel data loading workers (4 is a safe default)
        val_fraction  : fraction of train set held out for validation
        return_weights: if True, also returns pos_weight tensor for loss function
        pin_memory    : speeds up GPU transfer (set False if using CPU only)

    Returns:
        (train_loader, val_loader, test_loader)              if return_weights=False
        (train_loader, val_loader, test_loader, pos_weights) if return_weights=True
    """

    csv_path      = os.path.join(data_dir, "Data_Entry_2017_v2020.csv")
    train_val_txt = os.path.join(data_dir, "train_val_list.txt")
    test_txt      = os.path.join(data_dir, "test_list.txt")
    image_dir     = os.path.join(data_dir, "images")

    # 1. Parse labels
    df = parse_labels(csv_path)

    # 2. Apply official split
    train_df, val_df, test_df = apply_official_split(
        df, train_val_txt, test_txt, val_fraction
    )

    # 3. Build Dataset objects
    train_dataset = ChestXray14Dataset(image_dir, train_df, transform=get_transforms("train"))
    val_dataset   = ChestXray14Dataset(image_dir, val_df,   transform=get_transforms("val"))
    test_dataset  = ChestXray14Dataset(image_dir, test_df,  transform=get_transforms("test"))

    # 4. Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # avoid single-sample batches at the end
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if return_weights:
        pos_weights = compute_class_weights(train_df)
        return train_loader, val_loader, test_loader, pos_weights

    return train_loader, val_loader, test_loader


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this directly to verify the dataset loads correctly:
        python chestxray_dataset.py
    """
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../dataset/"

    print("=" * 55)
    print("  ChestX-ray14 Dataset — sanity check")
    print("=" * 55)

    train_loader, val_loader, test_loader, pos_weights = get_dataloaders(
        data_dir,
        batch_size=8,
        num_workers=0,   # 0 for debugging; increase for training
    )

    # Grab one batch
    images, labels = next(iter(train_loader))

    print(f"\n[Batch check]")
    print(f"  Image tensor shape : {images.shape}")   # (8, 3, 224, 224)
    print(f"  Label tensor shape : {labels.shape}")   # (8, 14)
    print(f"  Image dtype        : {images.dtype}")   # float32
    print(f"  Label dtype        : {labels.dtype}")   # float32
    print(f"  Pixel range        : [{images.min():.2f}, {images.max():.2f}]")
    print(f"  pos_weight shape   : {pos_weights.shape}")  # (14,)

    # Show which diseases are active in the first image
    active = [ALL_DISEASES[i] for i, v in enumerate(labels[0]) if v == 1]
    print(f"\n[First image labels]: {active if active else ['No Finding']}")

    print("\nAll checks passed — dataset is ready for training.")
