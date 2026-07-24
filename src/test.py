# from cam import load_single_image, visualize_prediction
# from model import get_model
# import torch

# device = torch.device("cpu")
# model  = get_model(14, device)   # untrained weights — CAM will look random, that's fine
# image  = load_single_image("../dataset/images/00000001_000.png")
# dummy_labels = [0] * 14

# fig = visualize_prediction(model, image, dummy_labels, device,
#                             target_class_idx=6,   # Pneumonia
#                             save_path="results/test_cam.png")

from evaluate import evaluate_full
from model import get_model
from chestxray_dataset import get_dataloaders
from pathlib import Path
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best checkpoint
model = get_model(num_classes=14, device=device)
ckpt  = torch.load("checkpoint/checkpoint_epoch8.tar",
                    map_location=device)
model.load_state_dict(ckpt["model_state"])
print(f"Loaded epoch {ckpt['epoch']} | val_loss {ckpt['val_loss']:.4f}")

# Test loader only
_, _, test_loader, _ = get_dataloaders(
    data_dir=Path("../dataset/"),
    batch_size=32,
    num_workers=4,
)

# Run full evaluation — this prints the AUROC table
results = evaluate_full(
    model=model,
    test_loader=test_loader,
    device=device,
    save_dir="results/"
)