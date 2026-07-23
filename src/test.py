from cam import load_single_image, visualize_prediction
from model import get_model
import torch

device = torch.device("cpu")
model  = get_model(14, device)   # untrained weights — CAM will look random, that's fine
image  = load_single_image("../dataset/images/00000001_000.png")
dummy_labels = [0] * 14

fig = visualize_prediction(model, image, dummy_labels, device,
                            target_class_idx=6,   # Pneumonia
                            save_path="results/test_cam.png")