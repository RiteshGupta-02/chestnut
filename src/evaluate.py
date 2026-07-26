from chestxray_dataset import get_dataloaders
from model import cheXNet, get_model
from logger import setup_logger

from pathlib import Path
import torch
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import os
from datetime import datetime
import json

logging = setup_logger()

DEBUG = True

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

PAPER_AUROC = {
    'Atelectasis':       0.8094,
    'Cardiomegaly':      0.9248,
    'Effusion':          0.8638,
    'Infiltration':      0.7345,
    'Mass':              0.8676,
    'Nodule':            0.7802,
    'Pneumonia':         0.7680,
    'Pneumothorax':      0.8887,
    'Consolidation':     0.7901,
    'Edema':             0.8878,
    'Emphysema':         0.9371,
    'Fibrosis':          0.8047,
    'Pleural_Thickening':0.8062,
    'Hernia':            0.9164,
}

def get_predictions(model, loader, device):

    model.eval()

    pred_list = []
    label_list = []

    logging.info("Collecting predictions on test set...")

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(loader):
            if batch_idx > 5 and DEBUG:
                break
            image = image.to(device)
            out = model(image)
            
            pred_list.append(out.cpu().numpy())
            label_list.append(label.numpy())

            if batch_idx % 50 == 0:
                logging.info(f"Processed {batch_idx * loader.batch_size} / {len(loader.dataset)} images")
    
    all_prediction = np.concatenate(pred_list,axis=0)
    all_labels = np.concatenate(label_list,axis=0)

    
    return all_prediction, all_labels

def compute_auroc(all_preds, all_labels):
    
    results = {}

    for idx, diesease in enumerate(ALL_DISEASES):
        y_true = all_labels[:,idx]
        y_pred = all_preds[:,idx]


        n_positive = int(y_true.sum())
        n_total    = len(y_true)

        if n_positive == 0:
            logging.warning(f"  {diesease}: no positive cases in test set — skipping AUROC")
            results[diesease] = {"auroc": None, "n_positive": 0}
            continue

        # Guard: cannot compute AUROC if all cases are positive
        if n_positive == n_total:
            logging.warning(f"  {diesease}: all cases are positive — skipping AUROC")
            results[diesease] = {"auroc": None, "n_positive": n_positive}
            continue

        print(y_true,"\n",y_pred)
        score = roc_auc_score(y_true, y_pred)

        entry = {
            "auroc" : round(score,4),
            "n_positive" : n_positive
        }

        results[diesease] = entry
        logging.info(f"  {diesease:<22}: AUROC = {score:.4f}"
                    + f"  [n_pos={n_positive}]")

    return results


def print_auroc_table(auroc_results):
    """
    Print a formatted side-by-side table comparing your results vs the paper.

    Example output:
    ┌──────────────────────┬──────────┬──────────┬──────────┬───────────────────┬────────┐
    │ Disease              │ Yours    │ Paper    │ delta    │ 95% CI            │ Beat?  │
    ├──────────────────────┼──────────┼──────────┼──────────┼───────────────────┼────────┤
    │ Pneumonia            │ 0.7823   │ 0.7680   │ +0.0143  │ [0.761 - 0.803]   │  true  │
    │ Effusion             │ 0.8501   │ 0.8638   │ -0.0137  │ [0.838 - 0.862]   │        │

    Args:
        auroc_results : dict from compute_auroc()
    """
    print("\n" + "═" * 80)
    print("  AUROC EVALUATION — CheXNet Replication Results")
    print("═" * 80)
    print(f"  {'Disease':<22} {'Yours':>8} {'Paper':>8} {'delta':>8}  {'95% CI':<20} {'Beat?':>6}")
    print("─" * 80)

    beat_count  = 0
    valid_count = 0
    your_scores = []

    for disease in ALL_DISEASES:
        result      = auroc_results.get(disease, {})
        your_auroc  = result.get("auroc")
        paper_auroc = PAPER_AUROC.get(disease, None)

        if your_auroc is None:
            print(f"  {disease:<22} {'N/A':>8} {paper_auroc:>8.4f} {'-':>8}  {'no positive cases':<20}")
            continue
        
        delta = float(your_auroc) - paper_auroc
        delta_str = f"{delta:+.4f}"

        ci_str = ""
        if "ci_lower" in result:
            ci_str = f"[{result['ci_lower']:.4f} - {result['ci_upper']:.4f}]"

        beat = "  true" if delta >= 0 else ""
        if delta >= 0:
            beat_count += 1

        valid_count += 1
        your_scores.append(your_auroc)

        print(f"  {disease:<22} {your_auroc:>8.4f} {paper_auroc:>8.4f} {delta_str:>8}  {ci_str:<20} {beat:>6}")

    print("─" * 80)

    if your_scores:
        mean_yours = np.mean(your_scores)
        mean_paper = np.mean(list(PAPER_AUROC.values()))
        print(f"  {'Mean AUROC':<22} {mean_yours:>8.4f} {mean_paper:>8.4f} {mean_yours - mean_paper:>+8.4f}")
        print(f"\n  Beat or matched paper on {beat_count}/{valid_count} diseases")

    print("═" * 80 + "\n")

def plot_roc_curves(all_labels, all_preds, auroc_results,save_path):
    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 5, figsize=(22, 14))
    axes = axes.flatten()

    # Colour palette — one colour per disease
    colours = plt.cm.tab20(np.linspace(0, 1, len(ALL_DISEASES)))

    for i, (disease, colour) in enumerate(zip(ALL_DISEASES, colours)):
        ax = axes[i]

        y_true  = all_labels[:, i]
        y_score = all_preds[:, i]

        result     = auroc_results.get(disease, {})
        your_auroc = result.get("auroc")

        # Random baseline — diagonal line
        ax.plot([0, 1], [0, 1],
                linestyle="--", color="gray", linewidth=1, alpha=0.6,
                label="Random (0.5000)")

        if your_auroc is None:
            # No positive cases — draw placeholder
            ax.text(0.5, 0.5, "No positive\ncases in test set",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
        else:
            # Compute ROC curve points
            fpr, tpr, _ = roc_curve(y_true, y_score)

            # Paper's AUROC — dashed reference line shows where paper sits
            paper_val = PAPER_AUROC.get(disease, 0)

            # Your curve
            ax.plot(fpr, tpr,
                    color=colour, linewidth=2,
                    label=f"Yours  ({your_auroc:.4f})")

            # Shade area under curve lightly
            ax.fill_between(fpr, tpr, alpha=0.08, color=colour)

            # Confidence interval annotation
            if "ci_lower" in result:
                ci_text = f"95% CI [{result['ci_lower']:.4f}-{result['ci_upper']:.4f}]"
                ax.text(0.97, 0.08, ci_text,
                        ha="right", va="bottom", transform=ax.transAxes,
                        fontsize=7, color="dimgray",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            # Paper reference score shown as text
            ax.text(0.97, 0.20, f"Paper  ({paper_val:.4f})",
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=8, color="steelblue")

            # Beat/missed indicator
            beat_str = "Beat paper" if your_auroc >= paper_val else "Below paper"
            beat_col = "#2E7D32" if your_auroc >= paper_val else "#C62828"
            ax.text(0.97, 0.32, beat_str,
                    ha="right", va="bottom", transform=ax.transAxes,
                    fontsize=8, color=beat_col, fontweight="bold")

        ax.set_title(disease, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right")

    # Hide the 15th subplot (3×5 grid has 15 slots, we only have 14 diseases)
    axes[-1].set_visible(False)

    # Overall figure title
    mean_auroc = np.mean([r["auroc"] for r in auroc_results.values()
                          if r.get("auroc") is not None])
    fig.suptitle(
        f"ROC Curves - CheXNet Replication \nMean AUROC: {mean_auroc:.4f}  "
        f"(Paper: {np.mean(list(PAPER_AUROC.values())):.4f})",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"ROC curves saved {save_path}")

def evaluate_full(model, test_loader, device, result_path):
    result_path = Path(result_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_preds, all_labels = get_predictions(model, test_loader, device)
    result = compute_auroc(all_preds,all_labels)
    print_auroc_table(result)

    save_path = (result_path / "plots" / f'{timestamp}.png')
    os.makedirs(save_path, exist_ok=True)

    plot_roc_curves(all_labels, all_preds, result, save_path)

    json_path = result_path / f"auroc_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp":    timestamp,
            "mean_auroc":   float(np.mean([
                float(r["auroc"]) for r in result.values()
                if r.get("auroc") is not None
            ])),
            "paper_mean":   float(np.mean(list(PAPER_AUROC.values()))),
            "per_disease":  result,
        }, f, indent=2)
    logging.info(f"Results saved : {json_path}")
    return all_labels, all_preds, result, save_path

def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    batch_size = 4
    num_workers = 0

    _, _, test_loader, _ = get_dataloaders(Path("../dataset/"),batch_size=batch_size,num_workers=num_workers)
    checkpoint = max(list(glob.glob("checkpoint/checkpoint_epoch10.tar")))
    print(checkpoint)

    model = get_model(14,device)
    checkpoint = torch.load(checkpoint,map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    all_preds, all_labels = get_predictions(model, test_loader, device)

    result = compute_auroc(all_preds,all_labels)

    print_auroc_table(result)

    save_path = (Path(__file__).parent / "plots" / f'{timestamp}')
    os.makedirs(save_path, exist_ok=True)

    plot_roc_curves(all_labels, all_preds, result, save_path)

    json_path = Path(__file__).parent / f"auroc_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp":    timestamp,
            "mean_auroc":   float(np.mean([
                r["auroc"] for r in result.values()
                if r.get("auroc") is not None
            ])),
            "paper_mean":   float(np.mean(list(PAPER_AUROC.values()))),
            "per_disease":  result,
        }, f, indent=2)
    logging.info(f"Results saved : {json_path}")

if "__main__" == __name__:
    main()