"""
Evaluate a fine-tuned PaliGemma model on an eval set and compute error statistics
using an arbitrarily specified error function, then produce a chart of the results.

Simple error functions receive (gt: int, pred: int) and return a float.
The special "ce_loss" metric uses a teacher-forced forward pass instead and
does not depend on the decoded integer — it measures the model's log-probability
of the ground-truth suffix tokens directly.

Swap out ERROR_FN or pass --error-fn on the CLI to change the metric.

Usage:
    python eval_errors.py                          # defaults (mae)
    python eval_errors.py --checkpoint /path/to/ckpt --num-samples 500
    python eval_errors.py --error-fn squared       # mae | squared | sign_correct | relative | ce_loss
    python eval_errors.py --error-fn ce_loss
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

from dataset import HistoryVisionOnlyDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID   = "google/paligemma2-3b-pt-224"
CHECKPOINT = (
    "/home/alekseyvalouev/goalnav/language-distance/"
    "language-distance-paligemma-history/checkpoint-2300"
)
NUM_SAMPLES = None

#SACSON_TEST_SCENES = [
#    "Dec-06-2022-bww8_00000007_0",
#    "Feb-09-2023-bww8-intloss_00000042_9",
#    "Jan-12-2023-bww8_00000007_22",
#    "Feb-03-2023-bww8-intloss_00000013_1",
#    "Feb-14-2023-bww8-intloss_00000008_25",
#]
SACSON_TEST_SCENES = [
    "Feb-15-2023-cory1_00000000_0", "Feb-15-2023-cory1_00000006_4", "Feb-16-2023-cory1-intloss_00000021_1",
    "Feb-15-2023-cory1_00000004_6", "Feb-15-2023-cory1_00000006_5", "Feb-16-2023-cory1-intloss_00000023_0"
]

MAPPING = {
    -1: 32,
    0: 2,
    1: 4,
    2: 8,
    3: 16
}

# ---------------------------------------------------------------------------
# Built-in error functions  (gt and pred are both ints parsed from model output)
# ---------------------------------------------------------------------------

def mae(gt: int, pred: int) -> float:
    """Absolute error."""
    return abs(gt - pred)

def squared(gt: int, pred: int) -> float:
    """Squared error."""
    return float((gt - pred) ** 2)

def sign_correct(gt: int, pred: int) -> float:
    """1.0 if prediction is wrong, 0.0 if correct (i.e. error = 1 - accuracy)."""
    return 0.0 if gt == pred else 1.0

def relative(gt: int, pred: int) -> float:
    """Relative absolute error (normalised by gt; clamped to avoid divide-by-zero)."""
    denom = max(abs(gt), 1)
    return abs(gt - pred) / denom

ERROR_FNS = {
    "mae":          mae,
    "squared":      squared,
    "sign_correct": sign_correct,
    "relative":     relative,
    # "ce_loss" is handled specially (teacher-forced forward pass); listed here
    # only so it appears in argparse choices.
    "ce_loss":      None,
}

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()
    return model, processor


def run_inference(model, processor, sample: dict) -> str:
    images_pil = [Image.fromarray(img.astype(np.uint8)) for img in sample["image"]]
    inputs = processor(
        text=sample["prefix"],
        images=images_pil,
        return_tensors="pt",
        padding=True,
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(
        output_ids[:, input_len:], skip_special_tokens=True
    )[0].strip()
    return MAPPING[parse_int(raw)]


def compute_ce_loss(model, processor, sample: dict) -> float:
    """
    Teacher-forced cross-entropy loss over the ground-truth suffix tokens.

    The processor encodes the full (prefix + suffix) sequence and marks prefix
    positions as -100 in `labels`, so the model's built-in loss averages CE only
    over the suffix tokens — exactly what we want.
    """
    images_pil = [Image.fromarray(img.astype(np.uint8)) for img in sample["image"]]
    inputs = processor(
        text=sample["prefix"],
        images=images_pil,
        suffix=sample["suffix"],
        return_tensors="pt",
        padding="longest",
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.loss.item()


def parse_int(s: str):
    """Extract the first integer from a string; return None on failure."""
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None

# ---------------------------------------------------------------------------
# Statistics and plotting
# ---------------------------------------------------------------------------

def compute_stats(errors: list[float]) -> dict:
    arr = np.array(errors, dtype=float)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "median": float(np.median(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "n":      len(arr),
    }


def plot_errors(
    errors: list[float],
    gt_labels: list[int],
    pred_labels: list[int],
    stats: dict,
    error_fn_name: str,
    save_path: str,
):
    unique_gts = sorted(set(gt_labels))

    # Per-GT-label breakdown
    per_label_errors: dict[int, list[float]] = {g: [] for g in unique_gts}
    for err, gt in zip(errors, gt_labels):
        per_label_errors[gt].append(err)

    per_label_means = [np.mean(per_label_errors[g]) for g in unique_gts]
    per_label_stds  = [np.std(per_label_errors[g])  for g in unique_gts]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Error statistics  |  metric={error_fn_name}  |  n={stats['n']}  |  "
        f"mean={stats['mean']:.3f}  |  std={stats['std']:.3f}  |  median={stats['median']:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    # --- 1. Histogram of per-sample errors ---
    ax = axes[0]
    ax.hist(errors, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(stats["mean"],   color="#DD8452", linewidth=2, linestyle="-",  label=f"mean={stats['mean']:.3f}")
    ax.axvline(stats["median"], color="#55A868", linewidth=2, linestyle="--", label=f"median={stats['median']:.3f}")
    ax.axvspan(
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        alpha=0.15, color="#DD8452", label=f"±1 std ({stats['std']:.3f})",
    )
    ax.set_xlabel(f"Error  ({error_fn_name})")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # --- 2. Per-GT-label mean ± std bar chart ---
    ax2 = axes[1]
    x = np.arange(len(unique_gts))
    bars = ax2.bar(x, per_label_means, color="#4C72B0", alpha=0.85, width=0.6)
    ax2.errorbar(
        x, per_label_means, yerr=per_label_stds,
        fmt="none", color="black", capsize=5, linewidth=1.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(g) for g in unique_gts])
    ax2.set_xlabel("Ground-truth label")
    ax2.set_ylabel(f"Mean error  ({error_fn_name})")
    ax2.set_title("Mean error ± std per GT label")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate count
    for bar, gt in zip(bars, unique_gts):
        n = len(per_label_errors[gt])
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(per_label_stds) * 0.05,
            f"n={n}",
            ha="center", va="bottom", fontsize=8,
        )

    # --- 3. Scatter: GT vs Pred ---
    ax3 = axes[2]
    ax3.scatter(gt_labels, pred_labels, alpha=0.4, s=12, color="#4C72B0", label="samples")
    lims = [
        min(min(gt_labels), min(pred_labels)) - 0.5,
        max(max(gt_labels), max(pred_labels)) + 0.5,
    ]
    ax3.plot(lims, lims, "k--", linewidth=1, label="perfect prediction")
    ax3.set_xlabel("Ground-truth label")
    ax3.set_ylabel("Predicted label")
    ax3.set_title("GT vs Predicted")
    ax3.legend(fontsize=9)
    ax3.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved chart to {save_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eval error statistics for PaliGemma history model.")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Path to LoRA checkpoint directory.")
    parser.add_argument(
        "--error-fn",
        default="mae",
        choices=list(ERROR_FNS.keys()),
        help="Error function to use: " + ", ".join(ERROR_FNS.keys()),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of eval samples (None = entire val set).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to eval_errors_<error_fn>.png next to this script.",
    )
    args = parser.parse_args()

    use_ce_loss = args.error_fn == "ce_loss"
    error_fn    = None if use_ce_loss else ERROR_FNS[args.error_fn]
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"eval_errors_{args.error_fn}.png",
    )

    print("Loading model…")
    model, processor = load_model(args.checkpoint)

    print("Loading val dataset…")
    val_dataset = HistoryVisionOnlyDataset(scenes=SACSON_TEST_SCENES)

    n = min(args.num_samples, len(val_dataset)) if args.num_samples else len(val_dataset)
    indices = np.linspace(0, len(val_dataset) - 1, n, dtype=int)

    gt_labels, pred_labels, errors = [], [], []
    invalid = 0

    for step, idx in enumerate(indices):
        sample = val_dataset[int(idx)]
        raw_pred = run_inference(model, processor, sample)
        gt   = MAPPING[parse_int(sample["suffix"])]
        pred = raw_pred

        if gt is None:
            invalid += 1
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  [{step+1}/{n}]  GT={sample['suffix']!r} unparseable, skipping")
            continue

        if use_ce_loss:
            err = compute_ce_loss(model, processor, sample)
            # Use greedy pred (may be None) only for the scatter plot; fall back to gt
            # so the scatter plot still has a point even if the decode failed.
            plot_pred = pred if pred is not None else gt
        else:
            if pred is None:
                invalid += 1
                if (step + 1) % 10 == 0 or step == 0:
                    print(f"  [{step+1}/{n}]  Pred={raw_pred!r} unparseable, skipping")
                continue
            err = error_fn(gt, pred)
            plot_pred = pred

        gt_labels.append(gt)
        pred_labels.append(plot_pred)
        errors.append(err)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{step+1}/{n}]  GT={gt}  Pred={raw_pred!r}  Error={err:.4f}")

    if invalid:
        print(f"\nWarning: {invalid}/{n} samples had unparseable predictions and were dropped.")

    if not errors:
        print("No valid samples — cannot compute statistics.")
        return

    stats = compute_stats(errors)
    print("\n--- Error statistics ---")
    for k, v in stats.items():
        print(f"  {k:>8s}: {v}")

    plot_errors(errors, gt_labels, pred_labels, stats, args.error_fn, out_path)


if __name__ == "__main__":
    main()
