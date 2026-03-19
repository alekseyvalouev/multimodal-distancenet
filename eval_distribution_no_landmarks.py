"""
Runs inference over the full val set and plots the distribution of
ground-truth labels vs. predicted labels side-by-side, plus a
confusion matrix.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel
import os

from dataset import LanguageDistanceDataset

MODEL_ID  = "google/paligemma2-3b-pt-224"
CHECKPOINT = "/home/alekseyvalouev/goalnav/language-distance/language-distance-paligemma-no-landmarks/checkpoint-1900"
ANNOTATIONS = "/home/alekseyvalouev/goalnav/language-annotations-test"
# Set to None to run on the entire val set
NUM_SAMPLES = 200


def load_model():
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
    model = PeftModel.from_pretrained(base_model, CHECKPOINT)
    model.eval()
    return model, processor


def run_inference(model, processor, sample):
    image_pil = Image.fromarray(sample["image"].astype(np.uint8))
    inputs = processor(
        text=sample["prefix"],
        images=image_pil,
        return_tensors="pt",
        padding=True,
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
    return raw


def parse_label(s):
    """Try to extract the first integer from a string; return None on failure."""
    import re
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def plot_distributions(gt_labels, pred_labels, save_path):
    all_vals = sorted(set(gt_labels) | set(pred_labels))

    gt_counts  = Counter(gt_labels)
    pred_counts = Counter(pred_labels)

    gt_freq   = [gt_counts.get(v, 0)   for v in all_vals]
    pred_freq = [pred_counts.get(v, 0) for v in all_vals]

    x = np.arange(len(all_vals))
    width = 0.35

    # ---------- confusion matrix ----------
    label_to_idx = {v: i for i, v in enumerate(all_vals)}
    n = len(all_vals)
    conf = np.zeros((n, n), dtype=int)
    for g, p in zip(gt_labels, pred_labels):
        if g in label_to_idx and p in label_to_idx:
            conf[label_to_idx[g], label_to_idx[p]] += 1

    # ---------- compute MAE on valid pairs ----------
    pairs = [(g, p) for g, p in zip(gt_labels, pred_labels) if p is not None]
    mae = np.mean([abs(g - p) for g, p in pairs]) if pairs else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Val set  |  n={len(gt_labels)}  |  MAE={mae:.2f}",
        fontsize=14, fontweight="bold"
    )

    # --- bar chart ---
    ax = axes[0]
    bars_gt   = ax.bar(x - width / 2, gt_freq,   width, label="Ground truth", color="#4C72B0", alpha=0.85)
    bars_pred = ax.bar(x + width / 2, pred_freq, width, label="Predicted",    color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_vals])
    ax.set_xlabel("Label (temporal distance)")
    ax.set_ylabel("Count")
    ax.set_title("Label distribution: GT vs Predicted")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # --- confusion matrix (row-normalised by GT count) ---
    row_sums = conf.sum(axis=1, keepdims=True)
    conf_norm = np.where(row_sums > 0, conf / row_sums, 0.0)

    ax2 = axes[1]
    im = ax2.imshow(conf_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([str(v) for v in all_vals], rotation=45, ha="right")
    ax2.set_yticklabels([str(v) for v in all_vals])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Ground truth")
    ax2.set_title("Confusion matrix (row-normalised)")
    plt.colorbar(im, ax=ax2, format="%.2f")

    # Annotate cells with percentage
    for i in range(n):
        for j in range(n):
            if conf[i, j] > 0:
                ax2.text(j, i, f"{conf_norm[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if conf_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {save_path}")


def main():
    print("Loading model...")
    model, processor = load_model()

    print("Loading val dataset...")
    val_dataset = LanguageDistanceDataset(annotations_folder=ANNOTATIONS, lang_labels=False)

    n = len(val_dataset) if NUM_SAMPLES is None else min(NUM_SAMPLES, len(val_dataset))
    indices = np.linspace(0, len(val_dataset) - 1, n, dtype=int)

    gt_labels, pred_labels = [], []
    for step, idx in enumerate(indices):
        sample = val_dataset[int(idx)]
        raw_pred = run_inference(model, processor, sample)
        gt  = parse_label(sample["suffix"])
        pred = parse_label(raw_pred)
        gt_labels.append(gt)
        pred_labels.append(pred)
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{step+1}/{n}]  GT={gt}  Raw pred='{raw_pred}'  Parsed={pred}")

    # Drop samples where gt or pred couldn't be parsed
    valid = [(g, p) for g, p in zip(gt_labels, pred_labels) if g is not None and p is not None]
    invalid = n - len(valid)
    if invalid:
        print(f"  Warning: {invalid} samples had unparseable predictions and were dropped.")
    gt_valid, pred_valid = zip(*valid) if valid else ([], [])

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_distribution.png")
    plot_distributions(list(gt_valid), list(pred_valid), save_path=out)


if __name__ == "__main__":
    main()
