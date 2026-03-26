import re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

from dataset import LanguageDistanceDataset

MODEL_ID = "google/gemma-2b"
CHECKPOINT = "/home/alekseyvalouev/goalnav/language-distance/language-distance-gemma/checkpoint-4236"
NUM_SAMPLES = 8


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT)
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, sample):
    prompt = sample["prefix"]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # The model may keep generating beyond the answer; extract the first integer.
    #match = re.search(r"\d+", raw)
    #prediction = match.group(0) if match else raw
    if "16" in raw:
        return "16"
    return raw[0]


def visualize(samples, predictions, save_path="inference_results_gemma.png"):
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n))
    if n == 1:
        axes = [axes]

    for ax, sample, pred in zip(axes, samples, predictions):
        prompt = sample["prefix"]
        gt = sample["suffix"]

        ax.axis("off")
        ax.set_title(f"GT: {gt}    Pred: {pred}", fontsize=12, fontweight="bold")
        ax.text(
            0.5, 0.4,
            prompt,
            transform=ax.transAxes,
            wrap=True,
            ha="center",
            fontsize=7,
            va="top",
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")


def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading val dataset...")
    val_dataset = LanguageDistanceDataset(
        annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-test",
        text_only=True,
    )

    # Pick evenly-spaced samples from the val set
    indices = np.linspace(0, len(val_dataset) - 1, NUM_SAMPLES, dtype=int)
    samples = [val_dataset[int(i)] for i in indices]

    print(f"Running inference on {NUM_SAMPLES} samples...")
    predictions = []
    for i, sample in enumerate(samples):
        pred = run_inference(model, tokenizer, sample)
        gt = sample["suffix"]
        print(f"  [{i+1}/{NUM_SAMPLES}]  GT={gt}  Pred={pred}")
        predictions.append(pred)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference_results_gemma.png"
    )
    visualize(samples, predictions, save_path=output_path)


if __name__ == "__main__":
    main()
