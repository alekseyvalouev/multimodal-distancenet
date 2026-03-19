import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel
import os

from dataset import LanguageDistanceDataset

MODEL_ID = "google/paligemma2-3b-pt-224"
CHECKPOINT = "/home/alekseyvalouev/goalnav/language-distance/language-distance-paligemma/checkpoint-1400"
NUM_SAMPLES = 8


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
    image_np = sample["image"]
    prompt = sample["prefix"]

    # Convert stitched numpy array to PIL
    image_pil = Image.fromarray(image_np.astype(np.uint8))

    inputs = processor(
        text=prompt,
        images=image_pil,
        return_tensors="pt",
        padding=True,
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return prediction


def visualize(samples, predictions, save_path="inference_results.png"):
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, sample, pred in zip(axes, samples, predictions):
        image = sample["image"]
        prompt = sample["prefix"]
        gt = sample["suffix"]

        ax.imshow(image.astype(np.uint8))
        ax.axis("off")
        ax.set_title(f"GT: {gt}    Pred: {pred}", fontsize=12, fontweight="bold")

        # Wrap the prompt as a caption
        ax.text(
            0.5, -0.02,
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
    model, processor = load_model()

    print("Loading val dataset...")
    val_dataset = LanguageDistanceDataset(
        annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-test"
    )

    # Pick evenly-spaced samples from the val set
    indices = np.linspace(0, len(val_dataset) - 1, NUM_SAMPLES, dtype=int)
    samples = [val_dataset[int(i)] for i in indices]

    print(f"Running inference on {NUM_SAMPLES} samples...")
    predictions = []
    for i, sample in enumerate(samples):
        pred = run_inference(model, processor, sample)
        gt = sample["suffix"]
        print(f"  [{i+1}/{NUM_SAMPLES}]  GT={gt}  Pred={pred}")
        predictions.append(pred)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference_results.png"
    )
    visualize(samples, predictions, save_path=output_path)


if __name__ == "__main__":
    main()
