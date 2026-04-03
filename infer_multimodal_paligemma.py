import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import PeftModel
import os

from dataset import LanguageDistanceDataset

MODEL_ID = "google/paligemma2-3b-pt-224"
CHECKPOINT = "/home/alekseyvalouev/goalnav/language-distance/language-distance-paligemma-multimodal-new-labels/checkpoint-3200"
NUM_SAMPLES = 16


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
    prompt = sample["prefix"]
    # Convert each numpy array in the image list to a PIL Image
    images_pil = [Image.fromarray(img.astype(np.uint8)) for img in sample["image"]]

    inputs = processor(
        text=prompt,
        images=images_pil,
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

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return prediction


def visualize(samples, predictions, save_path="inference_results_multimodal.png"):
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for ax, sample, pred in zip(axes, samples, predictions):
        images = sample["image"]  # list of numpy arrays
        prompt = sample["prefix"]
        gt = sample["suffix"]

        # Stitch images side-by-side if more than one, otherwise show the single image
        if len(images) > 1:
            combined = np.concatenate(images, axis=1)
        else:
            combined = images[0]

        ax.imshow(combined.astype(np.uint8))
        ax.axis("off")
        ax.set_title(f"GT: {gt}    Pred: {pred}", fontsize=12, fontweight="bold")

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
        annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-test-new"
    )

    indices = np.linspace(0, len(val_dataset) - 1, NUM_SAMPLES, dtype=int)
    samples = [val_dataset[int(i)] for i in indices]

    print(f"Running inference on {NUM_SAMPLES} samples...")
    predictions = []
    for i, sample in enumerate(samples):
        pred = run_inference(model, processor, sample)
        gt = sample["suffix"]
        num_images = len(sample["image"])
        print(f"  [{i+1}/{NUM_SAMPLES}]  GT={gt}  Pred={pred}  (#images={num_images})")
        predictions.append(pred)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference_results_multimodal.png"
    )
    visualize(samples, predictions, save_path=output_path)


if __name__ == "__main__":
    main()
