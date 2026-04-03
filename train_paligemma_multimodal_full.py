import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration, 
    PaliGemmaProcessor, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import os

from dataset import LanguageDistanceDataset

MODEL_ID = "google/paligemma2-3b-pt-224"

def setup_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config, 
        torch_dtype=torch.float16,
    )
    #processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    #model = PaliGemmaForConditionalGeneration.from_pretrained(
    #    MODEL_ID,
    #    torch_dtype=torch.float16,
    #    #device_map="auto"
    #)

    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False

    #model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def collate_fn(examples):
        texts = [example["prefix"] for example in examples]
        labels = [example["suffix"] for example in examples]
        images = [example["image"] for example in examples]
        
        # Processor handles the prompt + image + target labels
        tokens = processor(
            text=texts, 
            images=images, 
            suffix=labels, 
            return_tensors="pt", 
            padding="longest"
        )
        
        tokens["pixel_values"] = tokens["pixel_values"].to(torch.float16)
        # Ensure tokens are in the correct format for the model
        #tokens = {k: v.to(model.device) for k, v in tokens.items()}        
        return tokens

    return model, collate_fn

def train(model, collate_fn):
    os.environ["WANDB_PROJECT"] = "language-distance-paligemma-multimodal-directional" 
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"             

    train_dataset = LanguageDistanceDataset(annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-train", full_sample=True)
    val_dataset = LanguageDistanceDataset(annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-test", full_sample=True)

    args = TrainingArguments(
        output_dir="language-distance-paligemma-multimodal-directional",
        remove_unused_columns=False,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        bf16=False,
        fp16=True,
        push_to_hub=False,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        args=args
    )

    trainer.train()

def main():
    model, collate_fn = setup_model()
    train(model, collate_fn)

if __name__ == "__main__":
    main()

# run with CUDA_VISIBLE_DEVICES=0 python train_paligemma.py