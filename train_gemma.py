import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
# Assuming LanguageDistanceDataset is in your local dataset.py
from dataset import BinaryReachabilityDataset

MODEL_ID = "google/gemma-2b"

def setup_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, # Recommended for better efficiency
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Causal LM padding usually works best on the right for training
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Critical for 4-bit loading
        torch_dtype=torch.float16,
    )

    # 1. MUST prepare model for quantized training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def get_collate_fn(tokenizer):
    def collate_fn(examples):
        full_texts = [example["prefix"] + example["suffix"] for example in examples]
        prompts = [example["prefix"] for example in examples]

        # 1. Full encodings (the actual model input)
        full_encodings = tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # 2. Tokenize prompts INDIVIDUALLY to get their exact lengths
        # We don't return_tensors="pt" here to avoid the sequence length error
        prompt_encodings = tokenizer(
            prompts, 
            padding=False, 
            truncation=True, 
            add_special_tokens=False
        )

        input_ids = full_encodings["input_ids"]
        attention_mask = full_encodings["attention_mask"]
        labels = input_ids.clone()

        # 3. Mask the prompt part of the labels
        for i, prompt_ids in enumerate(prompt_encodings["input_ids"]):
            prompt_len = len(prompt_ids)
            # Ensure we don't exceed the actual input_ids length (in case of truncation)
            actual_prompt_len = min(prompt_len, input_ids.size(1))
            labels[i, :actual_prompt_len] = -100

        # 4. Mask the padding tokens
        labels[input_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn

def train(model, tokenizer):
    os.environ["WANDB_PROJECT"] = "binary-reachability-gemma"
    
    # Ensure dataset paths are correct for your environment
    train_dataset = BinaryReachabilityDataset(
        annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-train",
        text_only=True,
    )
    val_dataset = BinaryReachabilityDataset(
        annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-test",
        text_only=True,
    )

    args = TrainingArguments(
        output_dir="binary-reachability-gemma",
        remove_unused_columns=False,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-4, # Slightly higher LR is usually better for LoRA
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        fp16=True,
        report_to="wandb",
        # Fix for possible Gemma issues with gradient checkpointing
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_collate_fn(tokenizer),
        args=args,
    )

    # Required for training with 4-bit
    model.config.use_cache = False 

    trainer.train()

def main():
    model, tokenizer = setup_model()
    train(model, tokenizer)

if __name__ == "__main__":
    main()

# run with CUDA_VISIBLE_DEVICES=0 python train_gemma.py
