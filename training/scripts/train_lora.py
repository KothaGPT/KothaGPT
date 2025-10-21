#!/usr/bin/env python3
"""
KothaGPT LoRA Fine-Tuning Script
- Uses processed dataset and trained tokenizer
- Supports multi-GPU, checkpointing, and logging
"""

import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import sentencepiece as spm

# -----------------------------
# 1️⃣ Load Config
# -----------------------------
config_path = "training/configs/train_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

train_data_path = config["train_data"]
tokenizer_path = config["tokenizer_path"]
base_model_name = config["base_model"]
output_dir = config["output_dir"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]
lora_config_params = config["lora"]

# -----------------------------
# 2️⃣ Load Tokenizer and Dataset
# -----------------------------
print("Loading tokenizer...")
try:
    # Try loading as SentencePiece tokenizer first
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(tokenizer_path, "kothagpt_tokenizer.model"))
    tokenizer = sp
except:
    # Fall back to AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

print("Loading dataset...")
dataset = load_dataset('csv', data_files={'train': train_data_path}, split='train')

# Tokenize function - handle both SentencePiece and HuggingFace tokenizers
def tokenize_function(examples):
    if hasattr(tokenizer, 'encode'):
        # SentencePiece tokenizer
        return {
            'input_ids': [tokenizer.encode(text, add_bos=True, add_eos=True) for text in examples['text']],
            'attention_mask': [[1] * len(ids) for ids in [tokenizer.encode(text, add_bos=True, add_eos=True) for text in examples['text']]]
        }
    else:
        # HuggingFace tokenizer
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# -----------------------------
# 3️⃣ Load Base Model
# -----------------------------
print(f"Loading base model: {base_model_name} ...")
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)

# -----------------------------
# 4️⃣ Apply LoRA
# -----------------------------
print("Applying LoRA fine-tuning...")
lora_config = LoraConfig(
    r=lora_config_params["r"],
    lora_alpha=lora_config_params["alpha"],
    target_modules=lora_config_params["target_modules"],
    lora_dropout=lora_config_params["dropout"],
    bias=lora_config_params["bias"],
    task_type=lora_config_params["task_type"]
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 5️⃣ Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    load_best_model_at_end=True
)

# -----------------------------
# 6️⃣ Trainer Setup
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# -----------------------------
# 7️⃣ Start Training
# -----------------------------
print("Starting LoRA fine-tuning...")
trainer.train()

print(f"\n✅ Training complete! Model saved at {output_dir}")
