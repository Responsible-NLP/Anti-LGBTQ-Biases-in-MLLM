import os
import pickle
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelWithLMHead, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import math
import pandas as pd
import spacy
import sys
import random

# ARGUMENTS
# sys.argv[1]: pretrained model path
# sys.argv[2]: finetune file 1 path
# sys.argv[3]: finetune file 2 path
# sys.argv[4]: save path (optional)
# sys.argv[5]: model_max_length (optional, default 512)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

pretrained_model = sys.argv[1]
finetune_file_1 = "queer_articles_finetune.txt"
finetune_file_2 = "queer_posts_finetune.txt"
save_path = sys.argv[4] if len(sys.argv) > 4 else f"{pretrained_model.split('/')[-1]}-finetuned"
MODEL_MAX_LENGTH = int(sys.argv[5]) if len(sys.argv) > 5 else 512

# === Load tokenizer + model ===
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelWithLMHead.from_pretrained(pretrained_model)

# === Load and combine text data from both files ===
with open(finetune_file_1, 'r', encoding='utf-8') as f1, open(finetune_file_2, 'r', encoding='utf-8') as f2:
    lines_1 = [line.strip() for line in f1 if line.strip()]
    lines_2 = [line.strip() for line in f2 if line.strip()]

lines = lines_1 + lines_2

# Shuffle combined lines for better mixing
random.shuffle(lines)

# === Build Dataset ===
dataset = Dataset.from_dict({"text": lines})
datasets = dataset.train_test_split(test_size=0.05)

# === Tokenize ===
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MODEL_MAX_LENGTH)

print("Begin tokenizing...")
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# === Data Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir=save_path,
    eval_strategy="steps",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=1.0,
    fp16=True,
    per_device_train_batch_size=8,
    gradient_checkpointing=True,
    gradient_accumulation_steps=10,
)
