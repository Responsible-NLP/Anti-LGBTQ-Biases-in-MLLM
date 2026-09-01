import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import sys
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


# ARGUMENTS
# sys.argv[1]: pretrained model path
# sys.argv[2]: save path (optional)
# sys.argv[3]: model_max_length (optional, default 512)

pretrained_model = sys.argv[1]
finetune_file_1 = "queer_articles_finetune.txt"
finetune_file_2 = "queer_posts_finetune.txt"
save_path = sys.argv[2] if len(sys.argv) > 2 else f"{pretrained_model.split('/')[-1]}-finetuned"
MODEL_MAX_LENGTH = int(sys.argv[3]) if len(sys.argv) > 3 else 512

# Explicit model mask mapping
model_mask_map = {
    "german_bert": True,
    "multi_bert": True,
    "xlm-roberta-base": True,
    "gpt2": False,
    "bloom-560m": False,
    "opt-350m": False,
    "xlm-clm-ende-1024": False,
    "xlm-mlm-ende-1024": True,
}

model_name = pretrained_model.split("/")[-1].lower()
mlm = model_mask_map.get(model_name, None)
if mlm is None:
    # fallback heuristic if not in the map
    mlm = any(x in model_name for x in ["bert", "roberta", "albert", "distilbert", "electra", "xlm", "bart"])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model based on mlm flag
if mlm:
    model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
else:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)

# Load and combine text data from both files
with open(finetune_file_1, 'r', encoding='utf-8') as f1, open(finetune_file_2, 'r', encoding='utf-8') as f2:
    lines_1 = [line.strip() for line in f1 if line.strip()]
    lines_2 = [line.strip() for line in f2 if line.strip()]

lines = lines_1 + lines_2
random.shuffle(lines)

dataset = Dataset.from_dict({"text": lines})
datasets = dataset.train_test_split(test_size=0.05)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MODEL_MAX_LENGTH)

print("Begin tokenizing...")
tokenized_datasets = datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

training_args = TrainingArguments(
    output_dir=save_path,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=3,
    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=10,
    logging_steps=100,
    gradient_checkpointing=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
