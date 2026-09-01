import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import subprocess

# List of pretrained models to finetune (replace with your actual model paths/names)
models = [
    "german_bert",
    "multi_bert",
    "opt-350m",
    "bloom-560m",
    "gpt-2",
    "xlm-roberta-base",
    "xlm-clm-ende-1024",
    "xlm-mlm-ende-1024"
]

# Optional: model max length for tokenization
model_max_length = 512

for model_name in models:
    save_path = f"{model_name}-finetuned"
    print(f"Starting finetuning for {model_name}...")
    subprocess.run([
        "python", "finetune.py",
        model_name,
        save_path,
        str(model_max_length)
    ], check=True)
    print(f"Finished finetuning {model_name}\n")
