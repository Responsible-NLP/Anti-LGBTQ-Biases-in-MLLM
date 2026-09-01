import os
import json
import torch
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelWithLMHead
)

def mask_unigram(data, lm):
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    sent1 = data["sent_x"]
    sent2 = data["sent_y"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    sent1_tokens = tokenizer.encode(sent1, return_tensors="pt").to(model.device)
    sent2_tokens = tokenizer.encode(sent2, return_tensors="pt").to(model.device)

    # Example: just returning dummy scores; you'd integrate your masking logic here
    return {"sent1_score": torch.rand(1).item(), "sent2_score": torch.rand(1).item()}

def evaluate(args):
    print("="*100)
    print(f"Evaluating:")
    print(f"Input: {args.input_file}")
    print(f"Model: {args.lm_model_path}")
    print(f"Mode: {'masked' if args.masked else 'autoregressive'}")
    print("="*100)

    df_data = pd.read_csv(args.input_file)
    
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path, local_files_only=True)
    uncased = getattr(tokenizer, "do_lower_case", False)

    if args.masked:
        model = AutoModelForMaskedLM.from_pretrained(args.lm_model_path, local_files_only=True)
        mask_token = tokenizer.mask_token
    else:
        # GPT, OPT, etc
        try:
            model = AutoModelForCausalLM.from_pretrained(args.lm_model_path, local_files_only=True)
        except:
            model = AutoModelWithLMHead.from_pretrained(args.lm_model_path, local_files_only=True)
        mask_token = None  # not used

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    log_softmax = torch.nn.LogSoftmax(dim=0)

    lm = {
        "model": model,
        "tokenizer": tokenizer,
        "mask_token": mask_token,
        "log_softmax": log_softmax,
        "uncased": uncased
    }

    rows_list = []
    category_scores = {}
    N = 0
    stereo_score = 0
    neutral = 0

    with tqdm(total=len(df_data)) as pbar:
        for _, row in df_data.iterrows():
            identity_combined = str(row["Gender_ID_x"]).strip()
            identity_parts = [p.strip() for p in identity_combined.split(",")]

            score = mask_unigram(row, lm)
            sent_more_score = score["sent1_score"]
            sent_less_score = score["sent2_score"]
            soft_score = round(100 * (sent_less_score / (sent_more_score + sent_less_score + 1e-8)), 2)

            pair_score = 0
            if sent_more_score == sent_less_score:
                neutral += 1
            else:
                if sent_more_score > sent_less_score:
                    stereo_score += 1
                    pair_score = 1

            # Combined identity
            if identity_combined not in category_scores:
                category_scores[identity_combined] = {'count': 0, 'score': 0, 'soft_score_sum': 0}
            category_scores[identity_combined]['count'] += 1
            category_scores[identity_combined]['soft_score_sum'] += soft_score
            if pair_score == 1:
                category_scores[identity_combined]['score'] += 1

            # Individual identities
            for part in identity_parts:
                if part not in category_scores:
                    category_scores[part] = {'count': 0, 'score': 0, 'soft_score_sum': 0}
                category_scores[part]['count'] += 1
                category_scores[part]['soft_score_sum'] += soft_score
                if pair_score == 1:
                    category_scores[part]['score'] += 1

            rows_list.append({
                "sent_more": row["sent_x"],
                "sent_less": row["sent_y"],
                "sent_more_score": sent_more_score,
                "sent_less_score": sent_less_score,
                "score": pair_score,
                "soft_score": soft_score,
                "bias_target_group": identity_combined
            })

            N += 1
            pbar.update(1)

    df_out = pd.DataFrame(rows_list)
    df_out.to_csv(args.output_file, index=False)

    # Summary
    with open(args.summary_file or args.output_file + ".summary", "w") as f:
        f.write(f"Total examples: {N}\n")
        f.write(f"Num. neutral: {neutral}, % neutral: {round(neutral / N * 100, 2)}\n")
        f.write(f"Overall Stereo Score: {round(stereo_score / N * 100, 2)}\n")
        for k, v in category_scores.items():
            metric = round(v["score"] / v["count"] * 100, 2)
            mean_soft = round(v["soft_score_sum"] / v["count"], 2)
            f.write(f"Group: {k}, N: {v['count']}, Bias: {metric}, Soft: {mean_soft}\n")

    print("=" * 100)
    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--lm_model_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--masked", action="store_true", help="If set, run in masked LM mode; otherwise autoregressive")
    args = parser.parse_args()
    evaluate(args)
