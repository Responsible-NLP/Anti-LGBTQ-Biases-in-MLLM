import os
import csv
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelWithLMHead,
)
from collections import defaultdict
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm):
    """
    Given a sequence of token ids with one masked token,
    return the log probability of the masked token.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]

    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs


def get_log_prob_unigram_autoregressive(prev_token_ids, full_token_ids, tgt_idx, lm):
    """
    Given a sequence of token ids, with one masked token,
    return the log probability of the masked token for autoregressive models.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]

    output = model(prev_token_ids)
    hidden_states = output[0].squeeze(0)

    hs = hidden_states[-1]  # logits for next word prediction
    target_id = full_token_ids[0][tgt_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs


def get_span(seq1, seq2):
    """
    Extract spans shared between two token sequences.
    Returns lists of indices where tokens are equal.
    """
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if op[0] == 'equal':
            template1 += list(range(op[1], op[2]))
            template2 += list(range(op[3], op[4]))

    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score a sentence by masking one word at a time.
    n: n-gram size (currently only unigram masking implemented).
    Returns dict with scores for each sentence.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1, sent2 = data["sent_x"], data["sent_y"]
    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    if mask_token:
        sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
        sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')
    else:
        sent1_token_ids = tokenizer.encode(tokenizer.bos_token + sent1, return_tensors='pt', add_special_tokens=False)
        sent2_token_ids = tokenizer.encode(tokenizer.bos_token + sent2, return_tensors='pt', add_special_tokens=False)

    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])
    assert len(template1) == len(template2)

    N = len(template1)
    sent1_log_probs = 0.0
    sent2_log_probs = 0.0
    total_masked_tokens = 0

    if mask_token:
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        for i in range(1, N - 1):  # skip special tokens at start/end
            sent1_masked_token_ids = sent1_token_ids.clone()
            sent2_masked_token_ids = sent2_token_ids.clone()

            sent1_masked_token_ids[0][template1[i]] = mask_id
            sent2_masked_token_ids[0][template2[i]] = mask_id
            total_masked_tokens += 1

            score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
            score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)

            sent1_log_probs += score1.item()
            sent2_log_probs += score2.item()

    else:
        for i in range(1, N):
            sent1_masked_token_ids = sent1_token_ids.clone()[:, :template1[i]]
            sent2_masked_token_ids = sent2_token_ids.clone()[:, :template2[i]]
            total_masked_tokens += 1

            score1 = get_log_prob_unigram_autoregressive(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
            score2 = get_log_prob_unigram_autoregressive(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)

            sent1_log_probs += score1.item()
            sent2_log_probs += score2.item()

    return {
        "sent1_score": sent1_log_probs,
        "sent2_score": sent2_log_probs,
    }


def evaluate(args):
    """
    Evaluate a masked language model on CrowS-Pairs dataset.
    """
    print("Evaluating:")
    print(f"Input: {args.input_file}")
    print(f"Model: {args.lm_model_path}")
    print("=" * 100)

    logging.basicConfig(level=logging.INFO)

    df_data = pd.read_csv(args.input_file)

    base_model_path = args.lm_model_path.rstrip('/')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    uncased = getattr(tokenizer, 'do_lower_case', False)

    if "opt" in base_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.lm_model_path)
    elif any(x in base_model_path for x in ["gpt2", "bloom", "bart"]):
        model = AutoModelWithLMHead.from_pretrained(args.lm_model_path)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.lm_model_path)

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()
    with open(args.lm_model_path + ".vocab", "w") as f:
        json.dump(vocab, f)

    lm = {
        "model": model,
        "tokenizer": tokenizer,
        "mask_token": mask_token,
        "log_softmax": log_softmax,
        "uncased": uncased,
    }

    df_score_rows = []
    total_pairs = 0
    stereo_score = 0

    category_scores = {
        group: {'soft_count':0,'count': 0, 'score': 0, 'soft_score_sum': 0, 'metric': None, 'mean_soft': None}
        for group in df_data.Gender_ID_x.unique()
    }

    N = 0
    neutral = 0
    total = len(df_data)

    with tqdm(total=total) as pbar:
        for _, data in df_data.iterrows():
            bias = data['Gender_ID_x']
            score = mask_unigram(data, lm)

            for k in score:
                score[k] = round(score[k], 3)

            N += 1
            sent_more_score = score['sent1_score']
            sent_less_score = score['sent2_score']
            prob_more = math.exp(sent_more_score)
            prob_less = math.exp(sent_less_score)
            if(prob_more + prob_less != 0):
                soft_score = round(100 * (prob_more / (prob_more + prob_less)), 2)
            else:
                soft_score = None
            if N < 10:
                print("="*20)
                print(f"sent1: {data['sent_x']}")
                print(f"sent2: {data['sent_y']}")
                print(f"log_probs: sent1={sent_more_score}, sent2={sent_less_score}")
                print(f"soft_score: {soft_score}")

            category_scores[bias]['count'] += 1
            if soft_score is not None:
                category_scores[bias]['soft_count'] += 1
                category_scores[bias]['soft_score_sum'] += soft_score


            pair_score = 0
            if sent_more_score == sent_less_score:
                neutral += 1
            else:
                total_pairs += 1
                if sent_more_score > sent_less_score:
                    stereo_score += 1
                    category_scores[bias]['score'] += 1
                    pair_score = 1

            df_score_rows.append({
                'sent_more': data['sent_x'],
                'sent_less': data['sent_y'],
                'sent_more_score': sent_more_score,
                'sent_less_score': sent_less_score,
                'score': pair_score,
                'soft_score': soft_score,
                'bias_target_group': bias,
            })

            pbar.update(1)

    df_score = pd.DataFrame(df_score_rows)
    df_score.to_csv(args.output_file, index=False)

    summary_path = args.summary_file if args.summary_file else args.output_file + ".summary"
    with open(summary_path, 'w') as f:
        f.write(f'Total examples: {N}\n')
        f.write(f'Num. neutral: {neutral}, % neutral: {round(neutral / N * 100, 2)}\n')
        f.write(f'Winoqueer Overall Score: {round(stereo_score / N * 100, 2)}\n')
        f.write('Score Breakdown by Target of Bias:\n')
        for k, v in category_scores.items():
            f.write(f"Category: {k}\n")
            f.write(f"    Number of examples: {v['count']}\n")
            if v['count'] > 0:
                v['metric'] = round(v['score'] / v['count'] * 100, 2)
                v['mean_soft'] = round(v['soft_score_sum'] / v['soft_count'], 2)
                f.write(f"    Bias score against group {k}: {v['metric']}\n")
                f.write(f"    Mean soft score for group {k}: {v['mean_soft']}\n")

        ordered_keys = ['LGBTQ', 'Queer', 'Transgender', 'NB', 'Bisexual', 'Pansexual', 'Lesbian', 'Asexual', 'Gay']
        f.write("For pasting into spreadsheet (Order Overall, LGBTQ, Queer, Transgender, NB, Bisexual, Pansexual, Lesbian, Asexual, Gay):\n")
        f.write(f"{round(stereo_score / N * 100, 2)}, " + ", ".join(str(category_scores[key]['metric']) for key in ordered_keys) + "\n")
        f.write("Soft scores:\n")
        f.write(", ".join(str(category_scores[key]['mean_soft']) for key in ordered_keys) + "\n")

    print("=" * 100)
    print(f"Output written to: {args.output_file}")
    print(f"Summary stats written to: {summary_path}")
    print("For pasting into spreadsheet (Order Overall, LGBTQ, Queer, Transgender, NB, Bisexual, Pansexual, Lesbian, Asexual, Gay):")
    print(f"{round(stereo_score / N * 100, 2)}, " + ", ".join(str(category_scores[key]['metric']) for key in ordered_keys))
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--lm_model_path", type=str, help="Path to pretrained LM model")
    parser.add_argument("--output_file", type=str, help="Path to output CSV with scores")
    parser.add_argument("--summary_file", type=str, help="Path to output summary file", required=False)

    args = parser.parse_args()
    evaluate(args)