#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: eval.py
"""
 
import sys
import math
from rouge import Rouge

    
def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None: 
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def count(pred_tokens, gold_tokens, ngram, result):
    """
    count
    """
    cover_count, total_count = result
    pred_dict = get_dict(pred_tokens, ngram)
    gold_dict = get_dict(gold_tokens, ngram)
    cur_cover_count = 0
    cur_total_count = 0
    for token, freq in pred_dict.items():
        if gold_dict.get(token) is not None:
            gold_freq = gold_dict[token]
            cur_cover_count += min(freq, gold_freq)
        cur_total_count += freq
    result[0] += cur_cover_count
    result[1] += cur_total_count


def calc_bp(pair_list):
    """
    calc_bp
    """
    c_count = 0.0
    r_count = 0.0
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        c_count += len(pred_tokens)
        r_count += len(gold_tokens)
    bp = 1
    if c_count < r_count:
        bp = math.exp(1 - r_count / c_count)
    return bp 


def calc_cover_rate(pair_list, ngram):
    """
    calc_cover_rate
    """
    result = [0.0, 0.0] # [cover_count, total_count]
    for pair in pair_list:
        pred_tokens, gold_tokens = pair
        count(pred_tokens, gold_tokens, ngram, result)
    cover_rate = result[0] / result[1]
    return cover_rate 


def calc_bleu(pair_list):
    """
    calc_bleu
    """
    bp = calc_bp(pair_list)
    cover_rate1 = calc_cover_rate(pair_list, 1)
    cover_rate2 = calc_cover_rate(pair_list, 2)
    cover_rate3 = calc_cover_rate(pair_list, 3)
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    if cover_rate1 > 0:
        bleu1 = bp * math.exp(math.log(cover_rate1))
    if cover_rate2 > 0:
        bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
    if cover_rate3 > 0:
        bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
    return [bleu1, bleu2, bleu3]


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1 
        #if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    distinct3 = calc_distinct_ngram(pair_list, 3)
    pair_list = [[y,x] for x,y in pair_list]
    gold_distinct1 = calc_distinct_ngram(pair_list, 1)
    gold_distinct2 = calc_distinct_ngram(pair_list, 2)
    gold_distinct3 = calc_distinct_ngram(pair_list, 3)
    return distinct1, distinct2, distinct3, gold_distinct1, gold_distinct2, gold_distinct3


def calc_knowledge(data):
    """
    calc_f1
    """
    cue_token_total = 0.0
    sent_token_total = 0.0
    hit_token_total = 0.0
    for sent, cue in data:
        hit_set = sent & cue
        hit_token_total += len(hit_set)
        cue_token_total += len(cue)
        sent_token_total += len(sent)
    r = hit_token_total / cue_token_total
    p = hit_token_total / sent_token_total
    f1 = 2 * p * r / (p + r)
    return r, p, f1

def eval_function(results, stopwords=None):
    rouge = Rouge()
    sents = []
    pred_sets = []
    gold_sets = []
    for result in results:
        pred_tokens = result.preds.strip().split(" ")
        gold_tokens = result.tgt.strip().split(" ")
        pred_set = set(pred_tokens) - stopwords
        gold_set = set(gold_tokens) - stopwords
        cue_set = set(" ".join(result.cue).split(" ")) - stopwords
        pred_sets.append([pred_set, cue_set])
        gold_sets.append([gold_set, cue_set])
        sents.append([pred_tokens, gold_tokens])
    # calc f1
    pred_r, pred_p, pred_f1 = calc_knowledge(pred_sets)
    gold_r, gold_p, gold_f1 = calc_knowledge(gold_sets)
    # calc bleu
    bleu1, bleu2, bleu3 = calc_bleu(sents)
    scores = rouge.get_scores([result.preds for result in results], [result.tgt for result in results])
    # bleu1 = [score['rouge-1']['p'] for score in scores]
    # bleu2 = [score['rouge-2']['p'] for score in scores]
    # bleuL = [score['rouge-l']['p'] for score in scores]
    rouge1 = [score['rouge-1']['r'] for score in scores]
    rouge2 = [score['rouge-2']['r'] for score in scores]
    rougeL = [score['rouge-l']['r'] for score in scores]
    # bleu1 = sum(bleu1)/len(bleu1)
    # bleu2 = sum(bleu2)/len(bleu2)
    # bleuL = sum(bleuL)/len(bleuL)
    rouge1 = sum(rouge1)/len(rouge1)
    rouge2 = sum(rouge2)/len(rouge2)
    rougeL = sum(rougeL)/len(rougeL)
    # calc distinct
    distinct1, distinct2, distinct3, gold_distinct1, gold_distinct2, gold_distinct3 = calc_distinct(sents)

    return (pred_r, pred_p, pred_f1), (gold_r, gold_p, gold_f1), (bleu1, bleu2, bleu3), (rouge1, rouge2, rougeL), (distinct1, distinct2, distinct3), (gold_distinct1, gold_distinct2, gold_distinct3)
    # return (pred_r, pred_p, pred_f1), (gold_r, gold_p, gold_f1), (bleu1, bleu2, bleu3), (distinct1, distinct2, distinct3), (gold_distinct1, gold_distinct2, gold_distinct3)
