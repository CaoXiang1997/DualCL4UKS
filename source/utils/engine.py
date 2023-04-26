#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/engine.py
"""

from email.policy import strict
import os
import time
import shutil
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast

from collections import defaultdict
from tensorboardX import SummaryWriter

from source.utils.metrics import EmbeddingMetrics
from source.utils.attribute_dict import AttributeDict
# from stop_words import get_stop_words
from string import punctuation
from source.utils.eval import eval_function


def padding(tensor_list):
    tensor_list = deepcopy(tensor_list)
    dim = tensor_list[0].dim()
    max_sizes = [0]*dim
    for tensor in tensor_list:
        for i in range(dim):
            if tensor.shape[i] > max_sizes[i]:
                max_sizes[i] = tensor.shape[i]
    for i in range(len(tensor_list)):
        for j in range(dim-1,-1,-1):
            shape = tensor_list[i].shape
            pad_shape = list(shape[:j]) + [max_sizes[j]-shape[j]] + list(shape[j+1:])
            zero = torch.zeros(size=pad_shape, dtype=tensor_list[i].dtype).to(tensor_list[i].device)
            tensor_list[i] = torch.cat([tensor_list[i], zero], dim=j)
    return tensor_list

def flat_batch(inputs):
    batch = AttributeDict()
    batch['src'], batch['tgt'], batch['cue'], batch['contrastive_matrix'] = [], [], [], []
    src_values, src_lens = [], []
    tgt_values, tgt_lens = [], []
    cue_values, cue_lens = [], []
    num_samples = []
    for i in range(len(inputs.src[0])):
        num_samples.append(0)
        for j in range(len(inputs.src[0][i])):
            if inputs.src[0][i][j].sum().item() != 0:
                num_samples[-1] += 1
                src_values.append(inputs.src[0][i][j])
                src_lens.append(inputs.src[1][i][j])
                tgt_values.append(inputs.tgt[0][i][j])
                tgt_lens.append(inputs.tgt[1][i][j])
                cue_values.append(inputs.cue[0][i])
                cue_lens.append(inputs.cue[1][i])
    src_values = torch.stack(src_values, dim=0)
    src_lens = torch.stack(src_lens, dim=0)
    batch['src'] = (src_values, src_lens)
    tgt_values = torch.stack(tgt_values, dim=0)
    tgt_lens = torch.stack(tgt_lens, dim=0)
    batch['tgt'] = (tgt_values, tgt_lens)
    cue_values = torch.stack(cue_values, dim=0)
    cue_lens = torch.stack(cue_lens, dim=0)
    batch['cue'] = (cue_values, cue_lens)
    for i in range(len(num_samples)):
        for j in range(num_samples[i]):                
            line = torch.Tensor([0.0]*sum(num_samples[0:i]) + [1.0]*num_samples[i]).to(inputs.src[0].device)
            line[sum(num_samples[0:i]) + j] = 0.0
            batch['contrastive_matrix'].append(line)
    batch['contrastive_matrix'] = torch.stack(padding(batch['contrastive_matrix']), dim=0)
    return batch



class MetricsManager(object):
    """
    MetricsManager
    """

    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0
        self.num_words = 0

    def update(self, metrics):
        """
        update
        """
        num_samples = metrics.pop("num_samples", 1)
        self.num_samples += num_samples

        for key, val in metrics.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.metrics_cum[key] += np.array(val * num_samples)
            self.metrics_val[key] = val

    def clear(self):
        """
        clear
        """
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def get(self, name):
        """
        get
        """
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        """
        report_val
        """
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{} {:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
            if key=='nll':
                metric_str = "PPL {:.3f}".format(np.exp(min(val, 100)))
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        """
        report_cum
        """
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{} {:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)
            
            if key == 'nll':
                ppl = np.exp(min(val / self.num_samples, 100))
                metric_str = "PPL {:.3f}".format(ppl)
                metric_strs.append(metric_str)
            
            if key == 'draft':
                if num_words is not None:
                    ppl = np.exp(min(val / num_words, 100))
                else:
                    ppl = np.exp(min(val / self.num_samples, 100))
                metric_str = "DRAFT_PPL {:.3f}".format(ppl)
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs


def evaluate(model, data_iter, multi_turn=False, use_posterior=False):
    """
    evaluate
    """
    model.eval()
    mm = MetricsManager()
    with torch.no_grad():
        i = 0
        for inputs in tqdm(data_iter):
            if i >= 1:
                break
            i += 1
            if not multi_turn:
                inputs = flat_batch(inputs)
            _, metrics = model(inputs, training=False, epoch=-1, use_posterior=use_posterior)
            mm.update(metrics)
    # data_iter.num_samples = 0
    return mm

class FakeLogger():
    def info(self, args):
        pass
    

class Trainer(object):
    """
    Trainer
    """

    def __init__(self,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 test_iter=None,
                 generator=None,
                 valid_metric_name="-loss",
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_interval=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 save_summary=False,
                 pretrain_epoch=5,
                 multi_turn=False,
                 annealing_steps=None,
                 use_float16=False,
                 parallel=False,
                 local_rank=-1,
                 train_sampler=None):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter
        self.generator = generator
        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_interval = valid_interval
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary
        self.multi_turn = multi_turn
        self.annealing_steps = annealing_steps
        self.use_float16 = use_float16
        self.parallel = parallel
        self.local_rank = local_rank
        self.train_sampler = train_sampler
        self.enable_write = ((not self.parallel) or self.local_rank==0)

        self.logger = FakeLogger()
        if (not self.parallel) or self.local_rank==0:
            self.logger = logger

        if self.save_summary and self.enable_write:
            self.train_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "train"))
            self.valid_writer = SummaryWriter(
                os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float(
            "inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.pretrain_epoch = pretrain_epoch
        self.batch_num = 0
        self.step = 0

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaulation " + "-" * 33

    def summarize_train_metrics(self, metrics, global_step):
        """
        summarize_train_metrics
        """
        for key, val in metrics.items():
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                self.train_writer.add_scalar(key, val, global_step)

    def summarize_valid_metrics(self, metrics_mm, global_step):
        """
        summarize_valid_metrics
        """
        for key in metrics_mm.metrics_cum.keys():
            val = metrics_mm.get(key)
            self.valid_writer.add_scalar(key, val, global_step)

    def train_epoch(self):
        """
        train_epoch
        """
        train_mm = MetricsManager()
        num_batches = len(self.train_iter)
        self.logger.info(self.train_start_message)

        store_best = False
        # while self.train_iter.num_samples < self.train_iter.num_total_samples:
            # inputs = next(self.train_iter)
        if self.parallel:
            self.train_sampler.set_epoch(self.epoch)       
        self.epoch += 1

        for batch_id, inputs in enumerate(self.train_iter, 1):
            self.model.train()
            start_time = time.time()
            # Do a training iteration
            if not self.multi_turn:
                inputs = flat_batch(inputs)
            if self.use_float16:
                with autocast():
                    _, metrics = self.model(inputs, epoch=self.epoch, training=True)
                    self.optimizer.zero_grad()
                    metrics.loss.backward()
                    if self.grad_clip is not None and self.grad_clip > 0:
                        clip_grad_norm_(parameters=self.model.parameters(),
                                        max_norm=self.grad_clip)
                    self.optimizer.step()
            else:
                _, metrics = self.model(inputs, epoch=self.epoch, training=True)
                self.optimizer.zero_grad()
                metrics.loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    clip_grad_norm_(parameters=self.model.parameters(),
                                    max_norm=self.grad_clip)
                self.optimizer.step()
            
            elapsed = time.time() - start_time

            train_mm.update(metrics)
            self.step += 1
            self.batch_num += 1
            batch_id += 1
            if batch_id % self.log_steps == 0:
                message_prefix = "[Train][{:2d}][{}/{}]".format(
                    self.epoch, batch_id, num_batches)
                metrics_message = train_mm.report_val()
                message_posfix = "TIME {:.2f}".format(elapsed)
                self.logger.info("   ".join(
                    [message_prefix, metrics_message, message_posfix]))
                if self.save_summary:
                    self.summarize_train_metrics(metrics, self.batch_num)
            
            if (not self.parallel) or self.local_rank==0:
                if self.epoch > self.pretrain_epoch:
                    if self.valid_interval=="steps" and batch_id % self.valid_steps == 0:
                        self.model.eval()
                        self.logger.info(self.valid_start_message)
                        valid_mm = evaluate(self.model, self.valid_iter, multi_turn=self.multi_turn, use_posterior=True)

                        message_prefix = "[Valid][{:2d}][{}/{}]".format(
                            self.epoch, batch_id, num_batches)
                        metrics_message = valid_mm.report_cum()
                        self.logger.info("   ".join([message_prefix, metrics_message]))

                        if self.save_summary:
                            self.summarize_valid_metrics(valid_mm, self.batch_num)

                        cur_valid_metric = valid_mm.get(self.valid_metric_name)
                        if self.is_decreased_valid_metric:
                            is_best = cur_valid_metric < self.best_valid_metric
                        else:
                            is_best = cur_valid_metric > self.best_valid_metric
                        if is_best:
                            store_best = True
                            self.best_valid_metric = cur_valid_metric
                            self.save(is_best)
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step(cur_valid_metric)
                        
                        test_mm = evaluate(self.model, self.test_iter, multi_turn=self.multi_turn)

                        message_prefix = "[TEST][{:2d}][{}/{}]".format(
                            self.epoch, batch_id, num_batches)
                        metrics_message = test_mm.report_cum()
                        self.logger.info("   ".join([message_prefix, metrics_message]))
                        
                        self.logger.info("-" * 85 + "\n")
        is_best = False
        if self.valid_interval=="epoch":
            if self.epoch > self.pretrain_epoch:
                self.model.eval()
                self.logger.info(self.valid_start_message)
                valid_mm = evaluate(self.model, self.valid_iter, multi_turn=self.multi_turn, step=self.step, annealing_steps=self.annealing_steps)

                message_prefix = "[Valid][{:2d}][{}/{}]".format(
                    self.epoch, batch_id, num_batches)
                metrics_message = valid_mm.report_cum()
                self.logger.info("   ".join([message_prefix, metrics_message]))

                if self.save_summary:
                    self.summarize_valid_metrics(valid_mm, self.batch_num)

                cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(cur_valid_metric)
                self.logger.info("-" * 85 + "\n")
            if is_best:
                store_best = True
                self.best_valid_metric = cur_valid_metric
                self.save(is_best)
        
        self.stop_training = (not store_best) and (self.epoch > self.pretrain_epoch)
    
    def train(self):
        """
        train
        """
        # valid_mm = evaluate(self.model, self.valid_iter, multi_turn=self.multi_turn, step=self.step, annealing_steps=self.annealing_steps, use_posterior=True)
        # self.logger.info(valid_mm.report_cum())
        for _ in range(self.epoch):
            for _ in self.train_iter:
                continue
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()
            if self.enable_write and self.stop_training:
                break

    def save(self, is_best=False):
        """
        save
        """
        # model_file = os.path.join(
        #     self.save_dir, "state_epoch_{}.model".format(self.epoch))
        # torch.save(self.model.state_dict(), model_file)
        # self.logger.info("Saved model state to '{}'".format(model_file))

        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        # train_file = os.path.join(
        #     self.save_dir, "state_epoch_{}.train".format(self.epoch))
        # torch.save(train_state, train_file)
        # self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best and self.enable_write:
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            # shutil.copy(model_file, best_model_file)
            # shutil.copy(train_file, best_train_file)
            torch.save(self.model.state_dict(), best_model_file)
            torch.save(train_state, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_prefix):
        """
        load
        """
        model_file = "{}.model".format(file_prefix)
        train_file = "{}.train".format(file_prefix)

        model_state_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict, strict=False)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


def evaluate_generation(generator,
                        data_iter,
                        save_file=None,
                        response_file=None,
                        num_batches=None,
                        verbos=False,
                        stopwords=None):
    """
    evaluate_generation
    """
    results = generator.generate(batch_iter=data_iter,
                                 num_batches=num_batches)
    # stopwords = set(get_stop_words('en')) | set(punctuation)
    pred_k, gold_k, bleu, rouge, pre_distinct, gold_distinct = eval_function(results, stopwords=stopwords)
    # pred_k, gold_k, bleu, pre_distinct, gold_distinct = eval_function(results, stopwords=stopwords)
    pred_r, pred_p, pred_f1 = pred_k
    gold_r, gold_p, gold_f1 = gold_k
    bleu1, bleu2, bleu3 = bleu
    rouge1, rouge2, rougeL = rouge
    pred_distinct1, pred_distinct2, pred_distinct3 = pre_distinct
    gold_distinct1, gold_distinct2, gold_distinct3 = gold_distinct

    report_message = []
    report_message.append("Prediction:\nAvg_Len {:.3f}".format(np.average([len(result.preds[0].split(" ")) for result in results])))
    report_message.append("Bleu-1/2/3 {:.4f} {:.4f} {:.4f}".format(bleu1, bleu2, bleu3))
    report_message.append("ROUGE-1/2/L {:.4f} {:.4f} {:.4f}".format(rouge1, rouge2, rougeL))
    report_message.append("Dist-1/2/3 {:.4f} {:.4f} {:.4f}".format(pred_distinct1, pred_distinct2, pred_distinct3))
    report_message.append("Knowledge-R/P/F1 {:.4f} {:.4f} {:.4f}".format(pred_r, pred_p, pred_f1))
    report_message = "   ".join(report_message)

    target_message = []
    target_message.append("Target:\nAvg_Len {:.3f}".format(np.average([len(result.tgt.split(" ")) for result in results])))
    target_message.append("Dist-1/2/3 {:.4f} {:.4f} {:.4f}".format(gold_distinct1, gold_distinct2, gold_distinct3))
    target_message.append("Knowledge-R/P/F1 {:.4f} {:.4f} {:.4f}".format(gold_r, gold_p, gold_f1))
    target_message = "   ".join(target_message)

    message = report_message + "\n" + target_message

    if save_file is not None:
        write_results(results, save_file, response_file)
        print("Saved generation results to '{}'".format(save_file))
    if verbos:
        print(message)
    else:
        return message


def write_results(results, results_file, response_file):
    """
    write_results
    """
    with open(results_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write("{}\t{}\t{}\n".format(' '.join(result.cue), result.src, result.preds))
    with open(response_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.preds + '\n')

