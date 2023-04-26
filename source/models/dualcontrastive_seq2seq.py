#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/knowledge_seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import math

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.hgfu_rnn_decoder import RNNDecoder
from source.modules.decoders.rnn_decoder import RNNDecoder
from source.modules.decoders.hgfu_rnn_decoder import RNNDecoder as HGFUDecoder
# from source.utils.criterions import NLLLoss
from torch.nn import NLLLoss
from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.utils.metrics import attn_accuracy
from source.utils.metrics import perplexity
from source.modules.attention import Attention


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class DualContrastiveSeq2Seq(BaseModel):
    """
    DualContrastiveSeq2Seq
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, padding_idx=None,
                 num_layers=1, bidirectional=True, attn_mode="mlp", attn_hidden_size=None,
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_bow=False,
                 use_kd=False, use_dssm=False, use_posterior=False, posterior_detach=False, weight_control=False,
                 use_pg=False, use_gs=False, concat=False, pretrain_epoch=0, alpha=0.0, beta=1.0, loss_form="kl_loss"):
        super(DualContrastiveSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_bow = use_bow
        self.use_dssm = use_dssm
        self.weight_control = weight_control
        self.use_kd = use_kd
        self.use_pg = use_pg
        self.use_gs = use_gs
        self.use_posterior = use_posterior
        self.posterior_detach = posterior_detach
        self.pretrain_epoch = pretrain_epoch
        self.baseline = 0
        self.concat = concat
        self.loss_form = loss_form
        self.alpha = alpha
        self.beta = beta

        self.enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  embedder=self.enc_embedder, num_layers=self.num_layers,
                                  bidirectional=self.bidirectional, dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

        if self.tie_embedding:
            assert self.src_vocab_size == self.tgt_vocab_size
            self.dec_embedder = self.enc_embedder
            self.knowledge_embedder = self.enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size, padding_idx=self.padding_idx)
            knowledge_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                          embedding_dim=self.embed_size,
                                          padding_idx=self.padding_idx)

        self.knowledge_encoder = RNNEncoder(input_size=self.embed_size,
                                            hidden_size=self.hidden_size,
                                            embedder=self.knowledge_embedder,
                                            num_layers=self.num_layers,
                                            bidirectional=self.bidirectional,
                                            dropout=self.dropout)

        self.prior_attention = Attention(query_size=self.hidden_size,
                                         memory_size=self.hidden_size,
                                         hidden_size=self.hidden_size,
                                         mode=self.attn_mode)

        self.posterior_attention = Attention(query_size=self.hidden_size,
                                             memory_size=self.hidden_size,
                                             hidden_size=self.hidden_size,
                                             mode=self.attn_mode)

        if self.concat:
            self.decoder = RNNDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                    output_size=self.tgt_vocab_size, embedder=self.dec_embedder,
                                    num_layers=self.num_layers, attn_mode=self.attn_mode,
                                    memory_size=self.hidden_size, feature_size=self.hidden_size,
                                    dropout=self.dropout)
        else:
            self.decoder = HGFUDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                    output_size=self.tgt_vocab_size, embedder=self.dec_embedder,
                                    num_layers=self.num_layers, attn_mode=self.attn_mode,
                                    memory_size=self.hidden_size, feature_size=None,
                                    dropout=self.dropout, concat=concat)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        if self.use_bow:
            self.bow_output_layer = nn.Sequential(
                nn.Linear(in_features=self.hidden_size,
                          out_features=self.hidden_size),
                nn.Tanh(),
                nn.Linear(in_features=self.hidden_size,
                          out_features=self.tgt_vocab_size),
                nn.LogSoftmax(dim=-1))

        if self.use_dssm:
            self.dssm_project = nn.Linear(in_features=self.hidden_size,
                                          out_features=self.hidden_size)
            self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if self.use_kd:
            self.knowledge_dropout = nn.Dropout()

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, training=False, use_posterior=False):
        """
        encode
        """
        outputs = Pack()
        enc_inputs = _, lengths = inputs.src[0][:, 1:-1], inputs.src[1]-2
        enc_outputs, enc_hidden = self.encoder(enc_inputs)

        if self.with_bridge:
            enc_hidden = self.bridge(enc_hidden)

        # knowledge
        batch_size, sent_num, sent = inputs.cue[0].size()
        tmp_len = inputs.cue[1]
        tmp_len[tmp_len > 0] -= 2
        cue_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], tmp_len.view(-1)
        cue_enc_outputs, cue_enc_hidden = self.knowledge_encoder(
            cue_inputs)
        cue_outputs = cue_enc_hidden[-1].view(batch_size, sent_num, -1)
        # Attention
        weighted_cue, cue_attn = self.prior_attention(query=enc_hidden[-1].unsqueeze(1),
                                                      memory=cue_outputs,
                                                      mask=inputs.cue[1].eq(0))
        cue_attn = cue_attn.squeeze(1)
        outputs.add(prior_attn=cue_attn)
        indexs = cue_attn.max(dim=1)[1]
        # hard attention
        if self.use_gs:
            knowledge = cue_outputs.gather(1,
                                           indexs.view(-1, 1, 1).repeat(1, 1, cue_outputs.size(-1)))
        else:
            knowledge = weighted_cue

        if self.use_bow:
            bow_logits = self.bow_output_layer(knowledge)
            outputs.add(bow_logits=bow_logits)
        
        if self.use_posterior and use_posterior:
            tgt_enc_inputs = inputs.tgt[0][:, 1:-1], inputs.tgt[1]-2
            _, tgt_enc_hidden = self.knowledge_encoder(tgt_enc_inputs)
            posterior_weighted_cue, posterior_attn = self.posterior_attention(
                query=tgt_enc_hidden[-1].unsqueeze(1),
                memory=cue_outputs,
                mask=inputs.cue[1].eq(0))
            posterior_attn = posterior_attn.squeeze(1)
            outputs.add(posterior_attn=posterior_attn)
            # Gumbel Softmax
            if self.use_gs:
                gumbel_attn = F.gumbel_softmax(
                    torch.log(posterior_attn + 1e-10), 0.1, hard=True)
                outputs.add(gumbel_attn=gumbel_attn)
                knowledge = torch.bmm(gumbel_attn.unsqueeze(1), cue_outputs)
                indexs = gumbel_attn.max(-1)[1]
            else:
                knowledge = posterior_weighted_cue
                indexs = posterior_attn.max(dim=1)[1]

            if self.use_dssm:
                dssm_knowledge = self.dssm_project(knowledge)
                outputs.add(dssm=dssm_knowledge)
                outputs.add(reply_vec=tgt_enc_hidden[-1])
                # neg sample
                neg_idx = torch.arange(
                    enc_inputs[1].size(0)).type_as(enc_inputs[1])
                neg_idx = (neg_idx + 1) % neg_idx.size(0)
                neg_tgt_enc_inputs = tgt_enc_inputs[0][neg_idx], tgt_enc_inputs[1][neg_idx]
                _, neg_tgt_enc_hidden = self.knowledge_encoder(neg_tgt_enc_inputs)
                pos_logits = (enc_hidden[-1] * tgt_enc_hidden[-1]).sum(dim=-1)
                neg_logits = (enc_hidden[-1] *
                              neg_tgt_enc_hidden[-1]).sum(dim=-1)
                outputs.add(pos_logits=pos_logits, neg_logits=neg_logits)
        elif training:
            if self.use_gs:
                gumbel_attn = F.gumbel_softmax(
                    torch.log(cue_attn + 1e-10), 0.1, hard=True)
                knowledge = torch.bmm(gumbel_attn.unsqueeze(1), cue_outputs)
                indexs = gumbel_attn.max(-1)[1]
            else:
                knowledge = weighted_cue

        outputs.add(indexs=indexs)
        if 'index' in inputs.keys():
            outputs.add(attn_index=inputs.index)

        if self.use_kd:
            knowledge = self.knowledge_dropout(knowledge)

        if self.weight_control:
            weights = (enc_hidden[-1] * knowledge.squeeze(1)).sum(dim=-1)
            weights = self.sigmoid(weights)
            # norm in batch
            # weights = weights / weights.mean().item()
            outputs.add(weights=weights)
            knowledge = knowledge * \
                weights.view(-1, 1, 1).repeat(1, 1, knowledge.size(-1))

        if self.concat:
            dec_init_state = self.decoder.initialize_state(
                hidden=enc_hidden,
                attn_memory=enc_outputs if self.attn_mode else None,
                memory_lengths=lengths if self.attn_mode else None,
                feature=knowledge)
        else:
            dec_init_state = self.decoder.initialize_state(
                hidden=enc_hidden,
                attn_memory=enc_outputs if self.attn_mode else None,
                memory_lengths=lengths if self.attn_mode else None,
                knowledge=knowledge)
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def collect_metrics(self, outputs, target, contrastive_matrix, epoch=-1, 
                        step=-1, use_posterior=False):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits = outputs.logits
        if isinstance(self.nll_loss, torch.nn.NLLLoss):
            nll = self.nll_loss(logits.transpose(1, -1), target)
        else:
            nll = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=nll, acc=acc)

        if self.use_posterior and use_posterior:
            
            if self.loss_form == "kl_loss":
                kl_loss = outputs.posterior_attn.detach() * (torch.log(outputs.posterior_attn.detach() + 1e-10) - torch.log(outputs.prior_attn + 1e-10))
            elif self.loss_form == "js_loss":
                avg_p = (outputs.posterior_attn.detach() + outputs.prior_attn) / 2
                kl_loss = outputs.posterior_attn.detach() * (torch.log(outputs.posterior_attn.detach() + 1e-10) - torch.log(avg_p + 1e-10)) / 2
                kl_loss += outputs.prior_attn * (torch.log(outputs.prior_attn + 1e-10) - torch.log(avg_p + 1e-10)) / 2
            elif self.loss_form == "dual_kl_loss":
                kl_loss = outputs.posterior_attn.detach() * (torch.log(outputs.posterior_attn.detach() + 1e-10) - torch.log(outputs.prior_attn + 1e-10)) / 2
                kl_loss += outputs.prior_attn * (torch.log(outputs.prior_attn + 1e-10) - torch.log(outputs.posterior_attn.detach() + 1e-10)) / 2
            
            kl_loss = kl_loss.mean()
            metrics.add(kl=kl_loss)
            
            sum_contrastive_matrix = contrastive_matrix.sum(1, keepdim=True)
            sum_contrastive_matrix[sum_contrastive_matrix==0] = 1

            avg_log_prior = (contrastive_matrix.unsqueeze(-1) * torch.log(outputs.prior_attn + 1e-10).unsqueeze(0).repeat(num_samples,1,1)).sum(1) / sum_contrastive_matrix
            prior_contrastive_loss = outputs.posterior_attn * (torch.log(outputs.prior_attn + 1e-10) - avg_log_prior)

            avg_log_post = (contrastive_matrix.unsqueeze(-1) * torch.log(outputs.posterior_attn + 1e-10).unsqueeze(0).repeat(num_samples,1,1)).sum(1) / sum_contrastive_matrix
            post_contrastive_loss = outputs.prior_attn * (torch.log(outputs.posterior_attn + 1e-10) - avg_log_post)
            
            contrastive_loss = self.alpha * prior_contrastive_loss + self.beta * post_contrastive_loss
            prior_contrastive_loss = prior_contrastive_loss.mean()
            metrics.add(prior_contrastive=prior_contrastive_loss)
            post_contrastive_loss = post_contrastive_loss.mean()
            metrics.add(post_contrastive=post_contrastive_loss)
            contrastive_loss = contrastive_loss.mean()
            metrics.add(contrastive=contrastive_loss)

        if self.use_bow:
            bow_logits = outputs.bow_logits
            bow_labels = target[:, :-1]
            bow_logits = bow_logits.repeat(1, bow_labels.size(-1), 1)
            if isinstance(self.nll_loss, torch.nn.NLLLoss):
                bow = self.nll_loss(bow_logits.transpose(1, -1), bow_labels)
            else:
                bow = self.nll_loss(bow_logits, bow_labels)
            loss += bow
            metrics.add(bow=bow)
        
        if epoch == -1 or epoch > self.pretrain_epoch or self.use_bow is False:
            loss += nll
            if self.use_posterior and use_posterior:
                loss += kl_loss
                loss += contrastive_loss
        metrics.add(loss=loss)
        return metrics

    def forward(self, inputs, training=False, epoch=-1, use_posterior=True):
        """
        forward
        """
        dec_inputs = inputs.tgt[0][:, :-1], inputs.tgt[1] - 1
        target = inputs.tgt[0][:, 1:]
        contrastive_matrix = inputs.contrastive_matrix
        outputs, dec_init_state = self.encode(inputs, training=training, use_posterior=use_posterior)
        logits, _ = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=logits)
        metrics = self.collect_metrics(outputs, target, contrastive_matrix, epoch=epoch, use_posterior=use_posterior)
        return outputs, metrics