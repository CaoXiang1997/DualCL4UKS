#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/generator.py
"""

import torch
from source.utils.engine import flat_batch

from source.utils.misc import sequence_mask
from source.utils.misc import list2tensor
from source.utils.misc import Pack
from source.utils.engine import flat_batch
from tqdm import tqdm

class Generator(object):
    """
    Generator
    """
    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 cue_field=None,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False,
                 multi_turn=False):
        self.model = model.cuda() if use_gpu else model
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.cue_field = cue_field
        self.max_length = max_length
        self.ignore_unk = ignore_unk
        self.length_average = length_average
        self.use_gpu = use_gpu
        self.multi_turn = multi_turn
        if tgt_field.tokenizer is not None:
            self.PAD = tgt_field.tokenizer.pad_token_id
            self.UNK = tgt_field.tokenizer.unk_token_id
            self.BOS = tgt_field.tokenizer.cls_token_id
            self.EOS = tgt_field.tokenizer.sep_token_id
            self.V = tgt_field.tokenizer.vocab_size
        else:
            self.PAD = tgt_field.stoi[tgt_field.pad_token]
            self.UNK = tgt_field.stoi[tgt_field.unk_token]
            self.BOS = tgt_field.stoi[tgt_field.bos_token]
            self.EOS = tgt_field.stoi[tgt_field.eos_token]
            self.V = tgt_field.vocab_size

    def forward(self, inputs, enc_hidden=None):
        """
        forward
        """
        # switch the model to evaluate mode
        self.model.eval()

        with torch.no_grad():
            enc_outputs, dec_init_state = self.model.encode(inputs, training=False, use_posterior=False)
            preds, lens, scores = self.decode(dec_init_state)

        return enc_outputs, preds, lens, scores

    def decode(self, dec_state):
        """
        decode
        """
        return NotImplementedError

    def generate(self, batch_iter, num_batches=None):
        """
        generate
        """
        results = []
        batch_cnt = 0
        # while batch_iter.num_samples < batch_iter.num_total_samples:
        #     batch = next(batch_iter)
        i = 0
        for batch in tqdm(batch_iter):
            if i >= 1:
                break
            i += 1
            if not self.multi_turn:
                batch = flat_batch(batch)
            enc_outputs, preds, lengths, scores = self.forward(
                inputs=batch, enc_hidden=None)

            # denumericalization
            src = batch.src[0]
            tgt = batch.tgt[0]
            preds = preds.reshape([src.shape[0], -1, preds.shape[-1]])
            src = self.src_field.denumericalize(src)
            tgt = self.tgt_field.denumericalize(tgt)
            preds = self.tgt_field.denumericalize(preds.squeeze(1))
            
            if 'cue' in batch:
                cue = self.tgt_field.denumericalize(batch.cue[0])

            flattened_src = []
            flattened_tgt = []
            flattened_preds = []
            flattened_cue = []
            for i in range(len(src)):
                if isinstance(src[i], list):
                    for j in range(len(src[i])):
                        if src[i][j] != "":
                            flattened_src.append(src[i][j])
                            flattened_tgt.append(tgt[i][j])
                            flattened_preds.append(preds[i][j])
                            flattened_cue.append(cue[i])
                else:
                    if src[i] != "":
                        flattened_src.append(src[i])
                        flattened_tgt.append(tgt[i])
                        flattened_preds.append(preds[i])
                        flattened_cue.append(cue[i])
            src = flattened_src
            tgt = flattened_tgt
            preds = flattened_preds
            cue = flattened_cue
            
            scores = scores.tolist()


            enc_outputs.add(src=src, tgt=tgt, preds=preds, cue=cue, scores=scores)
            enc_outputs = enc_outputs.flatten()
            result_batch = [Pack() for _ in range(len(enc_outputs))]            
            for i in range(len(result_batch)):                
                for beam_size,v in enc_outputs[i].items():
                    if not isinstance(v, torch.Tensor):
                        result_batch[i][beam_size] = v
            results += result_batch
            batch_cnt += 1
            if batch_cnt == num_batches:
                break
        # batch_iter.num_samples = 0
        return results

    def interact(self, src, cue=None):
        """
        interact
        """
        if src == "":
            return None

        inputs = Pack()
        src = self.src_field.numericalize([src])
        inputs.add(src=list2tensor(src))

        if cue is not None:
            cue = self.cue_field.numericalize([cue])
            inputs.add(cue=list2tensor(cue))
        if self.use_gpu:
            inputs = inputs.cuda()
        _, preds, _, _ = self.forward(inputs=inputs)

        pred = self.tgt_field.denumericalize(preds[0][0])

        return pred


class BeamGenerator(Generator):
    """
    BeamGenerator
    """
    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 cue_field=None,
                 beam_size=1,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False,
                 multi_turn=False):
        super(BeamGenerator, self).__init__(model,
                                            src_field,
                                            tgt_field,
                                            cue_field=cue_field,
                                            max_length=max_length,
                                            ignore_unk=ignore_unk,
                                            length_average=length_average,
                                            use_gpu=use_gpu,
                                            multi_turn=multi_turn)
        self.beam_size = beam_size

    def decode(self, dec_state):
        """
        decode
        """
        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor

        batch_size = dec_state.get_batch_size()

        # [[0], [beam_size*1], [beam_size*2], ..., [beam_size*(batch_size-1)]]
        self.pos_index = (long_tensor_type(range(batch_size)) * self.beam_size).view(-1, 1)

        # Inflate the initial hidden states to be of size: (batch_size*beam_size, H)
        dec_state = dec_state.inflate(self.beam_size)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top beam_size
        sequence_scores = long_tensor_type(batch_size * self.beam_size).float()
        sequence_scores.fill_(-float('inf'))
        sequence_scores.index_fill_(0, long_tensor_type(
            [i * self.beam_size for i in range(batch_size)]), 0.0)

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * batch_size * self.beam_size)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for t in range(1, self.max_length+1):
            # Run the RNN one step forward
            output, dec_state, attn = self.model.decode(input_var, dec_state)

            log_softmax_output = output.squeeze(1)

            # To get the full sequence scores for the new candidates, add the
            # local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = sequence_scores.unsqueeze(1).repeat(1, self.V)
            if self.length_average and t > 1:
                sequence_scores = sequence_scores * \
                    (1 - 1/t) + log_softmax_output / t
            else:
                sequence_scores += log_softmax_output

            scores, candidates = sequence_scores.view(
                batch_size, -1).topk(self.beam_size, dim=1)

            # Reshape input = (batch_size*beam_size, 1) and sequence_scores = (batch_size*beam_size)
            input_var = (candidates % self.V)
            sequence_scores = scores.view(batch_size * self.beam_size)

            input_var = input_var.view(batch_size * self.beam_size)

            # Update fields for next timestep
            predecessors = (
                candidates // self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.beam_size)

            dec_state = dec_state.index_select(predecessors)

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero(as_tuple=True)[0].numel() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            if self.ignore_unk:
                # Erase scores for UNK symbol so that they aren't expanded
                unk_indices = input_var.data.eq(self.UNK)
                if unk_indices.nonzero(as_tuple=True)[0].numel() > 0:
                    sequence_scores.data.masked_fill_(
                        unk_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)

        predicts, scores, lengths = self._backtrack(
            stored_predecessors, stored_emitted_symbols, stored_scores, batch_size)

        predicts = predicts[:, :1]
        scores = scores[:, :1]
        lengths = long_tensor_type(lengths)[:, :1]
        mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        predicts[mask] = self.PAD

        return predicts, lengths, scores

    def _backtrack(self, predecessors, symbols, scores, batch_size):
        p = list()
        l = [[self.max_length] * self.beam_size for _ in range(batch_size)]

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(
            batch_size, self.beam_size).topk(self.beam_size, dim=1)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        # the number of EOS found in the backward loop below for each batch
        batch_eos_found = [0] * batch_size

        t = self.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with batch_size*beam_size as the first dimension.
        t_predecessors = (
            sorted_idx + self.pos_index.expand_as(sorted_idx)).view(batch_size * self.beam_size)

        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors)

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.eq(self.EOS).nonzero(as_tuple=True)
            if eos_indices[0].numel() > 0:
                for i in range(len(eos_indices)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with batch_size*beam_size as the first dimension, and batch_size, beam_size for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0].item() // self.beam_size
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.beam_size - (batch_eos_found[b_idx] % self.beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.beam_size)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()]
                        for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (
            re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(batch_size * self.beam_size)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        predicts = torch.stack(p[::-1]).t()
        predicts = predicts[re_sorted_idx].contiguous().view(
            batch_size, self.beam_size, -1).data
        # p = [step.index_select(0, re_sorted_idx).view(batch_size, self.beam_size).data for step in reversed(p)]
        scores = s.data
        lengths = l

        # if self.beam_size == 1:
        #     lengths = [_l[0] for _l in lengths]

        return predicts, scores, lengths

class GreedyGenerator(Generator):
    """
    GreedyGenerator
    """
    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 cue_field=None,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False,
                 multi_turn=False):
        super(GreedyGenerator, self).__init__(model,
                                            src_field,
                                            tgt_field,
                                            cue_field=cue_field,
                                            max_length=max_length,
                                            ignore_unk=ignore_unk,
                                            length_average=length_average,
                                            use_gpu=use_gpu,
                                            multi_turn=multi_turn)

    def decode(self, dec_state):
        """
        decode
        """
        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor

        batch_size = dec_state.get_batch_size()

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * batch_size)
        predicts = long_tensor_type([[0 for _ in range(self.max_length)] for _ in range(batch_size)])
        scores = long_tensor_type([[0 for _ in range(self.max_length)] for _ in range(batch_size)]).float()
        lengths = long_tensor_type([0 for _ in range(batch_size)]).int()

        for t in range(1, self.max_length+1):
            # Run the RNN one step forward
            output, dec_state, _ = self.model.decode(input_var, dec_state)
            log_softmax_output = output.squeeze(1)
            if self.ignore_unk:
                log_softmax_output[:, self.UNK] = -float('inf')
            scores[:, t-1], predicts[:, t-1] = log_softmax_output.max(dim=1)
            input_var = predicts[:, t-1]
            if t < self.max_length:
                lengths[(input_var==self.EOS) * (lengths==0)] = t
            else:
                lengths[lengths==0] = self.max_length

            if self.length_average and t > 1:
                scores[:, t-1] = scores[:, :t-1].mean(dim=1) * \
                    (1 - 1/t) + scores[:, t-1] / t
        predicts = predicts.unsqueeze(1)
        scores = scores.unsqueeze(1)
        lengths = lengths.unsqueeze(1)
        mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        predicts[mask] = self.PAD

        return predicts, lengths, scores


class TopKGenerator(Generator):
    """
    TopKGenerator
    """
    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 cue_field=None,
                 k=1,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False,
                 multi_turn=False):
        super(TopKGenerator, self).__init__(model,
                                            src_field,
                                            tgt_field,
                                            cue_field=cue_field,
                                            max_length=max_length,
                                            ignore_unk=ignore_unk,
                                            length_average=length_average,
                                            use_gpu=use_gpu,
                                            multi_turn=multi_turn)
        self.k = k

    def decode(self, dec_state):
        """
        decode
        """
        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor

        batch_size = dec_state.get_batch_size()

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * batch_size)
        predicts = long_tensor_type([[0 for _ in range(self.max_length)] for _ in range(batch_size)])
        scores = long_tensor_type([[0 for _ in range(self.max_length)] for _ in range(batch_size)]).float()
        lengths = long_tensor_type([0 for _ in range(batch_size)]).int()

        for t in range(1, self.max_length+1):
            # Run the RNN one step forward
            output, dec_state, attn = self.model.decode(input_var, dec_state)
            log_softmax_output = output.squeeze(1)
            if self.ignore_unk:
                log_softmax_output[:, self.UNK] = -float('inf')
            top_logits, top_indexs = log_softmax_output.topk(self.k, dim=1)
            top_probs = torch.softmax(top_logits, dim=1)
            sampled_index = torch.multinomial(top_probs, 1)
            predicts[:, t-1] = torch.gather(top_indexs, dim=1, index=sampled_index).squeeze(1)
            scores[:, t-1] = torch.gather(top_logits, dim=1, index=sampled_index).squeeze(1)
            input_var = predicts[:, t-1]
            if t<self.max_length:
                lengths[(input_var==self.EOS) * (lengths==0)] = t
            else:
                lengths[lengths==0] = self.max_length

            if self.length_average and t > 1:
                scores[:, t-1] = scores[:, :t-1].mean(dim=1) * \
                    (1 - 1/t) + scores[:, t-1] / t
        predicts = predicts.unsqueeze(1)
        scores = scores.unsqueeze(1)
        lengths = lengths.unsqueeze(1)
        mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        predicts[mask] = self.PAD

        return predicts, lengths, scores


class TopPGenerator(Generator):
    """
    TopPGenerator
    """
    def __init__(self,
                 model,
                 src_field,
                 tgt_field,
                 cue_field=None,
                 p=0.9,
                 max_length=10,
                 ignore_unk=True,
                 length_average=True,
                 use_gpu=False,
                 multi_turn=False):
        super(TopPGenerator, self).__init__(model,
                                            src_field,
                                            tgt_field,
                                            cue_field=cue_field,
                                            max_length=max_length,
                                            ignore_unk=ignore_unk,
                                            length_average=length_average,
                                            use_gpu=use_gpu,
                                            multi_turn=multi_turn)
        self.p = p

    def decode(self, dec_state):
        """
        decode
        """
        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor

        batch_size = dec_state.get_batch_size()

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * batch_size)
        predicts = long_tensor_type([[0 for _ in range(self.max_length)] for _ in range(batch_size)])
        scores = long_tensor_type([[0 for _ in range(self.max_length)] for _ in range(batch_size)]).float()
        lengths = long_tensor_type([0 for _ in range(batch_size)]).int()
        
        for t in range(1, self.max_length+1):
            # Run the RNN one step forward
            output, dec_state, attn = self.model.decode(input_var, dec_state)
            log_softmax_output = output.squeeze(1)
            if t<=2:
                log_softmax_output[:, self.EOS] = -float('inf')
            if self.ignore_unk:
                log_softmax_output[:, self.UNK] = -float('inf')
            sorted_logits, sorted_indices = torch.sort(log_softmax_output, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.p
            # # Shift the indices to the right to keep also the first token above the threshold
            # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            sorted_logits -= sorted_indices_to_remove * 1e10

            sorted_probs = torch.softmax(sorted_logits, dim=1)
            sampled_index = torch.multinomial(sorted_probs, 1)

            predicts[:, t-1] = torch.gather(sorted_indices, dim=1, index=sampled_index).squeeze(1)
            scores[:, t-1] = torch.gather(sorted_logits, dim=1, index=sampled_index).squeeze(1)
            input_var = predicts[:, t-1]
            if t < self.max_length:
                lengths[(input_var==self.EOS) * (lengths==0)] = t
            else:
                lengths[lengths==0] = self.max_length

            if self.length_average and t > 1:
                scores[:, t-1] = scores[:, :t-1].mean(dim=1) * \
                    (1 - 1/t) + scores[:, t-1] / t
        predicts = predicts.unsqueeze(1)
        scores = scores.unsqueeze(1)
        lengths = lengths.unsqueeze(1)
        mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        predicts[mask] = self.PAD

        return predicts, lengths, scores
