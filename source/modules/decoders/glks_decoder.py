#!/usr/bin/env python
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/rnn_decoder.py
"""

import torch
import torch.nn as nn

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class GLKSDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0):
        super(GLKSDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout

        self.rnn_input_size = self.input_size
        self.out_input_size = self.input_size + 2*self.hidden_size

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size

        if self.attn_mode is not None:
            self.src_attn = Attention(query_size=self.input_size+self.hidden_size*2,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode='mlp',
                                       project=False)
            self.cue_attn = Attention(query_size=self.input_size+self.hidden_size*2,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode='mlp',
                                       project=False)
            self.out_input_size += 2*self.memory_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         knowledge,
                         feature=None,
                         src_attn_memory=None,
                         src_attn_mask=None,
                         src_memory_lengths=None,
                         cue_attn_memory=None,
                         cue_attn_mask=None,
                         cue_memory_lengths=None):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None
        
        if src_memory_lengths is not None and src_attn_mask is None:
            max_len = src_attn_memory.size(1)
            src_attn_mask = sequence_mask(src_memory_lengths, max_len).eq(0)

        if cue_memory_lengths is not None and cue_attn_mask is None:
            max_len = cue_attn_memory.size(1)
            cue_attn_mask = sequence_mask(cue_memory_lengths, max_len).eq(0)

        init_state = DecoderState(
            hidden=hidden,
            knowledge=knowledge,
            feature=feature,
            src_attn_memory=src_attn_memory,
            src_attn_mask=src_attn_mask,
            cue_attn_memory=cue_attn_memory,
            cue_attn_mask=cue_attn_mask
        )
        return init_state

    def decode(self, input, state, training=False):
        """
        decode
        """
        hidden = state.hidden
        rnn_input_list = []
        out_input_list = []
        output = Pack()
        knowledge = state.knowledge

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        rnn_input_list.append(input)
        out_input_list.append(input)

        rnn_input = torch.cat(rnn_input_list, dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)
        out_input_list.append(rnn_output)

        out_input_list.append(knowledge)

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            rnn_input_list.append(feature)

        if self.attn_mode is not None:
            query = torch.cat([input, hidden[-1].unsqueeze(1), knowledge], 2)

            src_attn_memory = state.src_attn_memory
            src_attn_mask = state.src_attn_mask
            src_context, src_attn = self.src_attn(query=query,
                                                  memory=src_attn_memory,
                                                  mask=src_attn_mask)
            out_input_list.append(src_context)
            output.add(src_attn=src_attn)
            
            cue_attn_memory = state.cue_attn_memory
            cue_attn_mask = state.cue_attn_mask
            cue_context, cue_attn = self.cue_attn(query=query,
                                                  memory=cue_attn_memory,
                                                  mask=cue_attn_mask)
            out_input_list.append(cue_context)
            output.add(cue_attn=cue_attn)

        out_input = torch.cat(out_input_list, dim=-1)
        state.hidden = new_hidden

        if training:
            return out_input, state, output
        else:
            log_prob = self.output_layer(out_input)
            return log_prob, state, output

    def forward(self, inputs, state, teacher_forcing=True):
        """
        forward
        """
        inputs, lengths = inputs
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)

        hidden_memory = inputs.new_zeros(
            size=(batch_size, max_len, self.hidden_size),
            dtype=torch.float)
        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)
        hidden_memory = hidden_memory.index_select(0, indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            if teacher_forcing:
                dec_input = inputs[:num_valid, i]
            else:
                if i==0:
                    dec_input = inputs.new_ones(size=(num_valid,))*2
                else:
                    dec_input = out_inputs[:num_valid, i-1].argmax(-1)
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, _ = self.decode(
                dec_input, valid_state, training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            hidden_memory[:num_valid, i] = valid_state.hidden[-1]
            out_inputs[:num_valid, i] = out_input.squeeze(1)

        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        hidden_memory = hidden_memory.index_select(0, inv_indices)
        state.add(hidden_memory=hidden_memory)

        log_probs = self.output_layer(out_inputs)
        return log_probs, state
