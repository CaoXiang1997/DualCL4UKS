# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer model helper methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch


def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = torch.range(0, length-1, dtype=torch.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (float(num_timescales) - 1))
  inv_timescales = min_timescale * torch.exp(
      torch.range(0, num_timescales-1, dtype=torch.float32) * -log_timescale_increment)
  scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
  return signal


def get_decoder_self_attention_bias(length, dtype=torch.float32):
  """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.
    dtype: The dtype of the return value.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
  neg_inf = -65500.0 if dtype == torch.float16 else -1e9
  valid_locs = torch.tril(torch.ones([length, length], dtype=torch.float32))
  valid_locs = torch.reshape(valid_locs, [1, 1, length, length])
  decoder_bias = neg_inf * (1.0 - valid_locs)
  return decoder_bias


def get_padding(x, padding_value=0, dtype=torch.float32):
  """Return float tensor representing the padding values in x.

  Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: The dtype of the return value.

  Returns:
    float tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
  return x.eq(padding_value).to(dtype)


def get_padding_bias(x):
  """Calculate bias tensor from padding values in tensor.

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.

  Args:
    x: int tensor with shape [batch_size, length]

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
  padding = get_padding(x)
  attention_bias = padding * -1e9
  attention_bias = torch.unsqueeze(torch.unsqueeze(attention_bias, dim=1), dim=1)
  return attention_bias
