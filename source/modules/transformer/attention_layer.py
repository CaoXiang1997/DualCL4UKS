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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _float32_softmax(logits):
  """Computes a softmax activation in float32.

  When training a model using float16, softmax is still done in float32 for
  numeric stability.

  Args:
    logits: A tensor, with any shape accepted by `torch.softmax`.

  Returns:
    A tensor with the same dtype as `logits`.
  """
  input_dtype = logits.dtype
  logits = logits.to(torch.float32)
  output = torch.softmax(logits, dim=-1)
  return output.to(input_dtype)


class Attention(nn.Module):
  """Multi-headed attention layer."""

  def __init__(self, hidden_size, num_heads, attention_dropout):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout

    # Layers for linearly projecting the queries, keys, and values.
    self.q_dense_layer = nn.Linear(
        self.hidden_size, self.hidden_size, bias=False)
    self.k_dense_layer = nn.Linear(
        self.hidden_size, self.hidden_size, bias=False)
    self.v_dense_layer = nn.Linear(
        self.hidden_size, self.hidden_size, bias=False)
    self.output_dense_layer = nn.Linear(
        self.hidden_size, self.hidden_size, bias=False)

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    batch_size = x.shape[0]
    length = x.shape[1]

    # Calculate depth of last dimension after it has been split.
    depth = (self.hidden_size // self.num_heads)

    # Split the last dimension
    x = torch.reshape(x, [batch_size, length, self.num_heads, depth])

    # Transpose the result
    return torch.permute(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    batch_size = x.shape[0]
    length = x.shape[2]
    x = torch.permute(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
    return torch.reshape(x, [batch_size, length, self.hidden_size])

  def forward(self, x, y, bias, training, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    q = self.q_dense_layer(x)
    k = self.k_dense_layer(y)
    v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = torch.cat([cache["k"].to(k.dtype), k], dim=1)
      v = torch.cat([cache["v"].to(v.dtype), v], dim=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5

    # Calculate dot product attention
    logits = torch.matmul(q, k.transpose(-2,-1))
    logits += bias.to(logits.device)
    weights = _float32_softmax(logits)
    attention_output = torch.matmul(weights, v)

    # Recombine heads --> [batch_size, length, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def forward(self, x, bias, training, cache=None):
    return super(SelfAttention, self).forward(x, x, bias, training, cache)
