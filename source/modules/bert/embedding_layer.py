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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class EmbeddingSharedWeights(nn.Module):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, shared_weights):
    """Specify characteristic parameters of embedding layer.

    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
    """
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.shared_weights = shared_weights
    self.embedder = nn.Embedding(vocab_size, hidden_size, _weight=shared_weights)

  def forward(self, inputs, mode="embedding"):
    """Get token embeddings of inputs.

    Args:
      inputs: An int64 tensor with shape [batch_size, length]
      mode: string, a valid value is one of "embedding" and "linear".
    Returns:
      outputs: (1) If mode == "embedding", output embedding tensor, float32 with
        shape [batch_size, length, embedding_size]; (2) mode == "linear", output
        linear tensor, float32 with shape [batch_size, length, vocab_size].
    Raises:
      ValueError: if mode is not valid.
    """
    if mode == "embedding":
      return self._embedding(inputs)
    elif mode == "linear":
      return self._linear(inputs)
    else:
      raise ValueError("mode {} is not valid.".format(mode))

  def _embedding(self, inputs):
    """Applies embedding based on inputs tensor."""
    # Create binary mask of size [batch_size, length]
    mask = inputs.ne(0).to(torch.float32)
    embeddings = self.embedder(inputs)
    embeddings *= mask.unsqueeze(-1)
    # Scale embedding by the sqrt of the hidden size
    embeddings *= self.hidden_size ** 0.5

    return embeddings

  def _linear(self, inputs):
    """Computes logits by running inputs through a linear layer.

    Args:
      inputs: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    batch_size = inputs.shape[0]
    length = inputs.shape[1]

    x = torch.reshape(inputs, [-1, self.hidden_size])
    logits = torch.matmul(x, self.shared_weights.transpose(-2,-1))

    return torch.reshape(logits, [batch_size, length, self.vocab_size])
