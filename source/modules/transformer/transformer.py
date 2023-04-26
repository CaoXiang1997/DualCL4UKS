
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.base_model import BaseModel
from source.modules.transformer import model_utils
from source.modules.transformer import attention_layer
from source.modules.transformer.attention_layer import _float32_softmax
from source.modules.transformer import ffn_layer

class TransformerEncoder(BaseModel):
    """Transformer Encoder model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output ot generate
    probabilities for the output sequence.
    """

    def __init__(self,
                 hparams,
                 input_embedding_layer,
                 output_embedding_layer):
        """Initialize layers to build Transformer encoder model.

        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            name: name of the model.
        """
        super().__init__(hparams, "TransformerEncoder")
        self._input_embedding = input_embedding_layer
        self._output_embedding = output_embedding_layer
        self.encoder_stack = EncoderStack(hparams)


    def forward(self, inputs, inputs_embedding, training: bool = True):
        """Calculate target logits or inferred target sequences.

        Args:
            inputs: int tensor with shape [batch_size, input_length].
            training: boolean, whether in training mode or not.

        Returns:
            encoder outputs: float tensor with shape
            [batch_size, input_length, hidden_size]
        Even when float16 is used, the output tensor(s) are always float32.
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        attention_bias = model_utils.get_padding_bias(inputs)

        # Run the inputs through the encoder layer to map the symbol
        # representations to coninuous representations.
        encoder_outputs = self.encode(inputs, inputs_embedding, attention_bias, training)

        return encoder_outputs


    def encode(self, inputs, inputs_embedding, attention_bias, training):
        """Generate continuous representation for inputs.

        Args:
            inputs: int tensor with shape [batch_size, input_length].
            attention_bias: float tensort with shape [batch_size, 1, 1, input_length].
            training: boolean, whether in training mode or not.

        Returns:
            float tensor with shape [batch_size, input_length, hidden_size]
        """
        # Prepare inputs to the layer stack by adding positional encodings and
        # appling dropout.
        #embedded_inputs = self._input_embedding(inputs)
        embedded_inputs = inputs_embedding
        inputs_padding = model_utils.get_padding(inputs)

        length = embedded_inputs.shape[1]
        pos_encoding = model_utils.get_position_encoding(
            length, self.hidden_size)
        encoder_inputs = embedded_inputs + pos_encoding

        return self.encoder_stack(
            encoder_inputs, attention_bias, inputs_padding, training=training)

class TransformerDecoder(BaseModel):
    """Transformer Decoder model with Keras.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continuous
    representation, and the decoder uses the encoder output ot generate
    probabilities for the output sequence.
    """

    def __init__(self,
                 hidden_size,
                 num_layers,
                 num_heads,
                 filter_size,
                 max_len,
                 use_copy_decoder,
                 input_embedding_layer,
                 output_embedding_layer,
                 use_float16=False):
        """Initialize layers to build Transformer Decoder model.

        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            name: name of the model.
        """
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.use_copy_decoder = use_copy_decoder
        self._input_embedding = input_embedding_layer
        self._output_embedding =  output_embedding_layer
        self.decoder_stack = DecoderStack(num_layers, hidden_size, num_heads, filter_size)
        self.use_float16 = use_float16
        
        if use_copy_decoder:
            self._copy_q_layer = nn.Linear(
                hidden_size, hidden_size, bias=False)
            self._copy_k_layer = nn.Linear(
                hidden_size, hidden_size, bias=False)
            self._copy_v_layer = nn.Linear(
                hidden_size, hidden_size, bias=False)
            self._copy_layer = nn.Sequential(
                nn.Linear(hidden_size, 1, bias=False),
                nn.Sigmoid(),
            )
        
        self.timing_signal = model_utils.get_position_encoding(self.max_len + 1, hidden_size)
        dtype = torch.float16 if use_float16 else torch.float32
        self.self_attention_bias = model_utils.get_decoder_self_attention_bias(self.max_len, dtype=dtype)

    def forward(self, dec_init_state, training: bool = True):
        """Calculate target logits or inferred target sequences.
        Args:
            First item, mixed inputs: int tensor with shape
            [batch_size, knowledge_max_length + context_length]
            Second item, encoder_outputs: float tensor with shape
            [batch_size, sentence_max_length, hidden_size].
            Third item (optional), targets: int tensor with shape
            [batch_size, target_length].
            training: boolean, whether in training mode or not.

        Returns:
            If targets is defined, then return logits for each word in the target
            sequence. float tensor with shape [batch_size, target_length, vocab_size]
            If target is none, then generate output sequence one token at a time.
            returns a dictionary {
                outputs: [batch_size, decoded length]
                scores: [batch_size, float]}
        Even when float16 is used, the output tensor(s) are always float32.
        """
        mixed_inputs = dec_init_state.knowledge_context_sentences
        encoder_outputs = dec_init_state.knowledge_context_encoded
        decoder_inputs = dec_init_state.response_embedding
        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        attention_bias = model_utils.get_padding_bias(mixed_inputs)
        # Shift targets to the right, and remove the last element
        decoder_inputs = decoder_inputs[:, :-1]
        length = decoder_inputs.shape[1]
        pos_encoding = model_utils.get_position_encoding(length, self.hidden_size).to(decoder_inputs.device)
        decoder_inputs += pos_encoding

        # Run values
        dtype = torch.float16 if self.use_float16 else torch.float32
        self_attention_bias = model_utils.get_decoder_self_attention_bias(length, dtype=dtype)
        decoder_outputs = self.decoder_stack(
            decoder_inputs,
            encoder_outputs,
            self_attention_bias,
            attention_bias,
            training=training)

        if self.use_copy_decoder:
            logits = self.copy_decode(mixed_inputs, encoder_outputs,
                                        decoder_outputs, attention_bias, training)
            logits = torch.log(logits+1e-10)
        else:
            logits = self._output_embedding(decoder_outputs, mode="linear")
            logits = torch.log_softmax(logits+1e-10, dim=-1)
        sample_ids = torch.argmax(logits, dim=2)

        return logits, sample_ids

    def copy_decode(self, mixed_inputs, encoder_outputs, decoder_outputs, attention_bias, training):
        """ Generate softmax values of logits in the target sequence.

        Args: Same as decode function's arguments
            - mixed_inputs: input values of context and chosen knowledge. Int tensor with shape
            [batch_size, mixed_input_length]
            - encoder_outputs: continuous representation of input sequence. float tensor
            with shape [batch_size, sentence_max_length, hidden_size]
            - decoder_outputs: continuous representaiton of output sequence. float tensor
            with shape [batch_size, target_length - 1, hidden_size]
            - attention_bias: float tensor with shape [batch_size, 1, 1, sentence_max_length]
            training: boolean, whether in training mode or not.
        Returns:
            float32 tensor with shape [batch_size, target_length, vocab_size]
        """

        w_q = self._copy_q_layer
        w_k = self._copy_k_layer
        w_v = self._copy_v_layer

        q = w_q(decoder_outputs)
        k = w_k(encoder_outputs)
        v = w_v(encoder_outputs)

        # Codes for multi heads attention, but not necessary.

        q = self.decoder_stack.layers[-1][1].layer.split_heads(q)
        k = self.decoder_stack.layers[-1][1].layer.split_heads(k)
        v = self.decoder_stack.layers[-1][1].layer.split_heads(v)

        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        a_t = torch.matmul(q, k.transpose(-2,-1))
        a_t += attention_bias
        # [batch_size, num_heads, target_length - 1, mixed_input_length]
        p_att = _float32_softmax(a_t)

        # [batch_size, num_heads, target_length - 1, depth]
        hidden = torch.matmul(p_att, v)
        # [batch_size, target_length, hidden_size]
        p_att = p_att[:,0]
        hidden = self.decoder_stack.layers[-1][1].layer.combine_heads(hidden)
        hidden = self.decoder_stack.layers[-1][1].layer.output_dense_layer(hidden)
        # feed forward network
        hidden = self.decoder_stack.layers[-1][2](hidden, training=training)
        hidden = self.decoder_stack.output_normalization(hidden)
        # [batch_size, target_length - 1, vocab_size]
        p_vocab = _float32_softmax(self._output_embedding(decoder_outputs, mode="linear"))

        # matching (p_att.shape) to (p_vocab.shape)
        # initial_indices = mixed_inputs.unsqueeze(1).repeat(1, p_vocab.shape[1], 1)
        # i1, i2 = torch.meshgrid(torch.range(0, batch_size-1),
        #             torch.range(0, p_vocab.shape[1]-1))
        # i1 = i1.unsqueeze(-1).repeat(1, 1, p_att.shape[2]).to(initial_indices.device)
        # i2 = i2.unsqueeze(-1).repeat(1, 1, p_att.shape[2]).to(initial_indices.device)
        # # [batch_size, target_length - 1, mixed_input_length, 3]
        # indices = torch.stack([i1, i2, initial_indices], dim=-1).to(torch.int64)
        # [batch_size, target_length - 1, vocab_size]
        indices = mixed_inputs.unsqueeze(1).repeat(1, p_vocab.shape[1], 1)
        p_att = torch.scatter(torch.zeros_like(p_vocab), 2, indices, p_att)

        p_gen = self._copy_layer(hidden)
        # [batch_size, target_length - 1, vocab_size]
        p_gen = torch.tile(p_gen, [1, 1, self.vocab_size])
        # [batch_size, target_length - 1, vocab_size]
        p_word = (1 - p_gen) * p_vocab + p_gen * p_att

        return p_word

    def decode(self, input_ids, state, training=True):
        """Return predicted sequence."""
        # Currently, we always do prediction in float32
        # TODO(reedwm): Add float16 support.
        mixed_inputs = state.knowledge_context_sentences
        encoder_outputs = state.knowledge_context_encoded

        if not state.has('cache'):
            batch_size = mixed_inputs.shape[0]
            cache = {
                "layer_%d" % layer: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size], device=input_ids.device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size], device=input_ids.device)
                } for layer in range(self.num_layers)
            }
            cache["encoder_decoder_attention_bias"] = model_utils.get_padding_bias(mixed_inputs)
            state.add(cache=cache)
            state.add(step=0)
        
        cache = state.cache
        step = state.step

        # Add mixed input, encoder output and attention bias to the cache.
        cache["mixed_inputs"] = mixed_inputs
        cache["encoder_outputs"] = encoder_outputs
        
        logits, cache = self.symbols_to_logits_fn(input_ids.unsqueeze(1), step, cache, training=training)
        state.add(cache=cache)
        state.add(step=step+1)
        
        return logits, state

    def symbols_to_logits_fn(self, decoder_input, i, cache, training=True):
        """Generate logits for next potential IDs.

        Args:
            decoder_input: Current decoder input. int tensor with shape [batch_size *
            beam_size]
            i: Loop index
            cache: dictionary of values storing the encoder output, encoder-decoder
            attention bias, and previous decoder attention values.

        Returns:
            Tuple of
            (logits with shape [batch_size * beam_size, vocab_size],
                updated cache values)
        """
        attention_bias = cache['encoder_decoder_attention_bias']
        mixed_inputs = cache['mixed_inputs']
        encoder_outputs = cache['encoder_outputs']
        
        decoder_input = self._input_embedding(decoder_input)
        decoder_input += self.timing_signal[i:i + 1].to(decoder_input.device)
        self_attention_bias = self.self_attention_bias[:, :, i:i + 1, :i + 1]
        decoder_outputs = self.decoder_stack(
            decoder_input,
            encoder_outputs,
            self_attention_bias,
            attention_bias,
            training=training,
            cache=cache)

        if self.use_copy_decoder:
            logits = self.copy_decode(mixed_inputs, encoder_outputs,
                                        decoder_outputs, attention_bias, training)
            logits = torch.log(logits+1e-10)
        else:
            logits = self._output_embedding(decoder_outputs, mode="linear")
            logits = torch.log_softmax(logits+1e-10, dim=-1)

        return logits, cache


class PrePostProcessingWrapper(nn.Module):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, hidden_size):
        super(PrePostProcessingWrapper, self).__init__()

        self.layer = layer
        self.postprocess_dropout = 0

        # Create normalization layer
        self.layer_norm = nn.LayerNorm([hidden_size])

    def forward(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization

        y = self.layer_norm(x)
        # Get layer output
        y = self.layer(y, *args, **kwargs)

        return x + y

class EncoderStack(nn.Module):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, hidden_size, num_layers, num_heads, filter_size):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(num_layers):
            # Create sublayers for each laer.
            self_attention_layer = attention_layer.SelfAttention(
                hidden_size, num_heads,
                attention_dropout=0)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                hidden_size, filter_size,
                relu_dropout=0)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, hidden_size),
                PrePostProcessingWrapper(feed_forward_network, hidden_size)
            ])

        # Create final layer normalization laeyr.
        self.output_normalization = nn.LayerNorm([hidden_size])


    def forward(self, encoder_inputs, attention_bias, inputs_padding, training):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
            1, input_length]
          inputs_padding: tensor with shape [batch_size, input_length], inputs with
            zero paddings.
          training: boolean, whether in training mode or not.

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            encoder_inputs = self_attention_layer(
                encoder_inputs, attention_bias, training=training)
            encoder_inputs = feed_forward_network(
                encoder_inputs, training=training)

        return self.output_normalization(encoder_inputs)

class DecoderStack(nn.Module):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
        1. Self-attention layer
        2. Multi-headed attention layer combining encoder outputs with results from
           the previous self-attention layer.
        3. Feedforward network (2 fully-conneced layers)
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_heads,
                 filter_size):
        super(DecoderStack, self).__init__()
        self.layers = []
        for i in range(num_layers):
            self_attention_layer = attention_layer.SelfAttention(
                hidden_size, num_heads,
                attention_dropout=0.0)
            enc_dec_attention_layer = attention_layer.Attention(
                hidden_size, num_heads,
                attention_dropout=0.0)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                hidden_size, filter_size,
                relu_dropout=0.0)
            self_attention_layer = PrePostProcessingWrapper(self_attention_layer, hidden_size)
            enc_dec_attention_layer = PrePostProcessingWrapper(enc_dec_attention_layer, hidden_size)
            feed_forward_network = PrePostProcessingWrapper(feed_forward_network, hidden_size)
            setattr(self, f'self_attention_layer_{i}', self_attention_layer)
            setattr(self, f'enc_dec_attention_layer_{i}', enc_dec_attention_layer)
            setattr(self, f'feed_forward_network_{i}', feed_forward_network)
            self.layers.append([
                self_attention_layer,
                enc_dec_attention_layer,
                feed_forward_network
            ])
        self.output_normalization = nn.LayerNorm([hidden_size])


    def forward(self, decoder_inputs, encoder_outputs, self_attention_bias,
             attention_bias, training, cache=None):
        """Return the output of the decoder layer stacks.

        Args:
            decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
            encoder_outputs: tensor with shape [batch_size, sentence_max_length, hidden_size]
            self_attention_bias: bias for decoder self-attention layer. [1, 1,
                target_len, target_length]
            attention_bias: bias for encoder-decoder attention layer. [batch_size, 1,
                1, sentence_max_length]
            training: boolean, whether in training mode or not.
            cache: (Used for fast decoding) A nested dictionary storing previous
                decoder self-attention values. The items are:
                    {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                                "v": tensor with shape [batch_size, i, value_channels]},
                                ...}
        Returns:
            output of decoder layer stack.
            float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            decoder_inputs = self_attention_layer(
                decoder_inputs,
                self_attention_bias,
                training=training,
                cache=layer_cache)
            decoder_inputs = enc_dec_attention_layer(
                decoder_inputs,
                encoder_outputs,
                attention_bias,
                training=training)
            decoder_inputs = feed_forward_network(
                decoder_inputs, training=training)

        return self.output_normalization(decoder_inputs)
