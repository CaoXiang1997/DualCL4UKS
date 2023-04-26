import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.models.base_model import BaseModel
from source.modules.transformer import embedding_layer
from source.modules.transformer.transformer import TransformerDecoder
from source.modules.from_parlai import universal_sentence_embedding
from source.modules.bert import modeling
from source.modules.bert import embedding_layer as bert_embedding_layer
from source.utils.misc import sequence_mask


def load_pretrained_bert_model(bert_dir, max_length):
    bert_model = modeling.get_bert_model(
        torch.keras.layers.Input(
            shape=(max_length,), dtype=torch.int32, name='input_wod_ids'),
        torch.keras.layers.Input(
            shape=(max_length,), dtype=torch.int32, name='input_mask'),
        torch.keras.layers.Input(
            shape=(max_length,), dtype=torch.int32, name='input_type_ids'),
        config=modeling.BertConfig.from_json_file(os.path.join(bert_dir, 'bert_config.json')),
        float_type=torch.float32)

    # load pretrained model
    init_checkpoint = os.path.join(bert_dir, 'bert_model.ckpt')
    checkpoint = torch.train.Checkpoint(model=bert_model)
    checkpoint.restore(init_checkpoint)

    return bert_model


class SequentialKnowledgeTransformer(BaseModel):
    def __init__(self,
                 bert_dir,
                 use_posterior,
                 vocab_size,
                 hidden_size,
                 num_layers,
                 num_heads,
                 filter_size,
                 max_len,
                 use_gumbel=False,
                 gumbel_temperature=1.0,
                 use_copy_decoder=False,
                 use_float16=False):
        super(SequentialKnowledgeTransformer, self).__init__()
        self.use_posterior = use_posterior
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.use_gumbel = use_gumbel
        self.gumbel_temperature = gumbel_temperature

        self.encoder = BertModel.from_pretrained(bert_dir)
        self._embedding = bert_embedding_layer.EmbeddingSharedWeights(
            vocab_size, hidden_size, self.encoder.get_parameter('embeddings.word_embeddings.weight'))
        self._output_embedding = embedding_layer.EmbeddingSharedWeights(
            vocab_size, hidden_size)
        self.decoder = TransformerDecoder(
            hidden_size, num_layers, num_heads, filter_size, max_len, use_copy_decoder, self._embedding, self._output_embedding, use_float16=use_float16)

        self.dialog_rnn = nn.GRU(2*hidden_size, hidden_size, batch_first=True)
        self.history_rnn = nn.GRUCell(hidden_size, hidden_size)
        self.prior_query_layer = nn.utils.weight_norm(nn.Linear(3*hidden_size, hidden_size, bias=False))
        self.posterior_query_layer = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size, bias=False))
        
        self.nll_loss = nn.NLLLoss(ignore_index=0, reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='mean')


    def encode(self, inputs, training: bool = True, use_posterior=True):
        outputs = Pack()
        context = inputs.src[0]
        response = inputs.tgt[0]
        knowledge_sentences = inputs.cue[0]
        
        episode_length = inputs.src[1].ne(0).sum(-1)
        context_length = inputs.src[1]
        response_length = inputs.tgt[1]
        knowledge_length = inputs.cue[1]
        num_knowledges = inputs.cue[1].ne(0).sum(-1)

        batch_size, max_episode_length, max_context_length = context.shape
        _, _, max_response_length = response.shape
        _, max_num_knowledges, max_knowledge_length = knowledge_sentences.shape
        episode_batch_size = batch_size * max_episode_length

        # Collapse episode_length dimension to batch
        context = torch.reshape(context, [-1, max_context_length])
        response = torch.reshape(response, [-1, max_response_length])
        knowledge_sentences = torch.reshape(knowledge_sentences, [-1, max_num_knowledges, max_knowledge_length])
        context_length = torch.reshape(context_length, [-1])
        response_length = torch.reshape(response_length, [-1])
        knowledge_length = torch.reshape(knowledge_length, [-1, max_num_knowledges])
        num_knowledges = torch.reshape(num_knowledges, [-1])

        #################
        # Encoding
        #################
        # Dialog encode (for posterior)
        response_embedding = self._embedding(response)
        knowledge_sentences_embedding = self._embedding(knowledge_sentences)

        _, context_outputs = self.encode_sequence(context, context_length, training)
        context_output = universal_sentence_embedding(context_outputs, context_length)

        _, response_outputs = self.encode_sequence(response, response_length, training)
        response_output = universal_sentence_embedding(response_outputs, response_length)

        # Dialog encode (for posterior)
        context_response_output = torch.cat([context_output, response_output], dim=1)
        context_response_output = torch.reshape(context_response_output, [batch_size, max_episode_length, 2 * self.hidden_size])
        dialog_outputs, _ = self.dialog_rnn(context_response_output)
        dialog_outputs = torch.reshape(dialog_outputs, [episode_batch_size, self.hidden_size])

        # Dialog encode (for prior)
        start_pad = torch.zeros([batch_size, 1, self.hidden_size], dtype=torch.float32, device=dialog_outputs.device)
        shifted_dialog_outputs = torch.reshape(dialog_outputs, [batch_size, max_episode_length, self.hidden_size])
        shifted_dialog_outputs = torch.cat([start_pad, shifted_dialog_outputs[:, :-1]], dim=1)
        shifted_dialog_outputs = torch.reshape(shifted_dialog_outputs, [episode_batch_size, self.hidden_size])
        prior_dialog_outputs = torch.cat([context_output, shifted_dialog_outputs], dim=1)

        # Knowledge encode
        pooled_knowledge_embeddings, knowledge_embeddings = self.encode_knowledges(
            knowledge_sentences, num_knowledges, knowledge_length, training)
        knowledge_mask = sequence_mask(num_knowledges, dtype=torch.bool)

        # Knowledge selection (prior & posterior)
        _, prior, posterior = self.sequential_knowledge_selection(
            pooled_knowledge_embeddings, knowledge_mask,
            prior_dialog_outputs, dialog_outputs, episode_length,
            training=training
        )

        prior_attentions, prior_argmaxes = prior
        posterior_attentions, posterior_argmaxes = posterior
        outputs.add(prior_attentions=prior_attentions, posterior_attentions=posterior_attentions)

        #################
        # Decoding
        #################
        if self.use_posterior and use_posterior:
            ks_attentions = posterior_attentions
            argmaxes = posterior_argmaxes
        else:
            ks_attentions = prior_attentions
            argmaxes = prior_argmaxes
        chosen_sentences = []
        knowledge_sentences = knowledge_sentences.unsqueeze(1).repeat(1,max_episode_length,1,1).reshape(episode_batch_size, max_num_knowledges, max_knowledge_length)
        for i,num in enumerate(argmaxes):
            chosen_sentences.append(knowledge_sentences[i, num])
        chosen_sentences = torch.stack(chosen_sentences, dim=0)
        knowledge_embeddings = knowledge_embeddings.unsqueeze(1).repeat(1,max_episode_length,1,1,1).reshape(episode_batch_size, max_num_knowledges, max_knowledge_length, -1)
        chosen_embeddings = torch.sum(ks_attentions * knowledge_embeddings.permute(2,3,0,1), dim=-1).permute(2,0,1)

        knowledge_context_encoded = torch.cat([chosen_embeddings, context_outputs], dim=1)  # [batch, length, embed_dim]
        knowledge_context_sentences = torch.cat([chosen_sentences, context], dim=1)  # For masking [batch, lenth]
        dec_init_state = DecoderState(
            knowledge_context_sentences=knowledge_context_sentences,
            knowledge_context_encoded=knowledge_context_encoded,
            response=response,
            response_embedding=response_embedding,
        )
        #################
        # Loss
        #################
        return outputs, dec_init_state

    def encode_sequence(self, sequence, sequence_length, training):
        # suppose that there is only 1 type embedding
        # shape of sequence: [batch_size, sequence_length, hidden_size]
        attention_mask = sequence_mask(sequence_length, dtype=torch.int32)
        sequence_type_ids = torch.zeros_like(sequence, dtype=torch.int32)
        output = self.encoder(input_ids=sequence,
                              attention_mask=attention_mask,
                              token_type_ids=sequence_type_ids)
        sequence_outputs = output[0]
        pooled_output = output[1]
        return pooled_output, sequence_outputs

    def encode_knowledges(self, knowledge_sentences, num_knowledges, sentences_length, training):
        max_num_knowledges = torch.max(num_knowledges)
        max_sentences_length = torch.max(sentences_length).to(torch.int32)
        episode_batch_size = knowledge_sentences.shape[0]

        squeezed_knowledge = torch.reshape(
            knowledge_sentences, [episode_batch_size * max_num_knowledges, max_sentences_length])
        squeezed_knowledge_length = torch.reshape(sentences_length, [-1])
        _, encoded_knowledge = self.encode_sequence(squeezed_knowledge, squeezed_knowledge_length, training)

        # Reduce along sequence length
        flattened_sentences_length = torch.reshape(sentences_length, [-1])
        sentences_mask = sequence_mask(flattened_sentences_length, dtype=torch.float32)
        encoded_knowledge = encoded_knowledge * torch.unsqueeze(sentences_mask, dim=-1)

        reduced_knowledge = universal_sentence_embedding(encoded_knowledge, flattened_sentences_length)
        embed_dim = encoded_knowledge.shape[-1]
        reduced_knowledge = torch.reshape(reduced_knowledge, [episode_batch_size, max_num_knowledges, embed_dim])
        encoded_knowledge = torch.reshape(encoded_knowledge, [episode_batch_size, max_num_knowledges, max_sentences_length, embed_dim])

        return reduced_knowledge, encoded_knowledge

    def compute_knowledge_attention(self, knowledge, query, knowledge_mask):
        knowledge_innerp = torch.squeeze(torch.matmul(knowledge, torch.unsqueeze(query, dim=-1)), dim=-1)
        knowledge_innerp -= torch.logical_not(knowledge_mask).to(torch.float32) * 1e20  # prevent softmax from attending masked location
        knowledge_attention = torch.softmax(knowledge_innerp, dim=1)
        if 0 < self.gumbel_temperature < 1 and self.use_gumbel:
            knowledge_attention = F.gumbel_softmax(
                torch.log(knowledge_attention+1e-10), self.gumbel_temperature, hard=True)

        knowledge_argmax = torch.argmax(knowledge_attention, dim=1)
        knowledge_argmax = knowledge_argmax.to(torch.int32)

        return knowledge_attention, knowledge_argmax

    def sequential_knowledge_selection(self, knowledge, knowledge_mask,
                                       prior_context, posterior_context,
                                       episode_length, training=True):
        batch_size = episode_length.shape[0]
        max_episode_length = torch.max(episode_length)
        max_num_knowledges = knowledge.shape[1]
        embed_dim = knowledge.shape[2]
        prior_embed_dim = prior_context.shape[1]
        prior_context = torch.reshape(prior_context, [batch_size, max_episode_length, prior_embed_dim])
        posterior_context = torch.reshape(posterior_context, [batch_size, max_episode_length, embed_dim])

        
        knowledge_states = []
        prior_attentions = []
        prior_argmaxes = []
        posterior_attentions = []
        posterior_argmaxes = []

        knowledge_state = torch.zeros([batch_size, embed_dim], dtype=torch.float32, device=prior_context.device)

        for current_episode in range(max_episode_length):
            current_knowledge_candidates = knowledge
            current_knowledge_mask = knowledge_mask

            current_prior_context = prior_context[:, current_episode]
            current_posterior_context = posterior_context[:, current_episode]

            # Make query
            current_prior_query = self.prior_query_layer(torch.cat([current_prior_context, knowledge_state], dim=1))
            current_posterior_query = self.posterior_query_layer(torch.cat([current_posterior_context, knowledge_state], dim=1))

            # Compute attention
            prior_knowledge_attention, prior_knowledge_argmax = self.compute_knowledge_attention(
                current_knowledge_candidates, current_prior_query, current_knowledge_mask)
            posterior_knowledge_attention, posterior_knowledge_argmax = self.compute_knowledge_attention(
                current_knowledge_candidates, current_posterior_query, current_knowledge_mask)

            # Sample knowledge from posterior
            chosen_knowledges = []
            for i,num in enumerate(posterior_knowledge_argmax):
                chosen_knowledges.append(current_knowledge_candidates[i, num])
            chosen_knowledges = torch.stack(chosen_knowledges, dim=0)
            # Roll-out one step
            knowledge_state = self.history_rnn(
                chosen_knowledges,
                knowledge_state,
            )

            # Update TensorArray
            knowledge_states.append(knowledge_state)
            prior_attentions.append(prior_knowledge_attention)
            prior_argmaxes.append(prior_knowledge_argmax)
            posterior_attentions.append(posterior_knowledge_attention)
            posterior_argmaxes.append(posterior_knowledge_argmax)

            current_episode += 1

        knowledge_states = torch.stack(knowledge_states, dim=1).reshape([-1, embed_dim])
        prior_attentions = torch.stack(prior_attentions, dim=1).reshape([-1, max_num_knowledges])
        prior_argmaxes = torch.stack(prior_argmaxes, dim=1).reshape([-1])
        posterior_attentions = torch.stack(posterior_attentions, dim=1).reshape([-1, max_num_knowledges])
        posterior_argmaxes = torch.stack(posterior_argmaxes, dim=1).reshape([-1])

        return knowledge_states, (prior_attentions, prior_argmaxes), \
            (posterior_attentions, posterior_argmaxes)
    
    def decode(self, input, state):
        logits, state = self.decoder.decode(input, state, training=False)
        return logits, state, None

    def forward(self, inputs, training=False, epoch=-1, use_posterior=True):
        """
        forward
        """
        outputs, dec_init_state = self.encode(inputs, training=training, use_posterior=use_posterior)
        logits, _ = self.decoder(dec_init_state, training=training)
        outputs.add(logits=logits)
        metrics = self.collect_metrics(outputs, inputs.tgt[0][:,:,1:], epoch=epoch)
        return outputs, metrics


    def collect_metrics(self, outputs, target, epoch=-1):
        """
        collect_metrics
        """
        target = target.reshape([outputs.logits.shape[0], -1])
        new_target = []
        logits = []
        prior_attentions = []
        posterior_attentions = []
        for i in range(len(target)):
            if target[i].sum().item()!=0:
                new_target.append(target[i])
                logits.append(outputs.logits[i])
                prior_attentions.append(outputs.prior_attentions[i])
                posterior_attentions.append(outputs.posterior_attentions[i])
        target = torch.stack(new_target, dim=0)
        logits = torch.stack(logits, dim=0)
        prior_attentions = torch.stack(prior_attentions, dim=0)
        posterior_attentions = torch.stack(posterior_attentions, dim=0)

        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        if isinstance(self.nll_loss, torch.nn.NLLLoss):
            nll = self.nll_loss(logits.transpose(1, -1), target)
        else:
            nll = self.nll_loss(logits, target)
        loss += nll
        acc = accuracy(logits, target, padding_idx=0)
        metrics.add(nll=nll, acc=acc)

        kl_loss = self.kl_loss(torch.log(prior_attentions + 1e-10), posterior_attentions)
        metrics.add(kl=kl_loss)
        
        loss += kl_loss
        metrics.add(loss=loss)
        
        return metrics
