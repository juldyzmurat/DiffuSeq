from transformers import AutoConfig
# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        ecc_mode=False,
        ecc_num_aux_copies=2,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.ecc_mode = ecc_mode
        self.ecc_num_aux_copies = ecc_num_aux_copies

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

        if self.ecc_mode:
            # Block ids: source block, main block, then one id per fixed aux block.
            self.copy_embeddings = nn.Embedding(2 + self.ecc_num_aux_copies, config.hidden_size)

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def _project_inputs(self, x):
        if self.input_dims != self.hidden_size:
            return self.input_up_proj(x)
        return x

    def _project_outputs(self, hidden_states, dtype):
        if self.output_dims != self.hidden_size:
            hidden_states = self.output_down_proj(hidden_states)
        return hidden_states.type(dtype)

    def _run_encoder(self, emb_inputs, attention_mask=None):
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        if attention_mask is None:
            return self.input_transformers(emb_inputs).last_hidden_state

        extended_attention_mask = attention_mask[:, None, None, :].to(dtype=emb_inputs.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return self.input_transformers(
            emb_inputs,
            attention_mask=extended_attention_mask,
        ).last_hidden_state

    def _forward_ecc(self, x, emb_t, input_mask, aux_states):
        if input_mask is None:
            raise ValueError("ECC mode requires input_mask.")
        if aux_states is None:
            raise ValueError("ECC mode requires fixed aux_states.")
        if len(aux_states) != self.ecc_num_aux_copies:
            raise ValueError(
                f"Expected {self.ecc_num_aux_copies} aux states, got {len(aux_states)}."
            )

        source_mask = input_mask == 0
        target_mask = input_mask == 1
        zero_block = th.zeros_like(x)

        block_inputs = [
            th.where(source_mask.unsqueeze(-1), x, zero_block),
            th.where(target_mask.unsqueeze(-1), x, zero_block),
        ]
        valid_masks = [source_mask, target_mask]

        for aux_state in aux_states:
            if aux_state.shape != x.shape:
                raise ValueError("Each aux state must match the main latent shape.")
            block_inputs.append(th.where(target_mask.unsqueeze(-1), aux_state, zero_block))
            valid_masks.append(target_mask)

        cat_inputs = th.cat(block_inputs, dim=1)
        emb_x = self._project_inputs(cat_inputs)

        seq_length = x.size(1)
        num_blocks = len(block_inputs)
        total_seq_length = seq_length * num_blocks
        position_ids = self.position_ids[:, :seq_length].repeat(1, num_blocks)
        copy_ids = th.cat(
            [
                th.full((1, seq_length), idx, device=x.device, dtype=th.long)
                for idx in range(num_blocks)
            ],
            dim=1,
        )

        emb_inputs = (
            self.position_embeddings(position_ids)
            + self.copy_embeddings(copy_ids)
            + emb_x
            + emb_t.unsqueeze(1).expand(-1, total_seq_length, -1)
        )
        attention_mask = th.cat(valid_masks, dim=1).to(device=x.device, dtype=emb_inputs.dtype)
        hidden_states = self._run_encoder(emb_inputs, attention_mask=attention_mask)

        source_hidden = hidden_states[:, :seq_length, :]
        main_hidden = hidden_states[:, seq_length : 2 * seq_length, :]
        source_output = self._project_outputs(source_hidden, x.dtype)
        main_output = self._project_outputs(main_hidden, x.dtype)
        return th.where(source_mask.unsqueeze(-1), source_output, main_output)

    def forward(self, x, timesteps, input_mask=None, aux_states=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.ecc_mode:
            return self._forward_ecc(x, emb_t, input_mask=input_mask, aux_states=aux_states)

        emb_x = self._project_inputs(x)

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        input_trans_hidden_states = self._run_encoder(emb_inputs)
        return self._project_outputs(input_trans_hidden_states, x.dtype)
