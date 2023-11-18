import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_flatten

import torch
import equinox as eqx
import haliax as hax
import haliax.nn as hnn

from levanter.compat.torch_serialization import (
    StateDictSerializationMixin as Serializable,
    apply_prefix,
    flatten_linear_layer,
    stack_state_dict,
    unflatten_linear_layer,
    unstack_state_dict,
)

from typing import Optional
from dataclasses import dataclass
from transformers import GPT2LMHeadModel


@dataclass
class GPT2Config:
    seq_len    : int   = 1024
    vocab_size : int   = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_layers : int   = 12
    num_heads  : int   = 12
    hidden_dims: int   = 768
    dropout    : float = 0.0
    use_bias   : bool  = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    def __post_init__(self):
        self.Pos      = hax.Axis(name='position',  size=self.seq_len)
        self.Vocab    = hax.Axis(name='vocab',     size=self.vocab_size)
        self.Embed    = hax.Axis(name='embed',     size=self.hidden_dims)
        self.Heads    = hax.Axis(name='heads',     size=self.num_heads)
        self.Layers   = hax.Axis(name='layers',    size=self.num_layers)
        self.Mlp      = hax.Axis(name='mlp',       size=self.hidden_dims * 4)
        self.HeadSize = hax.Axis(name='head_size', size=self.hidden_dims // self.num_heads)
        self.KeyPos   = self.Pos.alias('key_position')

    @classmethod
    def get_pretrained_config(cls, model_type: str, dropout: float = 0.0) -> GPT2Config:
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f'Loading pre-trained {model_type} weights...')

        config_args = {
            'gpt2'        :  dict(num_layers=12, num_heads=12, hidden_dims=768),  # 124M params
            'gpt2-medium' :  dict(num_layers=24, num_heads=16, hidden_dims=1024), # 350M params
            'gpt2-large'  :  dict(num_layers=36, num_heads=20, hidden_dims=1280), # 774M params
            'gpt2-xl'     :  dict(num_layers=48, num_heads=25, hidden_dims=1600), # 1558M params
        }[model_type] | {'vocab_size' : 50257, 'dropout' : dropout}

        return GPT2Config(**config_args, init_from=model_type) # type: ignore


class CausalSelfAttention(eqx.Module, Serializable): 
    attn     : hnn.Linear
    proj     : hnn.Linear
    dropout  : hnn.Dropout
    mask     : hax.NamedArray = eqx.static_field()
    _scale_f : float          = eqx.static_field()

    @staticmethod
    def init(config: GPT2Config, key) -> CausalSelfAttention:
        k1, k2   = jr.split(key, 2)
        Qkv      = hax.Axis('qkv', size=3) # generate queries, keys, and values all at once

        attn = hnn.Linear.init(
            In       = config.Embed, 
            Out      = (Qkv, config.Heads, config.HeadSize),
            key      = k1, 
            use_bias = config.use_bias
        )

        proj = hnn.Linear.init(
            In       = (config.Heads, config.HeadSize), 
            Out      = config.Embed,
            key      = k2,
            use_bias = config.use_bias
        )

        dropout     = hnn.Dropout(config.dropout)
        causal_mask = hnn.attention.causal_mask(config.Pos, config.KeyPos)

        return CausalSelfAttention(
            attn, 
            proj, 
            dropout, 
            causal_mask, 
            jnp.sqrt(config.HeadSize.size)
        )

    def _remove_padding(self, padding_mask: hax.NamedArray) -> hax.NamedArray:
        update_vector = hax.where(padding_mask, 0, 1).astype(jnp.float32)
        return self.mask * update_vector
  
    def __call__(
        self, 
        x: hax.NamedArray, 
        padding_mask: hax.NamedArray = None,
        inference: bool = False, 
        *, 
        key
    ) -> hax.NamedArray:
        qkv = self.attn(x).rearrange((..., 'qkv', 'heads', 'pos', 'head_size'))
        q, k, v, = qkv.unbind('qkv')

        # rename `Pos` axis for keys and values
        k = k.rename({'pos' : 'key_pos'})
        v = v.rename({'pos' : 'key_pos'})

        scores  = hax.dot('head_size', q, k) # of shape: [..., Pos, KeyPos]
        scores /= self._scale_f

        if padding_mask is not None:
            mask = self._remove_padding(padding_mask)
        else: mask = self.mask

        masked_scores      = hax.where(mask, scores, -jnp.inf)
        normalized_scores  = hnn.softmax(masked_scores, axis='key_position')
        regularized_scores = self.dropout(normalized_scores, inference, key=key)

        # ==============================
        # multiplying the output each head by its own matrix and then summing
        # is equivalent to first concatenating the output of each head and then
        # multiplying by one big matrix, since each region of the concatenated
        # vector only interacts with a subset of the overall matrix
        # ==============================
        x = hax.dot('key_position', regularized_scores, v)
        x = self.projection(x) # of shape: [..., Pos, Embed]
        return x

    def _state_dict_key_map(self):
        return {'attn' : 'c_attn', 'proj' : 'c_proj'}

    def from_state_dict(self, state_dict, prefix: str):
        unflattened_attn = unflatten_linear_layer(
            apply_prefix(prefix, 'c_attn'), 
            state_dict, 
            self.attn, 
            None
        )
        
        unflattened_proj = unflatten_linear_layer(
            apply_prefix(prefix, 'c_proj'), 
            state_dict, 
            self.proj, 
            None
        )

        unflattened_params = {**unflattened_attn, **unflattened_proj}
        new_params         = super().from_state_dict(unflattened_params, prefix) # extract PyTree from unflattened params
        return new_params

    def to_state_dict(self, state_dict, prefix: str):
        super().update_state_dict(state_dict, prefix) # write all model params into state_dict

        flattened_attn = flatten_linear_layer( 
            apply_prefix(prefix, 'c_attn'), 
            self.c_attn, 
            None
        )

        flattened_proj = flatten_linear_layer(
            apply_prefix(prefix, 'c_proj'), 
            self.c_proj, 
            None
        )

        state_dict.update({**flattened_attn, **flattened_proj}) # then update with flattened versions 
        return state_dict


class MLP(eqx.Module, Serializable):
    proj_up  : hnn.Linear
    proj_down: hnn.Linear
    dropout  : hnn.Dropout

    @staticmethod
    def init(config: GPT2Config, key) -> MLP:
        k1, k2 = jr.split(key, 2)

        proj_up = hnn.Linear.init(
            In       = config.Embed, 
            Out      = config.Mlp,
            key      = k1,
            use_bias = config.use_bias
        )

        proj_down = hnn.Linear.init(
            In       = config.Mlp,
            Out      = config.Embed,
            key      = k2,
            use_bias = config.use_bias
        )

        dropout = hnn.Dropout(config.dropout)
        return MLP(proj_up, proj_down, dropout)

    def __call__(
        self, 
        x: hax.NamedArray, 
        inference: bool = False, 
        *, 
        key
    ) -> hax.NamedArray:
        x = self.proj_up(x)
        x = hnn.gelu(x)
        x = self.proj_down(x)
        x = self.dropout(x, inference, key=key)
        return x

    def _state_dict_key_map(self):
        return {'proj_up' : 'c_fc', 'proj_down' : 'c_proj'}


class Block(eqx.Module):
    ln_1: hnn.LayerNorm
    ln_2: hnn.LayerNorm
    attn: CausalSelfAttention
    mlp : MLP

    @staticmethod
    def init(config: GPT2Config, key) -> Block:
        k1, k2 = jr.split(key, 2)

        ln_1 = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        ln_2 = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        attn = CausalSelfAttention.init(config, key=k1)
        mlp  = MLP.init(config, key=k2)

        return Block(ln_1, ln_2, attn, mlp)

    def __call__(
        self, 
        x: hax.NamedArray, 
        padding_mask: hax.NamedArray = None,
        inference: bool = False,
        *,
        key
    ) -> hax.NamedArray:
        k1, k2 = jr.split(key, 2)

        x      = self.ln_1(x)
        attn_x = self.attn(x, padding_mask, inference, key=k1)
        x      = x + attn_x 

        x      = self.ln_2(x)
        ff_x   = self.mlp(x, inference, key=k2)
        x      = x + ff_x
        
        return x


class GPT2(eqx.Module, Serializable):
    tok_embedding_table: hax.NamedArray
    pos_embedding_table: hax.NamedArray
    dropout            : hnn.Dropout
    ln_f               : hnn.LayerNorm
    blocks             : hnn.Stacked
    config             : GPT2Config = eqx.static_field()

    @staticmethod
    def init(config: GPT2Config, key) -> GPT2:
        k1, k2, k3, k4 = jr.split(key, 4)

        tok_embedding_table = hnn.Embedding.init(config.Vocab, config.Embed, key=k1)
        pos_embedding_table = hnn.Embedding.init(config.Pos,   config.Embed, key=k2)

        dropout = hnn.Dropout(config.dropout)
        ln_f    = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        blocks  = hnn.Stacked.init(config.Layers, Block)(
            config = config,
            key    = jr.split(k3, config.Layers.size)
        )

        # apply special scaled init to the residual projections, per GPT-2 paper
        scale_f    = 1 / jnp.sqrt(2 * config.Layers.size)
        get_params = lambda b: [b.stacked.attn.proj.weight, b.stacked.mlp.proj_down.weight]
        scaled     = [scale_f * i for i in get_params(blocks)]
        blocks     = eqx.tree_at(get_params, blocks, scaled)

        return GPT2(
            tok_embedding_table, 
            pos_embedding_table, 
            dropout,  
            ln_f,
            blocks,
            config
        )

    def __call__(
        self, 
        seq: hax.NamedArray, 
        inference: bool = False, 
        *, 
        key
    ) -> hax.NamedArray:
        padding_mask = seq == hax.full(self.config.Pos, -1)

        tok_embs = self.token_embedding_table.embed(seq) 
        pos_embs = self.position_embedding_table.embed(hax.arange(self.config.Pos))
        x        = self.dropout(tok_embs + pos_embs, inference, key=key)

        x = self.blocks.fold(
            x, 
            padding_mask, 
            inference, 
            key=jr.split(key, self.config.Layers.size)
        )

        x      = self.ln_f(x)
        logits = self.tok_embedding_table.unembed(x)
        return logits

    def count_params(self, non_embedding: bool = True):
        params = [
            self.tok_embedding_table, 
            self.pos_embedding_table if non_embedding else None, 
            self.dropout, 
            self.ln_f, 
            self.blocks
        ]

        leaves, _   = tree_flatten(params)
        param_count = sum([i.size for i in leaves])
        return param_count

    def _pad(self, seq: jnp.ndarray) -> jnp.ndarray:
        """Left-pad the array with -1s to maximum context length"""
        padded_array = jnp.pad(seq, (Block.size - len(seq), 0), constant_values=-1)
        return padded_array
        
    def generate(
        self,
        seq: jnp.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None, 
        *,
        key
    ) -> jnp.ndarray:
        for _ in range(max_new_tokens):
            Pos, Vocab = self.config.Pos, self.config.Vocab

            seq          = hax.named(self._pad(seq[-Pos.size:]), (Pos,))
            logits       = self(seq, inference=True, key=key)
            final_logits = logits[Pos, -1, Vocab, :] / temperature # look at prediction from last token

            if top_k is not None:
                final_logits = hax.top_k(final_logits, axis=Vocab, k=top_k)

            key, subkey  = jr.split(key)
            next_token   = hax.random.categorical(logits=final_logits, axis=Vocab, key=subkey)
            next_token   = jnp.expand_dims(next_token.array, axis=0) # reshape next_token to [next_token]
            seq          = jnp.concatenate([seq, next_token])

        return seq

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = self.count_params()
        L, H, Q, T = (
            self.config.num_layers, 
            self.config.num_heads,
            self.config.hidden_dims // self.num_heads,
            self.seq_len
        )

        flops_per_token  = 6*N + 12*L*H*Q*T # flops needed to compute one token
        flops_per_fwdbwd = flops_per_token * T # flops needed to compute one sequence
        flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter # flops needed to compute one iteration

        flops_achieved   = flops_per_iter * (1.0/dt) # per second
        flops_promised   = 312e12 # A100 GPU with bfloat16 has peak of 312 TFLOPS
        mfu              = flops_achieved / flops_promised

        return mfu

    def _state_dict_key_map(self):
        return {
            "blocks"              : "h",
            "tok_embedding_table" : "wte", 
            "pos_embedding_table" : "wpe"
        }

    def from_state_dict(self, state_dict, prefix: str):
        stacked_params   = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "h"))
        new_params       = super().from_state_dict(stacked_params, prefix=prefix) # extract PyTree from stacked params
        return new_params

    def update_state_dict(self, state_dict, prefix: str):
        super().update_state_dict(state_dict, prefix) # write all model params into state_dict

        unstacked_params = unstack_state_dict(state_dict, prefix=apply_prefix(prefix, "h"))
        state_dict.update(unstacked_params) # then then update with unstacked versions  
        return state_dict