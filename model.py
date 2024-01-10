import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_flatten

import torch
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from haliax import NamedArray

from levanter.compat.torch_serialization import (
    StateDictSerializationMixin as Serializable,
    StateDict,
    apply_prefix,
    flatten_linear_layers,
    stack_state_dict,
    unflatten_linear_layers,
    unstack_state_dict,
)

from typing import Optional
from dataclasses import dataclass
from jaxtyping import Array


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
    def get_pretrained_config(cls, model_type: str, dropout: float = 0.0) -> 'GPT2Config':
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f'Loading pre-trained {model_type} weights...')

        config_args = {
            'gpt2'        :  dict(num_layers=12, num_heads=12, hidden_dims=768),  # 124M params
            'gpt2-medium' :  dict(num_layers=24, num_heads=16, hidden_dims=1024), # 350M params
            'gpt2-large'  :  dict(num_layers=36, num_heads=20, hidden_dims=1280), # 774M params
            'gpt2-xl'     :  dict(num_layers=48, num_heads=25, hidden_dims=1600), # 1558M params
        }[model_type] | {'vocab_size' : 50257, 'dropout' : dropout}

        return GPT2Config(**config_args)


class CausalSelfAttention(eqx.Module, Serializable): 
    attn     : hnn.Linear
    proj     : hnn.Linear
    dropout  : hnn.Dropout
    mask     : NamedArray = eqx.static_field() # not learnable
    _scale_f : float      = eqx.static_field() # not learnable

    @staticmethod
    def init(config: GPT2Config, key) -> 'CausalSelfAttention':
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

    def __call__(
        self, 
        x: NamedArray, 
        padding_mask: NamedArray = None,
        *, 
        key
    ) -> NamedArray:
        qkv = self.attn(x).rearrange((..., 'qkv', 'heads', 'position', 'head_size'))
        q, k, v, = qkv.unbind('qkv')  # each of q, k, and v is of shape: [Heads, Pos, HeadSize] 

        # rename `Pos` axis for keys and values (because both axes of q • k cannot have the same name)
        k = k.rename({'position' : 'key_position'})
        v = v.rename({'position' : 'key_position'})

        scores  = hax.dot('head_size', q, k) # of shape: [..., Heads, Pos, KeyPos]
        scores  /= self._scale_f

        if padding_mask is not None:
            mask = self.mask * padding_mask
        else: mask = self.mask

        masked_scores      = hax.where(mask, scores, -jnp.inf) # cannot attend to tokens in the future
        normalized_scores  = hnn.softmax(masked_scores, axis='key_position')
        regularized_scores = self.dropout(normalized_scores, key=key)

        x = hax.dot('key_position', regularized_scores, v) # of shape: [..., Heads, Pos, HeadSize]
        # ==============================
        # multiplying the output of each head by its own matrix and then summing
        # is equivalent to first concatenating the output of each head and then
        # multiplying by one big matrix, since each region of the concatenated
        # vector only interacts with a subset of the overall matrix
        # ==============================
        x = self.proj(x) # of shape: [..., Pos, Embed]
        return x

    def _state_dict_key_map(self):
        return {'attn' : 'c_attn', 'proj' : 'c_proj'}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # and our c_proj is [heads, head_dim] -> [embed] and hf's is the flattened [heads * head_dim] -> [embed]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim
        d = {}
        d.update(unflatten_linear_layers(apply_prefix(prefix, "c_attn"), state_dict, self.attn, None))
        d.update(unflatten_linear_layers(apply_prefix(prefix, "c_proj"), state_dict, self.proj, None))

        return super().from_state_dict(d, prefix)

    # def from_state_dict(self, state_dict, prefix: str):
    #     unflattened_attn = unflatten_linear_layers(
    #         prefix,#apply_prefix(prefix, 'c_attn'), 
    #         state_dict, 
    #         self.attn, 
    #         None
    #     )
        
    #     unflattened_proj = unflatten_linear_layers(
    #         prefix,
    #         #apply_prefix(prefix, 'c_proj'), 
    #         state_dict, 
    #         self.proj, 
    #         None
    #     )

    #     unflattened_params = {}#{**unflattened_attn, **unflattened_proj}
    #     unflattened_params.update(unflattened_attn)
    #     unflattened_params.update(unflattened_proj)
    #     return super().from_state_dict(unflattened_params, prefix) # extract PyTree from unflattened params

    # def update_state_dict(self, state_dict, prefix: str):
    #     new_dict={}
    #     super().update_state_dict(new_dict, prefix) # write all model params into state_dict

    #     flattened_attn = flatten_linear_layers(apply_prefix(prefix, "c_attn"), self.attn, None)
    #     flattened_proj = flatten_linear_layers(apply_prefix(prefix, "c_proj"), self.proj, None)
    #     new_dict.update(flattened_attn)
    #     new_dict.update(flattened_proj)
    #     #new_dict.update({**flattened_attn, **flattened_proj}) # then update with flattened versions 
    #     state_dict.update(new_dict)
    #     return state_dict# | new_dict

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "c_attn"), self.attn, None))
        my_dict.update(flatten_linear_layers(apply_prefix(prefix, "c_proj"), self.proj, None))

        state_dict.update(my_dict)
        return state_dict



class MLP(eqx.Module, Serializable):
    proj_up  : hnn.Linear
    proj_down: hnn.Linear
    dropout  : hnn.Dropout

    @staticmethod
    def init(config: GPT2Config, key) -> 'MLP':
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

    def __call__(self, 
        x: NamedArray,
        *, 
        key
    ) -> NamedArray:
        x = self.proj_up(x)
        x = hnn.gelu(x)
        x = self.proj_down(x)
        x = self.dropout(x, key=key)
        return x

    def _state_dict_key_map(self):
        return {'proj_up' : 'c_fc', 'proj_down' : 'c_proj'}


class Block(eqx.Module):
    ln_1: hnn.LayerNorm
    ln_2: hnn.LayerNorm
    attn: CausalSelfAttention
    mlp : MLP

    @staticmethod
    def init(config: GPT2Config, key) -> 'Block':
        k1, k2 = jr.split(key, 2)

        ln_1 = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        ln_2 = hnn.LayerNorm.init(config.Embed, use_bias=config.use_bias)
        attn = CausalSelfAttention.init(config, key=k1)
        mlp  = MLP.init(config, key=k2)

        return Block(ln_1, ln_2, attn, mlp)

    def __call__(
        self, 
        x: NamedArray, 
        padding_mask: NamedArray = None,
        *,
        key
    ) -> NamedArray:
        k1, k2 = jr.split(key, 2)

        x      = self.ln_1(x)
        attn_x = self.attn(x, padding_mask, key=k1)
        x      = x + attn_x # residual connection

        x      = self.ln_2(x)
        ff_x   = self.mlp(x, key=k2)
        x      = x + ff_x # residual connection
        
        return x


class GPT2(eqx.Module, Serializable):
    tok_embedding_table: hax.NamedArray
    pos_embedding_table: hax.NamedArray
    dropout            : hnn.Dropout
    ln_f               : hnn.LayerNorm
    blocks             : hnn.Stacked
    config             : GPT2Config = eqx.static_field()
    inference          : bool       #= eqx.static_field()

    @staticmethod
    def init(config: GPT2Config, key) -> 'GPT2':
        k1, k2, k3, k4 = jr.split(key, 4)

        tok_embedding_table = hnn.Embedding.init(config.Vocab, config.Embed, key=k1)
        pos_embedding_table = hnn.Embedding.init(config.Pos, config.Embed, key=k2)

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
            inference=False,
            config=config
        )

    def __call__(
        self, 
        seq: hax.NamedArray, 
        *, 
        key
    ) -> hax.NamedArray:

        # ==============================
        # input sequence will always be padded to a length of Pos, e.g., during 
        # inference, so 
        if self.inference:
            seq_padding_mask  = (seq == hax.full(self.config.Pos, -1)).astype(jnp.int32)
            attn_padding_mask = seq_padding_mask.broadcast_axis(self.config.KeyPos) 
        else: attn_padding_mask = None

        tok_embs = self.tok_embedding_table.embed(seq) 
        pos_embs = self.pos_embedding_table.embed(hax.arange(self.config.Pos))
        x        = self.dropout(tok_embs + pos_embs, key=key)

        x = self.blocks.fold(
            x, 
            None, #attn_padding_mask,
            key=jr.split(key, self.config.Layers.size)
        )

        x      = self.ln_f(x)
        logits = self.tok_embedding_table.unembed(x)
        return logits

    def count_params(self, non_embedding: bool = True):
        params = eqx.filter([ # to exclude boolean inference parameter to dropout and layernorm
            self.tok_embedding_table, 
            self.pos_embedding_table if non_embedding else None, 
            self.dropout, 
            self.ln_f, 
            self.blocks
        ], eqx.is_inexact_array_like) 

        leaves, _   = tree_flatten(params)
        param_count = sum([i.size for i in leaves])
        return param_count

    def _pad(self, seq: jnp.ndarray) -> jnp.ndarray:
        """Left-pad the array with -1s to maximum context length"""
        padded_array = jnp.pad(seq, (self.config.Pos.size - len(seq), 0), constant_values=-1)
        return padded_array
        

    def generate(
        self,
        seq: Array = jnp.array([0]),
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None, 
        *,
        key
    ) -> Array: 
        # self.inference_mode() how to set this????
        print(self.inference)
        Pos, Vocab = self.config.Pos, self.config.Vocab
        NewTokens  = hax.Axis('new_tokens', size=max_new_tokens)
        keys      = jr.split(key, max_new_tokens)

        def _gen(seq, key):
            print(seq, seq.shape)
            logits       = self(hax.named(seq, (Pos,)), key=key)
            final_logits = logits[Pos, -1, Vocab, :] / temperature # look at prediction from last token

            if top_k is not None:
                final_logits = hax.top_k(final_logits, axis=Vocab, k=top_k)
            
            _, subkey  = jr.split(key)
            next_token   = hax.random.categorical(logits=final_logits, axis=Vocab, key=subkey)
            next_token   = jnp.expand_dims(next_token.array, axis=0) # reshape next_token to [next_token]
            seq          = jnp.concatenate([seq, next_token])

            return seq[-Pos.size:], next_token 

        return hax.scan(_gen, NewTokens)(self._pad(seq)[-Pos.size:], keys)

    # def generate( #rewrite as a fold and jit for max efficiency?
    #     self,
    #     seq: jnp.ndarray,
    #     max_new_tokens: int,
    #     temperature: float = 1.0,
    #     top_k: Optional[int] = None, 
    #     *,
    #     key
    # ) -> jnp.ndarray:
    #     for _ in range(max_new_tokens):
    #         Pos, Vocab = self.config.Pos, self.config.Vocab

    #         seq          = hax.named(self._pad(seq[-Pos.size:]), (Pos,))
    #         logits       = self(seq, key=key)
    #         final_logits = logits[Pos, -1, Vocab, :] / temperature # look at prediction from last token

 

    #         key, subkey  = jr.split(key)
    #         next_token   = hax.random.categorical(logits=final_logits, axis=Vocab, key=subkey)
    #         next_token   = jnp.expand_dims(next_token.array, axis=0) # reshape next_token to [next_token]
    #         seq          = jnp.concatenate([seq, next_token])

    #     return seq

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Compute model's flop utilization rate """
        N = self.count_params()
        L, H, Q, T = (
            self.config.num_layers, 
            self.config.num_heads,
            self.config.hidden_dims // self.config.num_heads,
            self.config.seq_len
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

    # def update_state_dict(self, state_dict, prefix: str):
    #     new_dict = {}
    #     super().update_state_dict(new_dict, prefix) # write all model params into state_dict

    #     unstacked_params = unstack_state_dict(new_dict, prefix=apply_prefix(prefix, "h"))
    #     #new_dict.update(unstacked_params) # then then update with unstacked versions  
    #     return state_dict | unstacked_params

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # this method needs to "devectorize" the blocks, so that we have a list of blocks h.0.FOO, h.1.FOO, etc.
        # first just do the normal thing with our own dict, which we'll post-process
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix)

        stacked_dict = unstack_state_dict(my_state_dict, apply_prefix(prefix, "h"))
        state_dict.update(stacked_dict)

        return state_dict