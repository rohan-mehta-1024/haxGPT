import jax, jmp, np
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import optax
import equinox as eqx
import haliax as hax
import haliax.nn as hnn

from functools import partial
from model import GPT2, GPT2Config
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable


@dataclass
class TrainerState:
    step         : int
    model        : eqx.Module
    opt_state    : optax.OptState
    train_key    : PRNGKeyArray
    eval_ket     : PRNGKeyArray
    best_val_loss: int


@dataclass 
class TrainerConfig:
    # for initializing
    seed   : int        = 0
    policy : jmp.Policy = jmp.get_policy("f32")
    loss_fn: Callable   = hnn.cross_entropy
    
    # for training
    data_dir     : str = 'data/shakespeare'
    batch_size   : int = 512 
    train_steps  : int = 600_000
    eval_steps   : int = 200
    eval_interval: int = 1_000

    # for parallelism
    batch_axis  : Optional[str] = 'microbatch' 
    fsdp_axis   : Optional[str] = 'embed' 
    num_devices : int           = jax.device_count() 
    per_device  : int           = -1 # -1 sets per_device batch size to devices // num_devices

    # for logging
    wandb_log   : bool          = False
    log_interval: int           = 1
    resume_from : Optional[str] = None

    def __post__init(self):
        param_mapping   = {self.fsdp_axis : 'data'} if self.fsdp_axis is not None else {}
        compute_mapping = {self.batch_axis : 'data'} if self.batch_axis is not None else {}

        if per_device == -1:
            self.per_device = self.batch_size // jax.device_count()

@dataclass
class OptimizerConfig:
    learning_rate: float = 6e-4 # need to reset all of these to karpathy's values
    weight_decay : float = 0.0
    beta1        : float = 0.9
    beta2        : float = 0.999
    warmup_ratio : float = 0.1 
    min_lr       : float = 6e-5
    grad_clip    : float = 1.0

    def _build_lr_scheduler(self, num_train_steps: int):
        warmup_steps   = int(self.warmup_ratio * num_train_steps)
        lr_decay_steps = num_train_steps - warmup_steps
        alpha          = self.min_lr / self.learning_rate

        schedule = optax.cosine_decay_schedule(
            init_value  = self.learning_rate,
            decay_steps = lr_decay_steps,
            alpha       = alpha
        )

        warmup = optax.linear_schedule(
            init_value       = 0.0, 
            end_value        = self.learning_rate, 
            transition_steps = warmup_steps
        )

        schedule = optax.join_schedules([warmup, schedule], [warmup_steps])
        return schedule

    def configure_optimizer(self, num_train_steps: int):
        self.schedule = self._build_lr_scheduler(num_train_steps)
        optimizer     = optax.adamw(
            learning_rate = self.schedule, 
            weight_decay  = self.weight_decay, 
            b1            = self.beta1, 
            b2            = self.beta2,
            mask          = lambda _, weight: len(weight.axes) >= 2 # don't apply weight decay to bias and layer norm parameters
        )  

        clip      = optax.clip_by_global_norm(self.grad_clip)
        optimizer = optax.chain(optimizer, clip)
        return optimizer



@dataclass
class GPT2Trainer:
    trainer_config : TrainerConfig        = field(default_factory=TrainerConfig)
    opt_config     : OptimizerConfig      = field(default_factory=OptimizerConfig)
    model_config   : Optional[GPT2Config] = field(default_factory=GPT2Config)
    state          : TrainerState         = field(init=False)

    def __post__init(self): # could prob make less ugly....
        self.optmizer = self.opt_config.configure_optimizer(self.trainer_config.num_train_steps)
        self.state    = self._init_state()

        self.train_data = np.memmap(f'{self.trainer_config.data_dir}/train.bin')
        self.val_data   = np.memmap(f'{self.trainer_config.data_dir}/val.bin')

        self.Batch      = hax.Axis(name='batch',      size=self.trainer_config.batch_size)
        self.Microbatch = hax.Axis(name='microbatch', size=self.trainer_config.num_devices * self.trainer_config.per_device)
        self.AccumStep  = hax.Axis(name='accum_step', size=self.Batch.size // self.Microbatch.size)

        self.compute_mapping = self.trainer_config.compute_mapping
        self.param_mapping   = self.trainer_config.param_mapping

        self.step      = property(lambda self: self.state.step)
        self.model     = property(lambda self: self.state.model)
        self.opt_state = property(lambda self: self.state.opt_state)
        self.train_key = property(lambda self: self.state.train_key)
        self.eval_key  = property(lambda self: self.state.eval_key)

    def _init_state(self) -> TrainerState:
        model     = GPT2(self.model_config)
        opt_state = self.optimizer.init(model) 

        if self.param_mapping is not None:
            model     = hax.shard_with_axis_mapping(model, self.param_mapping)
            opt_state = hax.shard_with_axis_mapping(opt_state, self.param_mapping)


        train_key, eval_key = jr.split(self.trainer_config.seed)

        return TrainerState(
            step      = 0,
            model     = model,
            opt_state = self.optimizer.init(model),
            train_key = train_key,
            eval_key  = eval_key
        )

    def _shard_params(self, params):
        if self.fsdp_axis is not None: 
            return hax.shard_with_axis_mapping(params, self.fsdp_axis)
        else: return params



        


    def accumulate_gradients(inputs, ...):



        
         # ==============================
         # Instead of having each device contain the gradients for its microbatch,
         # we have each device contain the gradients for the entire batch with respect
         # to a certain susbet of parameters – this way, the whole gradient doesn't need 
         # to be stored on any on device. In other words, instead of combining the gradients
         # at the end...
        # ===
        grad = hax.shard_with_axis_mapping(grad, param_axis_mapping) # ? which mapping to use here?
        # we don't want each device to hold the gradients for its own microbatch, but for the entire batch
        # with respect to some certain parameters – this way we don't need to store the whole gradient
        # on any one device (instead of smusing/averaging at the end, we do it periodicially so 
        # it doesnt get too big
        #)

        return hax.shard_with_axis_mapping(inputs, compute)

        def loop(carry, inputs):

    def get_batch(self, split: str, key): # -> Tuple[hax.NamedArray, hax.NamedArray, PRNGKeyArray]:
        data = self.train_data if split == "train" else self.val_data

        key, *subkeys  = jr.split(key, 3)
        data_subkeys   = jr.split(subkeys[0], self.Batch.size) # keys to randomly sample from dataset
        run_subkeys    = jr.split(subkeys[1], self.AccumStep.size) # keys for stochastic parts of model (e.g., dropout)

        @partial(jax.vmap, in_axes=(None, 0))
        def get_block(array, index):
            return jax.lax.dynamic_slice(array, (index,), (self.Pos.size,))

        def get_microbatch(batch):
            return batch.unflatten(Batch, (self.AccumStep, self.Microbatch)) 

        random_indices = jr.randint(key, (self.Batch.size,), 0, len(data) - self.Pos.size) 

        x = hax.named(get_block(data, random_indices),     (self.Batch, self.Pos))
        y = hax.named(get_block(data, random_indices + 1), (self.Batch, self.Pos))

        return (get_microbatch(x), get_microbatch(y), run_subkeys), key

    @hax.named_jit(in_axis_resources=self.compute_mapping, 
    axis_resources=self.compute_mapping,
    donate_argnums=(0, 1, 3))

    #@jax.jit(staticargnums = )
    def compute_loss(self, X, y, inference, key):
        preds = self.trainer_state.model(X, inference=inference, key=key)
        loss  = self.trainer_config.loss_fn(preds, y).scalar()
        return (key, loss)


    @cached_property
    def compute_loss(self):

        def loss_fn(self, X, y, inference, key):
            preds = self.trainer_state.model(X, inference=inference, key=key)
            loss  = self.trainer_config.loss_fn(preds, y).scalar()
            return (key, loss)

        jit_kwargs = ... # ok so i think bascially if u put a mapping, it will only try and map tensors with that mapping so we're good???

        if self.computer_mapping is not None: 
            return hax.named_jit(in_axis_resources=self.compute_mapping, axis_resources=self.compute_mapping, donate_argnums=(0, 1, 3))(loss_fn
        else: return jax.jit(loss_fn, static_argnums=(3), donate_argnums=(1, 2, ))



    def accumulate_gradients(self)
        params = eqx.filter(model, eqx.is_inexact_array_like)
        grad = jax.tree_util.tree_map(jnp.zeros_like, params)
        grad_fn = eqx.filter_value_and_grad(self.compute_loss)

        def _(acc, micro_batch)
            prev_loss, prev_grad = acc
            curr_loss, curr_grad = grad_fn(model, X, y, key)
            # curr_grad = hax.shard_with_axis_mapping(curr_grad, parameter_axis_mapping)

            prev_loss += curr_loss
            prev_grad = eqx.apply_updates(prev_grad, curr_grad)
            # prev_grad = hax.shard_with_axis_mapping(prev_grad, parameter_axis_mapping) ... do i have to edo this every time???


    @hax.named_jit()
    def compute_parallel_loss(self, X, y, inference, key): 
        X = hax.shard_with_axis_mapping()


    @jax.jit
    def estimate_loss(self, split: str, key):
        def _step(key, _):
            X, y, key = self.get_batch(key, split)
            loss      = self.compute_loss(X, y, inference, key=key)
            return (key, loss)

        key, losses = lax.scan(_step, key, jnp.zeros(self.trainer_config.eval_iters))
        avg_losses  = jnp.mean(losses)
        return (key, avg_losses)


    def train_step(self, X, y, key): # have to implement gradient accumulator logic???


        def loop(carry, inputs): 


            
        loss, grads        = eqx.filter_value_and_grad(self.compute_loss)(model, X, y, key)
        updates, opt_state = self.optimizer.update(grads, opt_state, params=self.model)
        model              = eqx.apply_updates(model, updates)

        self.state = TrainerState(
            step         = step + 1,
            model        = model,
            opt_state    = opt_state,
            training_key = jr.split(key)[0]
        )

        return (key, loss)


    def train(self):
        while self.step <= self.train_config.train_steps:
            if self.step % self.train_config.eval_interval == 0:
                key, train_loss    = self.estimate_loss('train', self.eval_data_key)
                _, val_loss        = self.estimate_loss('eval' , self.eval_data_key)
                self.eval_data_key = key 

                if self.trainer_config.wandb_log.
                    wandb.log({
                        'step'       : self.step,
                        'train/loss' : train_loss,
                        'eval/loss'  : eval_loss,
                        'lr'         : self.opt_config.build_lr_scheduler(self.trainer_config.train_steps)(self.step)
                    })
                print(f'Step: {self.step} | Average Train Loss: {train_loss:.2f} | Average Val Loss: {eval_loss:.2f}')

                if val_loss < self.best_val_loss or always_save_checkpoint:
                    save
            
            self.train_data_key, loss = self.train_step(self.data_key)

            if self.step % self.train_config.log_interval == 0:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    def _resume(self):
        if self.trainer_config.resume_from is None:
            return self.init_state()
        elif 'gpt2' in self.config.init_from:
            model = GPT2LMHeadModel.from_pretrained(self.config.init_from)
            sd    = model.state_dict()
            return self._init_state()
        else: 
            ckpt       = torch.load(self.config.init_from)
            sd         = ckpt['model'].state_dict()
            self.model_config = ckpt['model_args']

            model = GPT2(self.model_config).from_state_dict(sd, prefix='transformer')

            return TrainerState(
                step      = ckpt['step'],
                model     = model,
                opt_state = ckpt['opt_state'],
                train_key = ckpt['train_key']
                best_val_loss = ckpt['best_val_loss'])
    

# functionality to read in dataset metadata file and guess vocab size??? should go in main func not trainer class right?
    




# for iter in range(max_iters):
#     start_time = time.time()
#     #if iter % eval_interval == 0:
#     #    losses = estimate_loss(key, model)
#     #    print(f"iter {iter} | train loss: {losses['train']:.2f} | val loss: {losses['val']:.2f}")
    
#     x, y, key = get_batch(key, "train")
#     if manual_parallelism:
#         x = hax.shard_with_axis_mapping(x, dp_axis_mapping, mesh)
#         y = hax.shard_with_axis_mapping(y, dp_axis_mapping, mesh)
#     loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
#     updates, opt_state = optim.update(grads, opt_state, params=model)
#     model = eqx.apply_updates(model, updates)
#     key, _ = jr.split(key)
#     end_time = time.time()
#     print(end_time - start_time)
# #context = jnp.array([0])
# #print(decode(model.generate(key, context, max_new_tokens=100).tolist()))



    






# import jax
# import jax.numpy as jnp
# import jax.random as jr 
# import jax.sharding as sharding
# from jax.sharding import Mesh, NamedSharding, PartitionSpec

# import haliax as hax
# import haliax.nn as hnn
# import equinox as eqx
# import equinox.nn as nn
# import optax 
# import numpy as np

# import time
# import os
# from typing import Optional

# # ------ config + download and format data ------

# manual_parallelism=True
# if manual_parallelism:
#   os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=8"
#   mesh = Mesh(np.array(jax.devices()), ("data",))
#   dp_axis_mapping = {"batch": "data"}
# print(jax.devices())

# # hyperparameters
# batch_size = 32 # how many independent sequences will we process in parallel?
# if manual_parallelism:
#     batch_size *= 8
# block_size = 256 # what is the maximum context length for predictions?
# embed_size = 192 # how many dimensions will we use to represent each token?
# key_size = 16 # how many dimensions will we use to represents keys and queries?
# att_heads = 6 # how many attention heads will we use?
# max_iters = 10
# eval_interval = 500 
# learning_rate = 3e-4
# eval_iters = 200
# dropout = 0.2

# # random seed
# key = jr.PRNGKey(0)

# # load the dataset
# with open("input.txt", "r", encoding = "utf-8") as f:
#     text = f.read()

# # extract a vocabulary from the data
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# # create an encoding and decoding scheme
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: "".join([itos[i] for i in l])

# # create train and test splits
# data = jnp.array(encode(text), dtype=jnp.int32)
# n = int(0.9 * len(data))
# train_data = data[:n]
# val_data = data[n:]

# # define axes 
# Block = hax.Axis("block", block_size)
# Batch = hax.Axis("batch", batch_size)
# Vocab = hax.Axis("vocab", vocab_size)
# Embed = hax.Axis("embed", embed_size)
# Key = hax.Axis("key", key_size)
# Head = hax.Axis("head", att_heads)
# Mlp = hax.Axis("mlp", embed_size * 4) 
# Layer = hax.Axis("layer", 6)
# PerToken = Block.alias("per_token") # dimension for each token's attention coefficients
# # ------ config + download and format data ------

# def get_batch(key, split: str):
#     """Generate a single batch of training or testing data"""
#     data = train_data if split == "train" else val_data
#     key, subkey = jr.split(key)
#     subkeys = jr.split(subkey, Batch.size)
#     random_indices = jr.randint(key, (Batch.size,), 0, len(data) - Block.size) # sample `Batch` random indices

#     def get_block(array, index):
#         return jax.lax.dynamic_slice(array, (index,), (Block.size,))
    
#     get_blocks = jax.vmap(get_block, in_axes=(None, 0))
#     x = hax.named(get_blocks(data, random_indices), (Batch, Block)) # extract the corresponding blocks
#     y = hax.named(get_blocks(data, random_indices + 1), (Batch, Block)) # repeat, shifted one over
#     return x, y, key

# class Attention(eqx.Module): 
#     key: hnn.Linear
#     query: hnn.Linear
#     value: hnn.Linear
#     projection: hnn.Linear
#     dropout: hnn.Dropout

#     @staticmethod
#     def init(Key: hax.Axis, Embed: hax.Axis, Head: hax.Axis, key):
#         k1, k2, k3, k4, k5 = jr.split(key, 5)
#         key = hnn.Linear.init(Out=(Head, Key), In=Embed, key=k1, use_bias=False)
#         query = hnn.Linear.init(Out=(Head, Key), In=Embed, key=k2, use_bias=False)
#         value = hnn.Linear.init(Out=(Head, Key), In=Embed, key=k3, use_bias=False)
#         projection = hnn.Linear.init(Out=Embed, In=(Head, Key), key=k4, use_bias=False)
#         dropout = hnn.Dropout(0.2)
#         return Attention(key, query, value, projection, dropout)

#     def _generate_mask(self, scores: hax.NamedArray, mask_type: Optional[str]) -> hax.NamedArray:
#         mask = hax.ones(scores.axes)
#         return hax.tril(mask, PerToken, Block) if mask_type == "causal" else mask

#     def _remove_padding(
#         self, 
#         seq: hax.NamedArray, 
#         padding_mask: hax.NamedArray,
#         mask: hax.NamedArray
#     ) -> hax.NamedArray:
#         """Update the masking strategy to also ignore padding tokens"""
#         update_vector = hax.where(padding_mask, 0, 1).astype(jnp.float32)
#         return mask * update_vector

#     def __call__(
#         self, 
#         seq: hax.NamedArray, 
#         padding_mask: hax.NamedArray,
#         key,
#         inference,
#         mask_type: Optional[str] = "causal"
#     ) -> hax.NamedArray:
#         keys = self.key(seq) # of shape: [..., Block, Key]
#         queries = self.query(seq).rename({"block" : "per_token"}) # of shape: [..., PerToken, Key]
#         values = self.value(seq) # of shape: [..., Block, Key]
#         scores = hax.dot(Key, queries, keys) / jnp.sqrt(Key.size)

#         mask = self._generate_mask(scores, mask_type)
#         unpadded_mask = self._remove_padding(seq, padding_mask, mask)
#         masked_scores = hax.where(unpadded_mask, scores, -jnp.inf) # of shape: [..., PerToken, Block]
#         normalized_scores = hnn.softmax(masked_scores, axis=Block)
#         normalized_scores = self.dropout(normalized_scores, key=key, inference=inference) # neeed to add randomness, key = ?
#         outs = hax.dot(Block, normalized_scores, values)
#         outs = self.projection(outs) # multiplying each vector by its own matrix then summing is equivalent to first concatenating and then multiplying by one big matrix
#         return outs.rename({"per_token" : "block"})

# class FeedForward(eqx.Module):
#     proj_up: hnn.Linear
#     proj_down: hnn.Linear
#     dropout: hnn.Dropout

#     @staticmethod
#     def init(Embed, Intermediate, key):
#         k1, k2, k3 = jax.random.split(key, 3)
#         proj_up = hnn.Linear.init(Out=Intermediate, In=Embed, key=k1)
#         proj_down = hnn.Linear.init(Out=Embed, In=Intermediate, key=k2)
#         dropout = hnn.Dropout(0.2)
#         return FeedForward(proj_up, proj_down, dropout)

#     def __call__(self, x, key, inference):
#         x = self.proj_up(x)
#         x = hnn.relu(x)
#         x = self.proj_down(x)
#         return self.dropout(x, key=key, inference=inference) # neeed to add randomness, key = ?


# class TBlock(eqx.Module):
#     ffn: FeedForward
#     attn: Attention
#     ln1: hnn.LayerNorm
#     ln2: hnn.LayerNorm

#     @staticmethod
#     def init(key):
#         k1, k2 = jr.split(key, 2)
#         ffn = FeedForward.init(Embed, Mlp, key)
#         attn = Attention.init(Key, Embed, Head, key)
#         ln1 = hnn.LayerNorm.init(Embed)
#         ln2 = hnn.LayerNorm.init(Embed)
#         return TBlock(ffn, attn, ln1, ln2)

#     def __call__(self, x: hax.NamedArray, padding_mask, key, inference) -> hax.NamedArray:
#         x = x + self.attn(self.ln1(x), padding_mask, key, inference=inference)
#         x = x + self.ffn(self.ln2(x), key, inference=inference)
#         return x


# class LanguageModel(eqx.Module):
#     token_embedding_table: hnn.Embedding
#     position_embedding_table: hnn.Embedding
#     blocks: hnn.Stacked[TBlock]
#     lm_head: hnn.Linear
#     ln_f: hnn.LayerNorm

#     @staticmethod
#     def init(key):
#         k1, k2, k3, k4 = jr.split(key, 4)
#         token_embedding_table = hnn.Embedding.init(Vocab, Embed, key=k1)
#         position_embedding_table = hnn.Embedding.init(Block, Embed, key=k2)
#         blocks = hnn.Stacked.init(Layer, TBlock)(jr.split(k3, Layer.size))
#         ln_f = hnn.LayerNorm.init(Embed)
#         lm_head = hnn.Linear.init(Out=Vocab, In=Embed, key=k4)
#         return LanguageModel(token_embedding_table, position_embedding_table, blocks, lm_head, ln_f)

#     def __call__(self, seq: hax.NamedArray, key, inference=False) -> hax.NamedArray:
#         padding_mask = seq == hax.full(Block, -1)
#         tok_embs = self.token_embedding_table.embed(seq) 
#         pos_embs = self.position_embedding_table.embed(hax.arange(Block))
#         x = tok_embs + pos_embs
#         x = self.blocks.fold(x, padding_mask, jr.split(key, Layer.size), inference)
#         x = self.ln_f(x)
#         logits = self.lm_head(x)
#         return logits

#     def _pad(self, seq: jnp.ndarray) -> jnp.ndarray:
#         """Left-pad the array with -1s to maximum context length"""
#         padded_array = jnp.pad(seq, (Block.size - len(seq), 0), constant_values=-1)
#         return padded_array

#     def generate(self, key, seq: jnp.ndarray, max_new_tokens: int) -> jnp.ndarray:
#         """Generate `max_new_tokens` new tokens and append them to `seq`"""
#         for _ in range(max_new_tokens):
#             block = hax.named(self._pad(seq[-Block.size:]), (Block,))
#             logits = self(block, key)
#             last_token_logits = logits[Block, -1, Vocab, :] # look at prediction from last token
#             key, subkey = jr.split(key)
#             next_token = hax.random.categorical(key=subkey, logits=last_token_logits, axis=Vocab)
#             next_token = jnp.expand_dims(next_token.array, axis=0) # reshape next_token to [next_token]
#             seq = jnp.concatenate([seq, next_token])
#         return seq
        
# def cross_entropy(logits: hax.NamedArray, labels: hax.NamedArray) -> hax.NamedArray:
#     """Compute the cross_entropy loss between a batch of logits and labels"""
#     preds = hax.nn.log_softmax(logits, axis=Vocab)
#     loss = hax.take(preds, Vocab, labels) # extract log probability of the correct token
#     return -hax.mean(loss)

# @jax.jit 
# def loss_fn(model, x: hax.NamedArray, y: hax.NamedArray, key) -> jnp.ndarray:
#     """Evaluate the model at provided inputs and return the loss"""
#     x = model(x, key=key)
#     return cross_entropy(x, y).scalar()

# def estimate_loss(key, model):
#     """Compute the average loss for a few batches of data"""
#     out = {}
#     for split in ["train", "val"]:
#         losses = jnp.zeros((eval_iters,))
#         for k in range(eval_iters):
#             x, y, key = get_batch(key, split)
#             loss = loss_fn(model, x, y, key)
#             losses = losses.at[k].set(loss)
#         out[split] = jnp.mean(losses)
#     return out


# model = LanguageModel.init(key)
# optim = optax.adamw(learning_rate)
# opt_state = optim.init(model)
# import time


# for iter in range(max_iters):
#     start_time = time.time()
#     #if iter % eval_interval == 0:
#     #    losses = estimate_loss(key, model)
#     #    print(f"iter {iter} | train loss: {losses['train']:.2f} | val loss: {losses['val']:.2f}")
    
#     x, y, key = get_batch(key, "train")
#     if manual_parallelism:
#         x = hax.shard_with_axis_mapping(x, dp_axis_mapping, mesh)
#         y = hax.shard_with_axis_mapping(y, dp_axis_mapping, mesh)
#     loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
#     updates, opt_state = optim.update(grads, opt_state, params=model)
#     model = eqx.apply_updates(model, updates)
#     key, _ = jr.split(key)
#     end_time = time.time()
#     print(end_time - start_time)
# #context = jnp.array([0])
# #print(decode(model.generate(key, context, max_new_tokens=100).tolist()))




    # def _init_from(self):
    #     if self.config.init_from is None:
    #         return self
    #     elif 'gpt2' in self.config.init_from:
    #         model = GPT2LMHeadModel.from_pretrained(self.config.init_from)
    #         sd    = model.state_dict()
    #     else: 
    #         sd = torch.load(self.config.init_from)['model']

    #     loaded = self.from_state_dict(sd, prefix='transformer')
    #     return loaded