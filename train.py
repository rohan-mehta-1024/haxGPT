import jax
import jmp
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import chex
import optax
import equinox as eqx
import haliax as hax
import haliax.nn as hnn
from model import GPT2, GPT2Config

import os
import time
import torch
import wandb
import draccus

from haliax import NamedArray
from jaxtyping import Array, PRNGKeyArray
from functools import partial, cached_property
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Tuple, NamedTuple
from optax import OptState, Schedule, GradientTransformation

@chex.dataclass
@dataclass
class TrainerState:
    model         : GPT2
    opt_state     : optax.OptState
    master_key    : PRNGKeyArray
    step          : int 
    best_val_loss : float 

@chex.dataclass
@dataclass
class Batch:
    input_tokens  : NamedArray
    target_tokens : NamedArray
    model_key     : PRNGKeyArray


@dataclass 
class TrainerConfig:
    # for initializing
    seed    : int        = 0
    policy  : jmp.Policy = jmp.get_policy("p=f32")
    
    # for training / eval
    data_dir         : str = 'data/shakespeare'
    unit_batch_size  : int = 512
    train_steps      : int = 600_000
    grad_accum_steps : int = 1 
    eval_steps       : int = 200
    eval_interval    : int = 1_000

    # for parallelism
    batch_axis  : str = 'batch'
    fsdp_axis   : str = 'embed'
    num_devices : int = jax.device_count() 

    # for logging
    wandb_log        : bool          = False
    log_interval     : int           = 1
    resume_from      : Optional[str] = None
    always_save_ckpt : bool          = True
    output_dir       : str           = ''

    def __post_init__(self):
        self.batch_size      = self.unit_batch_size * self.num_devices
        self.compute_mapping = {self.batch_axis : 'data'} if self.batch_axis else None
        self.param_mapping   = {self.fsdp_axis  : 'data'} if self.fsdp_axis  else None


@dataclass
class OptimizerConfig:
    learning_rate : float = 6e-4
    weight_decay  : float = 1e-1
    beta1         : float = 0.9
    beta2         : float = 0.95
    warmup_ratio  : float = 0.1
    min_lr        : float = 6e-5
    grad_clip     : float = 1.0

    def _build_lr_scheduler(self, num_train_steps: int) -> optax.Schedule:
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

    def configure_optimizer(self, num_train_steps: int) -> optax.GradientTransformation:
        self.schedule = self._build_lr_scheduler(num_train_steps)
        optimizer     = optax.adamw(
            learning_rate = self.schedule, 
            weight_decay  = self.weight_decay, 
            b1            = self.beta1, 
            b2            = self.beta2,
            #mask          = partial(jax.tree_map, lambda weight: len(weight.shape) >= 2) # don't apply weight decay to bias and layer norm parameters  
        )  

        clip      = optax.clip_by_global_norm(self.grad_clip)
        optimizer = optax.chain(optimizer, clip)
        return optimizer


@dataclass
class TrainingRunConfig:
    trainer   : TrainerConfig           = field(default_factory=TrainerConfig)
    optimizer : OptimizerConfig         = field(default_factory=OptimizerConfig)
    model     : Union[GPT2Config, str]  = field(default_factory=GPT2Config)

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = GPT2Config.from_pretrained(self.model)


@dataclass
class GPT2Trainer:
    training_run   : TrainingRunConfig = field(default_factory=TrainingRunConfig)
    state          : TrainerState      = field(init=False)

    def __post_init__(self): 
        # destructure individual configs
        self.trainer_config = self.training_run.trainer
        self.opt_config     = self.training_run.optimizer
        self.model_config   = self.training_run.model

        # alias miscallaneous config values
        self.compute_mapping = self.trainer_config.compute_mapping
        self.param_mapping   = self.trainer_config.param_mapping
        self.policy          = self.trainer_config.policy
        self.num_devices     = self.trainer_config.num_devices
        self.parallelism     = self.num_devices > 1 

        # initialize model, optimizer, and state
        self.state = self._init_state()
        
        # load in data
        self.train_data = np.memmap(f'{self.trainer_config.data_dir}/train.bin', dtype=np.uint16)
        self.val_data   = np.memmap(f'{self.trainer_config.data_dir}/val.bin'  , dtype=np.uint16)

        # alias / define important axes
        self.Pos        = self.model.config.Pos
        self.Vocab      = self.model.config.Vocab
        self.Batch      = hax.Axis(name='batch',       size=self.trainer_config.batch_size)
        self.AccumSteps = hax.Axis(name='accum_steps', size=self.trainer_config.grad_accum_steps)
        self.EvalSteps  = hax.Axis(name='eval_steps',  size=self.trainer_config.eval_steps)

        if self.trainer_config.wandb_log:
            wandb.init(
                project = 'haxGPT',
                config  = asdict(self.training_run),
                resume  = 'allow'
            )

    def __getattr__(self, name):
        """Delegate attribute access to the state object."""
        if 'state' in self.__dict__ and hasattr(self.state, name):
            return getattr(self.state, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Delegate attribute assignment to the state object."""
        if 'state' in self.__dict__ and hasattr(self.state, name):
            setattr(self.state, name, value)
        else:
            super().__setattr__(name, value)

    def _init_state(self) -> TrainerState:
        base_key       = jr.PRNGKey(self.trainer_config.seed)
        key, model_key = jr.split(base_key)

        model  = GPT2.init(self.model_config, model_key) 
        model  = self.policy.cast_to_param(model)
        params = eqx.filter(model, eqx.is_inexact_array_like)

        self.optimizer = self.opt_config.configure_optimizer(self.trainer_config.train_steps)
        opt_state      = self.optimizer.init(model)
        opt_state      = self.policy.cast_to_param(opt_state)

        if self.parallelism:
            model     = hax.shard_with_axis_mapping(model,     self.param_mapping, self.mesh)
            opt_state = hax.shard_with_axis_mapping(opt_state, self.param_mapping, self.mesh)

        return TrainerState(
            model         = model,
            opt_state     = opt_state,
            master_key    = key, 
            step          = 0,
            best_val_loss = np.inf
        )

    def get_batch(self, split: str, key):
        data                = self.train_data if split == "train" else self.val_data
        model_key, data_key = jr.split(key) 
        data_keys           = jr.split(data_key, self.Batch.size) # keys to randomly sample from dataset

        @partial(jax.vmap, in_axes=(None, 0))
        def get_block(array, index):
            return jax.lax.dynamic_slice(array, (index,), (self.Pos.size,))

        random_indices = jr.randint(key, (self.Batch.size,), 0, len(data) - self.Pos.size) 
        X              = hax.named(get_block(data, random_indices),     (self.Batch, self.Pos))
        y              = hax.named(get_block(data, random_indices + 1), (self.Batch, self.Pos))

        return Batch(
            input_tokens  = X, 
            target_tokens = y, 
            model_key     = model_key
        )

    @cached_property
    def loss_fn(self):
        def _loss_fn(model, batch: Batch) -> Array:
            model = self.policy.cast_to_compute(model)
            y_hat = model(batch.input_tokens, key=batch.model_key)
            y     = hnn.one_hot(batch.target_tokens, self.Vocab)
            loss  = hnn.cross_entropy_loss(y_hat, self.Vocab, y)
            return loss.scalar()

        if self.parallelism:
            return hax.named_jit(
                in_axis_resources = self.param_mapping | self.compute_mapping, 
                axis_resources    = self.compute_mapping
            )(_loss_fn)
        else: 
            return eqx.filter_jit(_loss_fn)

    @cached_property
    def accumulate_gradients(self):
        
        def _init_grad(model: GPT2) -> GPT2:
            params  = eqx.filter(model, eqx.is_inexact_array_like)
            grad    = jax.tree_util.tree_map(jnp.zeros_like, params)
            return grad

        grad_fn = eqx.filter_value_and_grad(self.loss_fn)
        if self.parallelism:
            grad_fn = hax.named_jit(grad_fn, out_axis_resources=self.param_mapping)
            # ==============================
            # instead of having each device contain the gradients for its microbatch
            # (i.e., doing hax.named_jit(grad, out_axis_resources=self.compute_mapping), 
            # we have each device contain the gradients for the entire batch with respect
            # to a certain susbet of parameters – this way, the whole gradient doesn't need 
            # to be stored on any one device (so instead of keeping all the gradients 
            # distinct and averaging at the end, we average parts of them at each step). 
            # ==============================
            _init_grad = hax.named_jit(grad, out_axis_resources=self.param_mapping) 

        def _acc_fn(acc, key, model):
            (loss, grad)           = acc
            batch                  = self.get_batch('train', key)
            (this_loss, this_grad) = eqx.filter_value_and_grad(self.loss_fn)(model, batch)

            new_loss = loss + this_loss
            new_grad = eqx.apply_updates(grad, this_grad)
            return (new_loss, new_grad)

        def _acc(model, keys) -> Tuple[Array, GPT2]: 
            fun = partial(_acc_fn, model=model)
            (loss, grad) = hax.fold(fun, self.AccumSteps)((0.0, _init_grad(model)), keys) 
            grad         = jax.tree_map(lambda x: x / self.AccumSteps.size, grad)
            loss        /= self.AccumSteps.size
            return (loss, grad)

        return _acc

    @cached_property
    def update(self):

        @eqx.filter_jit
        def _update(model, opt_state, data_keys) -> Tuple[Array, GPT2, OptState]:
            (loss, grad)       = self.accumulate_gradients(model, data_keys)
            updates, opt_state = self.optimizer.update(grad, opt_state, model)
            model              = eqx.apply_updates(model, updates)
            return (model, opt_state, loss)

        return _update

    
    def train_step(self):
        key, data_key = jr.split(self.master_key)
        data_keys     = jr.split(data_key, self.AccumSteps.size)
        model, opt_state, loss = self.update(self.model, self.opt_state, data_keys)

        return TrainerState(   
            model         = model,
            opt_state     = opt_state,
            master_key    = key,
            step          = self.step + 1,
            best_val_loss = self.best_val_loss
        ), loss

        
    # @cached_property
    # def train_step(self):   

    #     @eqx.filter_jit
    #     def _train_step(state) -> Tuple[TrainerState, Array]:
    #         key, data_key = jr.split(state.master_key)
    #         data_keys     = jr.split(data_key, self.AccumSteps.size)

    #         (loss, grad)       = self.accumulate_gradients(state.model, data_keys)
    #         updates, opt_state = self.optimizer.update(grad, state.opt_state, state.model)
    #         model              = eqx.apply_updates(state.model, updates)

    #         return TrainerState(   
    #             model         = state.model,
    #             opt_state     = state.opt_state,
    #             master_key    = key,
    #             step          = self.step + 1,
    #             best_val_loss = self.best_val_loss
    #         ), loss
        
    #     return _train_step
    
    def val_step(self, split: str) -> Array:
        model     = eqx.nn.inference_mode(self.model, value = (split == 'train'))
        data_keys = jr.split(self.master_key, self.EvalSteps.size)

        def _step(loss, key):
            batch       = self.get_batch(split, key)
            this_loss   = self.loss_fn(model, batch)
            return loss + this_loss
        
        loss_acc = hax.fold(_step, self.EvalSteps)(0.0, data_keys)
        return loss_acc / self.EvalSteps.size

    def train(self):
        t0 = time.time()
        running_mfu = -1.0

        while self.step <= self.trainer_config.train_steps:
            if self.step % self.trainer_config.eval_interval == 0:
                train_loss = self.val_step('train')
                val_loss   = self.val_step('eval')

                if self.trainer_config.wandb_log:
                    wandb.log({
                        'step'       : self.step,
                        'train/loss' : train_loss,
                        'val/loss'  : val_loss,
                        'lr'         : self.opt_config._build_lr_scheduler(self.trainer_config.train_steps)(self.step),
                    })
                print(f'Step: {self.step} | Average Train Loss: {train_loss:.2f} | Average Val Loss: {val_loss:.2f}')

                if val_loss < self.best_val_loss or self.trainer_config.always_save_ckpt:
                    self.best_val_loss = val_loss

                    sd   = self.model.to_state_dict()
                    ckpt = asdict(self.state) | {
                        'model'        : sd,
                        'model_config' : self.model_config,
                        'opt_state'    : self.opt_state,
                        'train_key'    : self.master_key,
                    }
                    
                    print(f'Saving checkpoint to {self.trainer_config.output_dir}')
                    torch.save(ckpt, os.path.join(self.trainer_config.output_dir, 'ckpt.pt'))

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            self.state, loss = self.train_step() #self.train_step(self.state)

            if self.step % self.trainer_config.log_interval == 0:
               #print(loss)
                mfu = self.model.estimate_mfu(self.Batch.size * self.AccumSteps.size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"step: {self.step}, loss: {loss:.4f} time: {dt*1000:.2f}ms, mfu: {running_mfu*100:.2f}%")

    def _resume(self):
        if self.trainer_config.resume_from is None:
            return self.init_state()
        elif 'gpt2' in self.config.init_from:
            model = GPT2LMHeadModel.from_pretrained(self.config.init_from)
            sd    = model.state_dict()
            return self._init_state()
        else: 
            checkpoint = torch.load(self.config.init_from)
            sd         = ckpt['model'].state_dict()
            self.model_config = ckpt['model_args']

            model = GPT2(self.model_config).from_state_dict(sd, prefix='transformer')

            return TrainerState(
                step      = ckpt['step'],
                model     = model,
                opt_state = ckpt['opt_state'],
                train_key = ckpt['train_key'],
                best_val_loss = ckpt['best_val_loss']
            )

@draccus.wrap()
def main(cfg: TrainingRunConfig):
    #from jax.config import config
    #config.update('jax_disable_jit', True)
    trainer = GPT2Trainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
