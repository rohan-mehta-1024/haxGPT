import jax
import jax.numpy as jnp
import jax.random as jr 
import jax.sharding as sharding
import jax.experimental.mesh_utils as mesh_utils

import halaix as hax
import haliax.nn as hnn
import equinox as eqx
import equinox.nn as nn

import optax 
import time

# ------ config + download and format data ------
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300 
learning_rate = 1e-2
eval_iters = 200

# random seed
key = jr.PRNGKey(0)

# load the dataset
with open("input.txt", "r", encoding = "utf-8") as f:
    text = f.read()

# extract a vocabulary from the data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create an encoding and decoding scheme
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# create train and test splits
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# define axes 
Block = hax.Axis("block", block_size)
Batch = hax.axis("batch", batch_size)
Vocab = hax.axis("vocab", vocab_size)
Embed = hax.axis("embed", vocab_size)
# ------ config + download and format data ------


def get_batch(key, split):
    """Generate a single batch of training or testing data"""
    data = train_data if split == "train" else val_data
    keys = jr.split(key, Batch.size)
    random_indices = jr.randint(key, (Batch.size,), 0, len(data) - Block.size)

    def get_block(array, index):
        return jax.lax.dynamic_slice(array, (index,), (Block.size,))
    
    get_blocks = jax.vmap(get_block, in_axes=(None, 0))
    x = get_blocks(data, random_indices)
    y = get_blocks(data, random_indices + 1)
    return x, y

@jax.jit
def estimate_loss(key):
    out = {}
    for split in ["train", "val"]:
        losses = jnp.zeros((eval_iters,))
        for k in range(eval_iters):
            x, y = get_batch(key, split)
            losses = losses.at[k].set(loss_fn(model, x, y))
        out[split] = jnp.mean(losses)
    return out

# super simple bigram model
class BigramLM(eqx.Module):
    embedding_table: hnn.Embedding

    def __init__(self, vocab_size, key):
        # each token directly reads off the logits for the next token from a lookup table
        self.embedding_table = nn.Embedding(
            num_embeddings = Vocab.size,
            embedding_size = Embed.size,
            key = key
        )

    def __call__(self, seq):
        # idx is a (block_size,) array of integers
        logits = self.embedding_table(seq) 
        return logits

    def generate(self, seq, max_new_tokens, key):
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(seq)
            # focus only on the prediction made by the last token, since this is a bigram model
            logits = logits[:, -1, :] 
            # generate a new random key
            key, subkey = jr.split(key)
            # use this to sample for the distribution predicted by the last token
            new_token = jr.categorical(key=subkey, logits=logits)
            # reshape the sampled token
            new_token = jnp.expand_dims(new_token, axis=-1)
            # concatenate it to all the previous tokens, which includes the prompt plus everything generated so far
            idx = jnp.concatenate([idx, new_token], axis=-1)
        return idx

        # apply softmax to get a probability distribution over the vocabulary
        
def cross_entropy(logits_batch, labels_batch):
    logits_batch = jax.nn.log_softmax(logits_batch)
    # labels_batch is a (batch_size, block_size, vocab_size) array of logits
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, -1), axis=2)
    # we take the negative mean of the log probabilities of the correct tokens
    return -jnp.mean(pred_y)

#@eqx.filter_jit
@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0))
def loss_fn(model, x, y):
    preds = model(x)
    return cross_entropy(preds, y)

model = BigramLM(vocab_size, key)

# create and initialize optimizer
optim = optax.adamw(learning_rate)
opt_state = optim.init(model)

# training loop
for iter in range(max_iters):
    #print(iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(key)
        print(f"iter {iter} | train loss: {losses['train']:.2f} | val loss: {losses['val']:.2f}")

    # sample a batch of data
    x, y = get_batch(key, "train")
    # calculate the loss and compute gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    # pass the gradients and current model parameters to the optimizer to calculate updates
    updates, opt_state = optim.update(grads, opt_state, params=model)
    # apply these updates to the model
    model = eqx.apply_updates(model, updates)

# generate from the model
context = jnp.zeros((1,1), dtype=jnp.int32) 
print(decode(model.generate(context, max_new_tokens = 100, key=key)[0].tolist()))
t2 = time.time()
print(t2 - t1)


# 40 sec for 1000 no jit
# 154 sec for 1000 eqx.filter_jit
# 89 sec for normal jit 
# 85 sec for 3000 no jit
# 111 sec for 3000 jit