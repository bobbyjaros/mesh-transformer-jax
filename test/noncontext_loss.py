import jax
import jax.numpy as jnp
import numpy as np
from numpy import random

BACKGROUND_LEN = 5
MASKED_TOKENS_START = 1000

def set_background_tokens(targets):
  y = np.ones(targets.shape[-1])
  y[0:BACKGROUND_LEN] = 0.0
  z = np.zeros(targets.shape[-1])
  z[0:BACKGROUND_LEN] = MASKED_TOKENS_START
  targets = targets * y + z
  return jax.lax.stop_gradient(targets)

def loss(x, targets, z_loss=1):
  #############
  # T_predict = targets.shape[-1]
  # x = x[:,:,-T_predict:,:]
  #############

  logits = x
  print(f"targets: {targets}")
  targets = set_background_tokens(targets)
  print(f"targets: {targets}")

  global_max = logits.max(-1, keepdims=True)
  logits -= jax.lax.stop_gradient(global_max)

  # BJ: the logits of the target indices:
  gt_onehot = jax.nn.one_hot(targets, dim)
  predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
  if True:
    # Mask: no gradient on pad or style_vecs
    mask = jnp.less(targets, 3)
    predicted_logits *= mask

  exp_logits = jnp.exp(logits)

  sum_exp_logits = exp_logits.sum(axis=-1)

  loss = jnp.log(sum_exp_logits) - predicted_logits

  # BJ?: what is this?
  loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()

  correct = (0.0 == predicted_logits)

  return loss, correct


if __name__ == "__main__":
  num_shards = 2
  batch_size = 3
  T = 8
  T_predict = T # 3
  dim = 6
  x = jnp.array(random.random((num_shards,batch_size,T,dim)))
  targets = jnp.array(random.random_integers(0,dim-1,(num_shards,batch_size,T)))
  # targets = np.arange(num_shards*batch_size*T_predict).reshape((num_shards,batch_size,T_predict))
  l, correct = loss(x, targets)