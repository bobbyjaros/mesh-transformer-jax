
import haiku as hk
import jax
import jax.numpy as jnp

def f(x):
  mod = hk.Linear(10)
  return mod(x)
f = hk.without_state(hk.transform_with_state(f))
rng = jax.random.PRNGKey(42)
x = jnp.zeros([2, 3, 4])
params = f.init(rng, x)
out = f.apply(params, rng, x)
print(out.shape)