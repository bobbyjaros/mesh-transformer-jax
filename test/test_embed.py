from mesh_transformer.layers import EmbeddingShard
import json
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def embed_wrap(x):
    config = json.load(open("configs/6B_twitter_8.json"))
    mod = EmbeddingShard(config)
    x = mod(x)
    # Could stop here but we actually want to sum along the shard:
    x = jax.lax.psum(x, "shard")
    return x

if jax.device_count() == 1:
    mesh_shape = (1,1)
else:
    cores_per_replica = 4
    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    # mesh_shape (2,4) -> batch, shard
devices = np.array(jax.devices()).reshape(mesh_shape)
print(f"devices.shape {devices.shape}")

key = hk.PRNGSequence(42)
# This is also the only way I see to let us name the shard axis in init_xmap()
mp_per_host = cores_per_replica
key = jnp.array(key.take(mp_per_host))
# x = jnp.array([37, 42, 16, 18, 0, 3])
x = jnp.array([[37, 42, 16], [13, 0, 3], [2, 9, 4], [7, 8, 9], 
      [10, 11, 12], [13, 14, 13]])

with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    init_fn = hk.transform(hk.experimental.optimize_rng_use(embed_wrap)).init
    # This doesn't work because it doesn't let us name the shard dim:
    # init_fn2 = lambda x : init_fn(rng, x)
    init_xmap = jax.experimental.maps.xmap(fun=init_fn,
                           in_axes=(["shard", ...],["batch", ...]),
                           out_axes=["shard", ...],
                           axis_resources={'shard': 'mp', 'batch': 'dp'})
    params = init_xmap(key, x)
    print(f"params w.shape: {params['embedding_shard/~/linear']['w'].shape}")

    embed_fn = hk.without_apply_rng(hk.transform(embed_wrap)).apply
    # This doesn't work because it doesn't let us name the shard dim:
    # embed = lambda x : embed_fn(params, x)
    embed_xmap = jax.experimental.maps.xmap(fun=embed_fn,
                           in_axes=(["shard", ...],["batch", ...]),
                           out_axes=["batch", ...],
                           axis_resources={'shard': 'mp', 'batch': 'dp'})
    res = embed_xmap(params, x)
    print(f"res.shape {res.shape}")
    # Test that the same input produces the same embedding:
    assert (res[1,0,:] == res[5,2,:]).all()
