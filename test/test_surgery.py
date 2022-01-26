"""
Starting with a saved BJNetwork, load those values into a new BJNetwork2
that has a new structure (which is a superset of the old).

Relevant pattern is shown in update_param_fields().

Also (first) tests saving a network, reloading, and recomputing.
"""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import io
import functools

class BJLayer(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dim1 = 5
        self.dim2 = 6
        self.q = hk.Linear(self.dim1, with_bias=False)
        self.v = hk.Linear(self.dim2, with_bias=False)

    def __call__(self, x):
        y = self.q(x)
        z = self.v(y)
        return z

class BJNetwork(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.nlayers = 3

        self.layers = []
        for layeri in range(self.nlayers):
            self.layers.append(BJLayer(name=f"bjlayer_{layeri}"))

    def __call__(self, x):
        # Note what happens if you get rid of this 2 lines:
        for layer in self.layers:
          x = layer(x)
        return x

## Same thing but add an additional linear layer in the middle.
class BJLayer2(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dim1 = 5
        self.dim2 = 6
        self.q = hk.Linear(self.dim1, with_bias=False)
        self.v = hk.Linear(self.dim2, with_bias=False)
        # New one:
        self.additional = hk.Linear(self.dim1, with_bias=False, name="additional")

    def __call__(self, x):
        y = self.q(x)
        y = self.additional(y) # <-- new
        z = self.v(y)
        return z

## Same thing but with new layer.
class BJNetwork2(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.nlayers = 3

        self.layers = []
        for layeri in range(self.nlayers):
            self.layers.append(BJLayer2(name=f"bjlayer2_{layeri}"))

    def __call__(self, x):
        # Note what happens if you get rid of this 2 lines:
        for layer in self.layers:
          x = layer(x)
        return x


# Write
def write_ckpt(pytree, ckpt_dir):
    flattened, structure = jax.tree_flatten(pytree)

    # cpu_flattened = index_weights(flattened, 0)
    cpu_device = jax.devices("cpu")[0]
    cpu_flattened = jax.device_put(flattened, cpu_device)

    file_path = ckpt_dir + f"ckpt.npz"
    with open(file_path, "wb") as f:
        np.savez(f, *cpu_flattened)

# Read
def read_file(ckpt_dir):
    out = []
    file_path = ckpt_dir + f"ckpt.npz"
    with open(file_path, "rb") as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        deserialized = np.load(f_io)
        for i in deserialized:
            out.append(deserialized[i])
    return out

def read_ckpt(pytree, dir):
    old_flattened, structure = jax.tree_flatten(pytree)

    new_flattened = read_file(dir)

    loaded_pytree = jax.tree_unflatten(structure, new_flattened)

    return loaded_pytree


def is_match(key1,key2):
    tokens1 = key1.split('/~/')
    tokens2 = key2.split('/~/')
    layer1 = tokens1[1].split('_')[-1]
    layer2 = tokens2[1].split('_')[-1]
    is_match = layer1 == layer2 and tokens1[-1] == tokens2[-1]
    return is_match

def update_param_fields(params_from, params_to):
    # This was the key line that unblocked me from doing model surgery:
    params_to = hk.data_structures.to_mutable_dict(params_to)
    for key_to in params_to.keys():
        for key_from,val_from in params_from.items():
            if is_match(key_to,key_from):
                print(f"{key_from} --> {key_to}")
                params_to[key_to] = val_from
    return hk.data_structures.to_immutable_dict(params_to)

def update_param_fields_OLD(params_from, params_to):
    flattened_from, structure_from = jax.tree_flatten(params_from)
    key_order = [key for key in params_to.keys()]
    flattened_to, structure_to = jax.tree_flatten(params_to)
    key_order2 = [key for key in params_to.keys()]
    # HACK:
    params_to_order = jax.tree_unflatten(structure_to, flattened_to)
    key_order3 = [key for key in params_to_order.keys()]
    # key_lookup_from = dict([(key,ki) for (ki,key) in enumerate(params_from.keys())])
    for ki_to,key_to in enumerate(params_to_order.keys()):
        for ki_from,key_from in enumerate(params_from.keys()):
            if is_match(key_to,key_from):
                print(f"{ki_from} ({key_from}) --> {ki_to} ({key_to})")
                flattened_to[ki_to] = flattened_from[ki_from]
        #flattened_to[ki] = flattened_from[key_lookup_from(key)]
    print(params_to_order.keys())
    params_to2 = jax.tree_unflatten(structure_to, flattened_to)
    print(params_to2.keys())
    import pdb ; pdb.set_trace()
    return params_to2


if __name__ == "__main__":
    # n = BJNetwork()
    # -> All `hk.Module`s must be initialized inside an `hk.transform`.

    # 1) Create orig network
    def eval(x):
        network = BJNetwork()
        return network(x)
    param_init_fn, apply_fn = hk.without_apply_rng(hk.transform(eval))

    x = jnp.ones((3,4))
    seed = 42
    key = jax.random.PRNGKey(seed)
    params = param_init_fn(key, x)

    out = apply_fn(params, x)
    # out = apply_fn(params, key, x) # If no without_apply_rng
    print(out)

    # Compute manually:
    y = x
    for lname in [
        'bj_network/~/bjlayer_0/~/linear',
        'bj_network/~/bjlayer_0/~/linear_1',
        'bj_network/~/bjlayer_1/~/linear',
        'bj_network/~/bjlayer_1/~/linear_1',
        'bj_network/~/bjlayer_2/~/linear',
        'bj_network/~/bjlayer_2/~/linear_1'
    ]:
      y = jnp.matmul(y,params[lname]['w'])
    assert (out == y).all()

    # Save
    write_ckpt(params, "test_ckpt/")


    ## 2) Load matches?
    new_flattened = read_file("test_ckpt/")
    _, structure = jax.tree_flatten(params)
    params_from_file = jax.tree_unflatten(structure, new_flattened)
    out_from_file = apply_fn(params_from_file, x)
    assert (out == out_from_file).all()


    ## 3) Perform surgery
    # Create uninitated network with new structure
    def eval2(x):
        network = BJNetwork2()
        return network(x)
    param_init_fn2, apply_fn2 = hk.without_apply_rng(hk.transform(eval2))

    key2 = jax.random.PRNGKey(seed + 1)
    params2_rand = param_init_fn2(key2, x)

    params2 = update_param_fields(params, params2_rand)
    out2 = apply_fn2(params2, x)
    print(out2)

    ## Compute manually
    y = x
    for p,lname in [
        (params,'bj_network/~/bjlayer_0/~/linear'),
        (params2_rand,'bj_network2/~/bjlayer2_0/~/additional'),
        (params,'bj_network/~/bjlayer_0/~/linear_1'),
        (params,'bj_network/~/bjlayer_1/~/linear'),
        (params2_rand,'bj_network2/~/bjlayer2_1/~/additional'),
        (params,'bj_network/~/bjlayer_1/~/linear_1'),
        (params,'bj_network/~/bjlayer_2/~/linear'),
        (params2_rand,'bj_network2/~/bjlayer2_2/~/additional'),
        (params,'bj_network/~/bjlayer_2/~/linear_1')
    ]:
      y = jnp.matmul(y,p[lname]['w'])
    assert (out2 == y).all()
    # print(y)
    print(f"Surgery matches manual computation")

    s2 = hk.data_structures.to_immutable_dict(structure)
