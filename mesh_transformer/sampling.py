import jax
import jax.numpy as jnp


# takes in a logit distribution, softmax and then sample
def softmax_sample(key, logits, _, temp=1):
    ind = jax.random.categorical(key, logits/temp, -1).astype(jnp.uint32)
    return ind, logits[0,ind]


def nucleaus_filter(logits, top_p=0.9, top_k=None):
    sorted_logits = jnp.sort(logits)[:, ::-1] # sort descending
    sorted_indices = jnp.argsort(logits)[:, ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
    # -> Rows are cumulative up to 1.0

    if top_k is not None:
        # Keep only top_k tokens
        # BJ: Set logits below top_k threshold to a big negative number
        indices_range = jnp.arange(len(sorted_indices[0]))
        indices_range = jnp.stack([indices_range] * len(sorted_indices), axis=0)

        sorted_indices_to_remove = jnp.where(indices_range > top_k, sorted_indices, 0)
        # e.g. [[0, 0, 0, 1, 2],
        #       [0, 0, 0, 4, 3],
        #       [0, 0, 0, 3, 2],
        #       [0, 0, 0, 1, 4]]

        _, indices_to_remove = jax.lax.sort_key_val(sorted_indices, sorted_indices_to_remove)
        # e.g [[0, 1, 2, 0, 0],
        #      [0, 0, 0, 3, 4],
        #      [0, 0, 2, 3, 0],
        #      [0, 1, 0, 0, 4]]

        logit_mask = 1e10 * indices_to_remove

        logits -= logit_mask

    # Remove tokens with cumulative probability above a threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.concatenate((jnp.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove), axis=-1)[:, :-1]

    _, indices_to_remove = jax.lax.sort_key_val(sorted_indices, sorted_indices_to_remove)

    logit_mask = 1e10 * indices_to_remove

    logits -= logit_mask

    return logits

"""
BJ:
Returns:
    (ind, logit)
"""
def nucleaus_sample(key, logits, _, top_p=0.9, temp=1, top_k=None):
    log_partition = jnp.log(jnp.sum(jnp.exp(logits)))
    logits = nucleaus_filter(logits, top_p, top_k=top_k)

    ind, logit = softmax_sample(key, logits, None, temp=temp)
    return ind, logit - log_partition


if __name__ == "__main__":
    import numpy as np
    logits = np.array([[-2, -1, 0, 0.8, 0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, -3]])
    # print(nucleaus_filter(logits))
    key = jax.random.PRNGKey(42)
    ind, log_prob = nucleaus_sample(key, logits, None)