import numpy as np
import jax.numpy as jnp
import time

"""
Returns a mask that will exclude tokens between the markers (inclusive of first
marker and exclusive of second marker -- presumes there is a second mask
that will filter out the special tokens, including the markers).
Note that this method treats the two marker_tokens identically, and is therefore
not robust to omitting the first marker_token1, it may return the opposite of what
was intended.

Example:
marker_token1: 98
marker_token2: 99

input:
[[ 1  2  3 98  5  6 99  8]
 [11 12 98 14 15 16 17 99]
 [98 22 99 24 25 98 27 99]
 [98 32 99 34 35 98 37 38]
 [41 42 99 44 45 98 47 99]]

output:
[[ 1  1  1  0  0  0  1  1]
 [ 1  1  0  0  0  0  0  1]
 [ 0  0  1  1  1  0  0  1]
 [ 0  0  1  1  1  0  0  0]
 [ 1  1  0  0  0  1  1  0]]  # Reversed order of 98 and 99 --> opposite intent?
"""
def mask_exclude_between_markers(targets, marker_token1, marker_token2):
    neg_targets = jnp.equal(targets, marker_token1) + jnp.equal(targets, marker_token2)
    neg_targets = -2*neg_targets + 1
    between = jnp.cumprod(neg_targets, axis=-1)
    # print(f"between:\n{between}")
    keep = jnp.greater(between, 0)
    return keep

if __name__ == "__main__":
    ## mask_exclude_between_markers
    targets = jnp.array([
        [  1,  2,  3, 98,  5,  6,  99,  8],
        [ 11, 12, 98, 14, 15, 16,  17, 99],
        [ 98, 22, 99, 24, 25, 98,  27, 99], # multiple
        [ 98, 32, 99, 34, 35, 98,  37, 38], # buggy: no end
        [ 41, 42, 99, 44, 45, 98,  47, 99], # buggy: no start
        ])
    print(f"input targets:\n{targets}")
    res = mask_exclude_between_markers(targets, 98, 99)
    # print(f"mask_exclude_between_markers:\n{res}")
    print(f"Masked targets:\n{targets*res}")
