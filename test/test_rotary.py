
from jax._src.lax.control_flow import X
import jax.numpy as jnp
import jax.random as random
from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# def fixed_pos_embedding(x, seq_dim=0):
#     dim = x.shape[-1]
#     inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

#     sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq)

#     return np.sin(sinusoid_inp), np.cos(sinusoid_inp)
"""
BJ: shift_to_start_pos produces the embedding that would correspond to that start_pos
"""
def fixed_pos_embedding(x, seq_dim=0, shift_to_start_pos=0):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

    seq_len = x.shape[seq_dim]
    print(f"seq_len: {seq_len}")
    # i.e. sinusoid_inp[i,j] = i * inv_freq[j]
    sinusoid_inp = np.einsum('i , j -> i j', np.arange(shift_to_start_pos, shift_to_start_pos+seq_len), inv_freq)
    # shape (T, dim), where each row is an inverse decay from 1 to small, scaled by the timestep t.
    print(f"sinusoid_inp.shape: {sinusoid_inp.shape}")

    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2)[-x.shape[0]:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


# v2:
# def rotate_every_two_v2(x):
#     x1 = x[:, :, :, ::2]
#     x2 = x[:, :, :, 1::2]

#     y = jnp.stack((-x2, x1), axis=-1)

#     z = rearrange(y, '... d j -> ... (d j)')
#     # print(x[0,0,0,0:10])
#     # print(y[0,0,0,0:10])
#     # print(z[0,0,0,0:10])
#     return z

# def apply_rotary_pos_emb_v2(x, sincos):
#     sin, cos = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[-3]:, None, :], sincos)
#     return (x * cos) + (rotate_every_two_v2(x) * sin)

# x cos(i) + y sin(i)
# y cos(i) - x sin(i) 


b = 3
t = 100
h = 2 # Heads
d = 36
d_rotary = 24

key = random.PRNGKey(42)
k1, key = random.split(key)
k = random.permutation(k1, jnp.arange(t*h*d)).reshape((t,h,d))
k_rot = k[:, :, :d_rotary]
k_pass = k[:, :, d_rotary:]

# v2:
# k = random.permutation(k1, jnp.arange(b*t*h*d)).reshape((b,t,h,d))
# k_rot = k[:, :, :, :d_rotary]
# k_pass = k[:, :, :, d_rotary:]


# Code in layers.py:
#   sincos = fixed_pos_embedding(k_rot)
#   k_rot = apply_rotary_pos_emb(k_rot, sincos)

sincos = fixed_pos_embedding(k_rot)
sin_orig, cos_orig = sincos
sincos_shift = fixed_pos_embedding(k_rot[70:], shift_to_start_pos=70)
sin_orig_shift, cos_orig_shift = sincos_shift
print(f"sin.shape: {sin_orig.shape} <-- d_rotary/2 = {d_rotary/2}")
plt.title('sins')
for i in range(sin_orig.shape[1]):
    plt.plot(sin_orig[:,i])
k_rot2 = apply_rotary_pos_emb(k_rot, sincos)
k_rot2_shift = apply_rotary_pos_emb(k_rot[70:], sincos_shift)

# Looking inside apply_rotary_pos_emb:
x = k_rot
sin, cos = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[-3]:, None, :], sincos)
k_rot_every2 = rotate_every_two(k_rot)
k_rot3 = (k_rot * cos) + (k_rot_every2 * sin)
print(f"k_rot {k_rot[0:4,0,0:8]}")
print(f"k_rot_every2 {k_rot_every2[0:4,0,0:8]}")

# Visualize result of apply_rotary_pos_emb:
plt.figure()
plt.subplot(2,3,1)
print(f"sin.shape: {sin.shape}")
# plt.imshow(sin[:,0,:])
plt.imshow(sin_orig)
plt.subplot(2,3,2)
plt.imshow(k_rot[:,0,:])
plt.title('k_rot')
plt.subplot(2,3,3)
plt.imshow(k_rot2[:,0,:])
plt.title('k_rot2')

sin_shift, cos_shift = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[-3]:, None, :], sincos_shift)
plt.subplot(2,3,4)
# plt.imshow(sin_256[:,0,:])
plt.imshow(sin_orig_shift)
plt.subplot(2,3,6)
plt.imshow(k_rot2_shift[:,0,:])
plt.title('k_rot2_256')

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(sin_orig)
# plt.subplot(1,3,2)
# plt.imshow(sin_orig_shift)
# plt.subplot(1,3,3)
# plt.imshow(sin_orig[70:] - sin_orig_shift)