import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

from mesh_transformer.util import f_psum, g_psum, maybe_shard, head_print
from mesh_transformer.mask import mask_exclude_between_markers
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import thread_resources

# DO_BACKGROUND is a mode where the first #BACKGROUND_LEN tokens in each sample are background tokens;
#               During training we don't enforce a loss on the targets in the background.
#               During sampling we always have the background.
# DO_CAUSAL_BOTTLENECK is a GPT modification where we append #STYLE_VECS_LEN tokens to the background, and
#               add edit the causal mask such that generated tokens can see the style vectors but not the
#               background itself. Hence the information from the background can only be passed through
#               the states produced in the style vector timesteps.
# DO_IGNORE_BACKGROUND simply pretends like there's no background (or style vectors).
DO_BACKGROUND = True
DO_CAUSAL_BOTTLENECK = True
DO_IGNORE_BACKGROUND = False # Predict totally ignores the background
BACKGROUND_LEN = 1024
STYLE_VECS_LEN = 64
MASKED_TOKENS_START = 50257 # Mask timesteps with target with this token value or higher.
# REPLIES:  the tfrecords will surround the tokens corresponding to tweet this user is replying to with markers.
#           Assumed to come after the style vecs in the tokenizer.
# KEYWORDS: the tfrecords prompt each tweet with 1 or multiple tokens, surrounded by markers. So the challenge
#           is to produce sensible tweets containing those word(s).
TFRECORDS_FORMAT = ["REPLIES", "KEYWORDS"][0]
REPLY_START_TOKEN_ID = MASKED_TOKENS_START + STYLE_VECS_LEN + 1
REPLY_END_TOKEN_ID = MASKED_TOKENS_START + STYLE_VECS_LEN + 2

class ReplicatedLayerNorm(hk.Module):
    def __init__(self, offset=True):
        super().__init__()
        self.offset = offset

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = hk.get_parameter("scale", param_shape, inputs.dtype, init=jnp.ones)
        scale = jax.lax.all_gather(scale, "shard")[0]

        offset = hk.get_parameter("offset", param_shape, inputs.dtype, init=jnp.zeros)
        offset = jax.lax.all_gather(offset, "shard")[0]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + 1e-5)
        if self.offset:
            return inv * (inputs - mean) + offset
        else:
            return inv * (inputs - mean)


class RMSNorm(hk.Module):
    def __init__(self, offset, elementwise):
        super().__init__()
        self.offset = offset
        self.elementwise = elementwise

    def __call__(self, x):
        param_shape = (x.shape[-1],) if self.elementwise else ()
        normed = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)

        scale = hk.get_parameter('scale', param_shape, init=hk.initializers.Constant(x.shape[-1] ** 0.5))
        scale = jax.lax.pmean(scale, "shard")
        normed = normed * scale

        if self.offset:
            offset = hk.get_parameter('offset', param_shape, init=jnp.zeros)
            offset = jax.lax.pmean(offset, "shard")
            normed = normed + offset

        return normed


def getnorm(type):
    if type == "layernorm":
        return ReplicatedLayerNorm()
    if type == "layernorm-desync":
        return hk.LayerNorm(-1, True, True)
    elif type == "layernorm-nobias":
        return ReplicatedLayerNorm(offset=False)
    elif type == "rmsnorm":
        return RMSNorm(False, True)
    elif type == "scalenorm":
        return RMSNorm(False, False)
    elif type == "rmsnorm-bias":
        return RMSNorm(True, True)
    elif type == "scalenorm-bias":
        return RMSNorm(True, False)
    else:
        raise Exception("Not implemented")


class RelativePositionEmbs(hk.Module):
    @staticmethod
    def _relative_position_bucket(relative_position,
                                  num_buckets=32,
                                  max_distance=128):
        ret = 0
        n = -relative_position
        n = np.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = (n < max_exact)
        val_if_large = max_exact + (
                np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                np.log(max_distance / max_exact) *
                (num_buckets - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    def __call__(self, qlen, klen, heads, num_buckets):
        """Produce relative position embedding attention biases.
        Returns:
          output: `(heads, q_len, k_len)` attention bias
        """
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(relative_position)
        relative_attention_bias = hk.get_parameter('rel_embedding', [heads, num_buckets],
                                                   init=hk.initializers.TruncatedNormal(stddev=0.02))
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(
            relative_attention_bias.dtype)
        # --> shape (qlen, klen, num_heads)
        values = jax.lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (
                ((1,), (0,)),  # rhs, lhs contracting dims
                ((), ())))  # no batched dims
        return values

"""
BJ:
Args:
    seq_dim: which dim of x corresponds to timesteps.
    shift_to_start_pos: produces the embedding that would correspond to that start_pos
"""
def fixed_pos_embedding(x, seq_dim=0, shift_to_start_pos=0):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

    seq_len = x.shape[seq_dim]
    range = jnp.arange(seq_len) + shift_to_start_pos
    # i.e. sinusoid_inp[i,j] = i * inv_freq[j]
    sinusoid_inp = jnp.einsum('i , j -> i j', range, inv_freq)
    # shape (T, dim), where each row is an inverse decay from 1 to small, scaled by the timestep t.

    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2)[-x.shape[0]:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


def rotate_every_two_v2(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb_v2(x, sincos):
    sin, cos = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (rotate_every_two_v2(x) * sin)


class EmbeddingShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        in_dim = config["n_vocab"]
        out_dim = config["d_model"]
        shards = config["cores_per_replica"]

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_dim_per_shard = in_dim // shards
        self.out_dim_per_shard = out_dim // shards

        if config["pe"] == "fixed":
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            self.positional_embeddings = hk.get_parameter('pos_embs', [config["seq"], self.out_dim_per_shard], init=embed_init)
        else:
            self.positional_embeddings = None

        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        shard_start_index = jax.lax.axis_index('shard') * self.in_dim_per_shard

        input_onehot = jax.nn.one_hot(x - shard_start_index, self.in_dim_per_shard)
        proj_out = self.proj(input_onehot)

        proj_out = g_psum(proj_out)

        if self.positional_embeddings is not None:
            all_pos_embed = jax.lax.all_gather(self.positional_embeddings, 'shard')

            all_pos_embed = hk.Flatten()(jnp.transpose(all_pos_embed, (1, 0, 2)))

            proj_out += all_pos_embed

        return proj_out


class EmbeddingShardV2(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        in_dim = config["n_vocab"]
        out_dim = config["d_model"]
        shards = config["cores_per_replica"]

        assert in_dim % shards == 0

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)))

    def __call__(self, x, dtype=jnp.bfloat16):
        input_onehot = jax.nn.one_hot(x, self.in_dim)
        input_onehot = maybe_shard(input_onehot, P("dp", None, "mp"))

        proj_out = self.proj(input_onehot)

        return proj_out


# We actually combine the FF and dense in one layer (i.e. compute in parallel) to minimize all reduces
# BJ: parameters per layer:
#   q: d_model x d_model   (actually 8 x d_model x d_model/8)
#   v: d_model x d_model   ""
#   k: d_model x d_model   ""
#   o: d_model x d_model   (actually 8 x d_model/8 x d_model)
#   dense_proj: d_model x 4*d_model     (actually 8 x d_model * d_model/2)
#            b: 4*d_model
#   dense_proj_o: 4*d_model x d_model   (actually 8 x d_model/2 * d_model)
#              b: d_model
#   total: 12 * (d_model**2) + 7*d_model
#
# e.g. d_model = 4096 ==> 201.4m params per layer
#  With 28 layers ==> 5.628m params in layer
#  Then add    + n_vocab (50400) * d_model = 206.4m
#              + another, for output = 206.4m
#       = 6.05B params
# (also saves replicated_layer_norm (offset and scale))
class TransformerLayerShard(hk.Module):
    def __init__(self, config, name=None, init_scale=1.):
        super().__init__(name=name)
        heads = config["n_heads"]
        dim = config["d_model"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])
        self.is_rotary = config["pe"] == "rotary"

        assert dim % heads == 0
        assert heads % shards == 0

        self.dim = dim
        self.dim_per_head = dim // heads
        # See qvk_proj().
        self.heads_per_shard = heads // shards
        self.dim_per_shard = dim // shards
        self.pe_rotary_dims = config.get("pe_rotary_dims", self.dim_per_head)

        self.norm = norm

        self.q = hk.Linear(self.dim_per_shard, with_bias=False)
        self.v = hk.Linear(self.dim_per_shard, with_bias=False)
        self.k = hk.Linear(self.dim_per_shard, with_bias=False)

        self.o = hk.Linear(self.dim, with_bias=False,
                           w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

        self.dense_proj = hk.Linear(self.dim_per_shard * 4)
        self.dense_proj_o = hk.Linear(self.dim,
                                      w_init=hk.initializers.TruncatedNormal(stddev=init_scale / np.sqrt(self.dim)))

    def self_attn(self, q, v, k, attn_bias, rotary_shift_to_start_pos=0):
        if self.is_rotary:
            k_rot = k[:, :, :self.pe_rotary_dims]
            k_pass = k[:, :, self.pe_rotary_dims:]

            q_rot = q[:, :, :self.pe_rotary_dims]
            q_pass = q[:, :, self.pe_rotary_dims:]

            sincos = fixed_pos_embedding(k_rot, shift_to_start_pos=rotary_shift_to_start_pos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)
            k_rot = apply_rotary_pos_emb(k_rot, sincos)

            k = jnp.concatenate([k_rot, k_pass], axis=-1)
            q = jnp.concatenate([q_rot, q_pass], axis=-1)

        # BJ:
        # t and T = timesteps (positions)
        # h = head
        # d = dim
        attention_logits = jnp.einsum("thd,Thd->htT", q, k)

        sqrt_key_size = np.sqrt(self.dim_per_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias

        attention_weights = jax.nn.softmax(attention_logits)
        # BJ: Apply attention and concat this shard's heads.
        attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v).reshape((-1, self.dim_per_shard))

        # BJ: We'll need to add this result to the self.o(.) from the other shards (other
        #     heads).  This happens via g_psum() in __call__().
        return self.o(attention_vec)

    def ff(self, x):
        dense_proj = self.dense_proj(x)
        dense_proj = jax.nn.gelu(dense_proj)
        return self.dense_proj_o(dense_proj)

    def qvk_proj(self, x):
        # Model parallelism (shards) splits along same axis as heads
        # e.g. 16 heads, 8 shards -> 2 heads_per_shard.
        q = self.q(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        v = self.v(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))
        k = self.k(x).reshape(x.shape[:-1] + (self.heads_per_shard, self.dim_per_head))

        return q, v, k

    def __call__(self, x, attn_bias):
        # head_print(f"x.shape: {x.shape}")
        x = f_psum(x)
        ## BJ: don't mess with length here, because it won't match targets & eval

        ## BJ: GPT-2 moved LayerNorm to the input.
        ##     See diagram in Fig 2 of Megatron-LM paper.
        x = self.norm(x)

        # In GPT-J, we apply the multihead attention module and the feedforward
        # module to the *same* input, and add the respective results, which becomes
        # the input to the next layer.
        #
        # The advantage, computationally, is that this saves you an all-reduce in the
        # forward pass and the backward pass. (See Shoeybi et al’s Megatron paper for
        # the model parallelism framework.)
        #
        # Aran Komatsuzaki:
        #   we compared the two approaches in a smaller scale and verified the parallel
        #   method performs slightly better in terms of downstreaming performance.
        # Ben Wang:
        #   slightly faster convergence, likely due to init
        #   but it was very close, and about 10% faster throughput
        #   so its worth it
        #   inspired by all-attn
        #   and layerdrop
        q, v, k = self.qvk_proj(x)
        # q,v,k shapes are (T, heads_per_shard, self.dim_per_head) i.e. the outputs for a single shard on a single sample
        #   e.g. 4096 dim, 16 heads, 8 shards --> 2 heads_per_shard, 256 dim_per_head. (Because 4096 // 16 = 256)

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if DO_CAUSAL_BOTTLENECK:
            # For example:
            #  BACKGROUND_LEN=3. STYLE_VECS_LEN=2 ==>
            #         |<--backgd--->|<-style->|<----gen----->
            #         col0 col1 col2 col3 col4 col5 col6 col7
            #  row0:    1    0    0    0    0    0    0    0
            #  row1:    1    1    0    0    0    0    0    0
            #  row2:    1    1    1    0    0    0    0    0
            #  row3:    1    1    1    1    0    0    0    0
            #  row4:    1    1    1    1    1    0    0    0
            #  row5:    0    0    0    1    1    1    0    0
            #  row6:    0    0    0    1    1    1    1    0
            #  row6:    0    0    0    1    1    1    1    1
            causal_mask[BACKGROUND_LEN+STYLE_VECS_LEN:,0:BACKGROUND_LEN] = 0.0
        if DO_IGNORE_BACKGROUND:
            nignore = BACKGROUND_LEN+STYLE_VECS_LEN
            causal_mask[:,0:nignore] = 0.0
            causal_mask[0:nignore,0:nignore] = np.eye(nignore)
        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        # BJ: attn_out and dense_out are (num_shards, batch_size, T, dims)
        return g_psum(attn_out + dense_out)

    def decode_once(self, decode_state, x, attn_bias):
        """iterate the decoding process by a single token
        BJ:
        Args:
            decode_state is
              {   k & v: (T=gen_length, heads_per_shard, dim_per_head)),
                  visible_ctx_length: length of context + length of gen so far.
              }
            x shape is (1, dim).  1 because we're generating one timestep at a time.
                                   *not* 1 because parent function was distributed via xmap. Same is true for e.g. __call__().
            attn_bias is 1-d
        """
        # print(f"BJ decode_once: x.shape: {x.shape}")
        x = f_psum(x)
        x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k = self.qvk_proj(x)
        # q,v,k shapes are (T=1, heads_per_shard, dim_per_head). See note above.

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=0)
        k = jnp.concatenate((decode_state["k"], k), axis=0)
        # BJ: keep same length:
        v = v[1:]
        k = k[1:]
        # v,k shapes are (T=orig_decode_state_length, heads_per_shard, dim_per_head)
        # q is still (1, heads_per_shard, dim_per_head) for the most recent step.

        # BJ: Try extend length, up to max_seq_len? No, doesn't work because
        #     generate_sample() uses jax.lax.scan, which requires output and input to
        #     be same shape. Else: "TypeError: scan carry output and input must have
        #     identical types".
        #         v = v[-self.max_seq_len:]
        #         k = k[-self.max_seq_len:]

        visible_ctx_length = decode_state["visible_ctx_length"] + 1
        length = v.shape[0]

        # BJ: visible_ctx_length is the context's length after get_init_decode_state()
        #     and increments by one every call to decode_once().
        #     It's actually an array, should be shape [1,1]
        masked_tokens = length - visible_ctx_length

        # If tokens_decoded = 3,
        # bias = [-1e10 -1e10 ... -1e10  0  0  0]
        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias
        # This is a 1-d bias.

        rotary_offset = decode_state["rotary_offset"] + 1
        if DO_BACKGROUND and self.is_rotary:
            # if DO_CAUSAL_BOTTLENECK:
            #     rotary_shift_to_start_pos = BACKGROUND_LEN + tokens_decoded - length
            # else:
            rotary_shift_to_start_pos = rotary_offset
            # print(f"rotary_shift_to_start_pos: {rotary_shift_to_start_pos}")
        else:
            rotary_shift_to_start_pos = 0
        attn_out = self.self_attn(q, v, k, bias, rotary_shift_to_start_pos)
        dense_out = self.ff(x)

        res = g_psum(attn_out + dense_out)
        new_decode_state = {
            "visible_ctx_length": visible_ctx_length,
            "rotary_offset": rotary_offset,
            "k": k,
            "v": v
        }
        return res, new_decode_state

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length, attn_bias, gen_length):
        """
        BJ:
        Args:
            x is 1-d.
            given_length is [1,1].
        """
        x = f_psum(x)
        x = self.norm(x)

        q, v, k = self.qvk_proj(x)

        full_length = x.shape[0]
        masked_tokens = full_length - given_length

        seq_len = x.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        if DO_CAUSAL_BOTTLENECK:
            causal_mask[BACKGROUND_LEN+STYLE_VECS_LEN:, 0:BACKGROUND_LEN] = 0.0

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        # If given_length = 3,
        # bias = [-1e10 -1e10 ... -1e10  0  0  0]
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)  # mask out zero tokens before context starts
        # BJ: For seq_len=7 and given_length=6:
        #      [[-BIG -BIG -BIG -BIG -BIG -BIG -BIG]
        #       [-BIG    1 -BIG -BIG -BIG -BIG -BIG]
        #       [-BIG    1    1 -BIG -BIG -BIG -BIG]
        #       [-BIG    1    1    1 -BIG -BIG -BIG]
        #       [-BIG    1    1    1    1 -BIG -BIG]
        #       [-BIG    1    1    1    1    1 -BIG]
        #       [-BIG    1    1    1    1    1    1]]
        #     Or if DO_CAUSAL_BOTTLENECK with BACKGROUND_LEN=2 and STYLE_VECS_LEN=2:
        #      [[-BIG -BIG -BIG -BIG -BIG -BIG -BIG]
        #       [-BIG    1 -BIG -BIG -BIG -BIG -BIG]
        #       [-BIG    1    1 -BIG -BIG -BIG -BIG]
        #       [-BIG    1    1    1 -BIG -BIG -BIG]
        #       [-BIG    1    1    1    1 -BIG -BIG]
        #       [-BIG -BIG -BIG    1    1    1 -BIG]
        #       [-BIG -BIG -BIG    1    1    1    1]]
        #
        bias += attn_bias  # finally add attn bias for rpe

        attn_out = self.self_attn(q, v, k, bias)
        dense_out = self.ff(x)

        res = g_psum(attn_out + dense_out)

        if DO_BACKGROUND and DO_CAUSAL_BOTTLENECK:
            # Assumes input is |-----------background-----------|--style--|-------predict---------|
            # i.e. no right padding to make even batch sizes. (TODO: should we have this?)
            initial_decode_state = {
                # BJ: enforces the bottleneck! TODO generalize
                "k": k[-gen_length:],
                "v": v[-gen_length:],
                "rotary_offset": full_length - gen_length,
                # visible_ctx_length is length of tokens that we can attend back to.
                "visible_ctx_length": full_length - BACKGROUND_LEN
            }
        else:
            initial_decode_state = {
                "k": k,
                "v": v,
                "rotary_offset": 0,
                "visible_ctx_length": given_length.astype(jnp.uint32)
            }
        return res, initial_decode_state


# This new class combines the input and output projection into one matmul for better efficiency
class TransformerLayerShardV2(hk.Module):
    def __init__(self, config, name=None, init_scale=1.):
        super().__init__(name=name)
        self.dim = config["d_model"]
        self.n_head = config["n_heads"]
        self.d_head = config["d_head"]
        self.d_rotary = config["pe_rotary_dims"]
        self.mp_num = thread_resources.env.shape['mp']

        self.norm = hk.LayerNorm(-1, True, True)
        self.input_proj = hk.Linear(self.d_head * self.n_head * 3 + self.dim * 8)
        self.output_proj = hk.Linear(self.dim,
                                     w_init=hk.initializers.TruncatedNormal(stddev=init_scale / jnp.sqrt(self.dim)))

    def self_attn(self, q, v, k, attn_bias):
        k_rot = k[:, :, :, :self.d_rotary]
        k_pass = k[:, :, :, self.d_rotary:]

        q_rot = q[:, :, :, :self.d_rotary]
        q_pass = q[:, :, :, self.d_rotary:]

        sincos = fixed_pos_embedding(k_rot, seq_dim=1)
        q_rot = apply_rotary_pos_emb_v2(q_rot, sincos)
        k_rot = apply_rotary_pos_emb_v2(k_rot, sincos)
        q_rot = maybe_shard(q_rot, P("dp", None, "mp", None))
        k_rot = maybe_shard(k_rot, P("dp", None, "mp", None))

        k = jnp.concatenate([k_rot, k_pass], axis=-1)
        q = jnp.concatenate([q_rot, q_pass], axis=-1)

        k = maybe_shard(k, P("dp", None, "mp", None))
        q = maybe_shard(q, P("dp", None, "mp", None))

        attention_logits = jnp.einsum("bthd,bThd->bhtT", q, k)

        attention_logits = maybe_shard(attention_logits, P("dp", "mp", None, None))

        sqrt_key_size = np.sqrt(self.d_head).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        attention_logits += attn_bias
        attention_logits = maybe_shard(attention_logits, P("dp", "mp", None, None))

        attention_weights = jax.nn.softmax(attention_logits)
        attention_weights = maybe_shard(attention_weights, P("dp", "mp", None, None))

        attention_vec = jnp.einsum("bhtT,bThd->bthd", attention_weights, v)

        attention_vec = maybe_shard(attention_vec, P("dp", None, "mp", None))
        sharded_attn_vec = attention_vec.reshape(attention_vec.shape[:2] + (self.mp_num, self.n_head//self.mp_num, -1))
        sharded_attn_vec = maybe_shard(sharded_attn_vec, P("dp", None, "mp", None, None))

        attention_vec = attention_vec.reshape(sharded_attn_vec.shape[:2] + (self.mp_num, -1))
        return maybe_shard(attention_vec, P("dp", None, "mp", None))

    # input: [batch, seq, dim]
    # output: [batch, seq, n_head, d_head]
    def head_split(self, x):
        reshaped = x.reshape(x.shape[:-1] + (self.n_head//self.mp_num, self.d_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1, ) + x.shape[-1:])

        # return reshaped
        return maybe_shard(reshaped, P("dp", None, "mp", None))

    def input(self, x):
        # [batch, seq, dim]
        projected = self.input_proj(x)

        # [batch, seq, mp, dim//mp]
        projected = maybe_shard(projected, P("dp", None, "mp"))
        mp_split = jnp.reshape(projected, projected.shape[:-1] + (self.mp_num, -1))
        mp_split = maybe_shard(mp_split, P("dp", None, "mp", None))

        local_dim = self.d_head * self.n_head // self.mp_num

        q, v, k, ff = jnp.split(mp_split, [local_dim, local_dim * 2, local_dim * 3], axis=-1)

        q = self.head_split(q)
        v = self.head_split(v)
        k = self.head_split(k)

        return q, v, k, ff

    def output(self, *x):
        out = jnp.concatenate(x, axis=-1)
        out = maybe_shard(out, P("dp", None, "mp", None))

        out = out.reshape(x[0].shape[:-2] + (-1,))
        out_shard = maybe_shard(out, P("dp", None, "mp"))

        return self.output_proj(out_shard)

    def __call__(self, x, attn_bias):

        x = self.norm(x)

        q, v, k, ff = self.input(x)

        # head_print("x.shape", x.shape)
        # head_print("attn_bias.shape", attn_bias.shape)

        seq_len = x.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))[None, :, :]
        bias = -1e10 * (1. - causal_mask)

        # head_print("bias.shape", bias.shape)

        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = self.glu(ff)

        return self.output(attn_out, ff_out)

    # [batch, seq, mp, dim*2//mp]
    def glu(self, x):
        out, gate = jnp.split(x, 2, axis=-1)

        return out * jax.nn.gelu(gate)

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x, attn_bias):
        x = self.norm(x)

        assert x.shape[0] == 1

        q, v, k, ff = self.input(x)

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=1)[1:]
        k = jnp.concatenate((decode_state["k"], k), axis=1)[1:]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        length = v.shape[1]

        masked_tokens = length - tokens_decoded

        attention_mask = jnp.arange(0, length) < masked_tokens
        bias = (-1e10 * attention_mask)
        bias += attn_bias

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = self.glu(ff)

        return self.output(attn_out, ff_out), {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length, attn_bias):
        x = self.norm(x)

        q, v, k, ff = self.input(x)

        full_length = x.shape[1]
        masked_tokens = full_length - given_length

        causal_mask = np.tril(np.ones((full_length, full_length)))

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        bias -= 1e10 * (jnp.arange(0, full_length) < masked_tokens)  # mask out zero tokens before context starts
        bias += attn_bias  # finally add attn bias for rpe

        attn_out = self.self_attn(q, v, k, bias)
        ff_out = self.glu(ff)

        return self.output(attn_out, ff_out), {
            "tokens_decoded": given_length.astype(jnp.uint32),
            "k": k,
            "v": v,
        }

def set_background_tokens(targets):
    """
    BJ:
        Want to do this:
            targets[...,0:BACKGROUND_LEN-1] = MASKED_TOKENS_START
        But JAX doesn't allow assignment, so:
    """
    y = np.ones(targets.shape[-1])
    y[0:BACKGROUND_LEN-1] = 0.0
    z = np.zeros(targets.shape[-1])
    z[0:BACKGROUND_LEN-1] = MASKED_TOKENS_START
    targets = targets * y + z
    return jax.lax.stop_gradient(targets)

class ProjectionShard(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        out_dim = config["n_vocab"]
        shards = config["cores_per_replica"]
        norm = getnorm(config["norm"])

        assert out_dim % shards == 0

        self.dim = out_dim
        self.dim_per_shard = out_dim // shards

        self.norm = norm

        self.proj = hk.Linear(self.dim_per_shard)

    def __call__(self, x):
        x = self.norm(x)
        proj = self.proj(x)

        all_proj = jax.lax.all_gather(proj, 'shard')
        # BJ: --> all_proj dims [#shard, t, dim_per_shard]

        return hk.Flatten()(jnp.transpose(all_proj, (1, 0, 2)))
        # BJ: --> output dims [t, #shard*dim_per_shard]

    def loss(self, x, targets, z_loss=1):
        """
        BJ:
        Args:
            x: (nsamples, T, dim)
            targets: (nsamples, T)
        """
        x = f_psum(x)
        x = self.norm(x)
        logits = self.proj(x)
        # BJ logits: (nsamples, T, vocab/nshards)
        # if DO_BACKGROUND:
        #     # Ignore background outputs in the loss.
        #     print(f"x.shape: {x.shape}")
        #     print(f"logits.shape: {logits.shape}")
        #     print(f"targets.shape: {targets.shape}")
        #     # Want to do this:
        #     #     targets[...,0:BACKGROUND_LEN] = MASKED_TOKENS_START
        #     # But JAX doesn't allow assignment, so:
        #     targets = set_background_tokens(targets)
        #
        #     # OLD VERSION: Only calculate loss on the predict tokens (not background)
        #     # T_predict = targets.shape[-1]
        #     # logits = logits[:,:,-T_predict:,:]

        # BJ: targets are over global # output, but each shard has its local chunk
        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        # BJ: stable_softmax subtracts max for numerical stability:
        #                          exp(logit_pred)           exp(logit_pred - C)
        #      softmax(logits) = --------------------  =  --------------------------
        #                         \sum_j exp(logit_j)      \sum_j exp(logit_j - C)
        #
        #      -log(softmax(logits)) = log(\sum_j exp(logit_j - C)) - (logit_pred - C)
        #
        #  But subtracting max requires us to use stop_gradient().
        #     See example in https://www.tensorflow.org/api_docs/python/tf/stop_gradient:
        #     "if the max values are not unique then the gradient could flow to the wrong
        #     input".  --> treat as a constant.
        global_max = jax.lax.pmax(jax.lax.stop_gradient(logits.max(-1, keepdims=True)), "shard")
        # BJ: global_max (nsamples, T, 1)
        # BJ:

        logits -= jax.lax.stop_gradient(global_max)
        # BJ: logits (nsamples, T, vocab/nshards)

        # BJ jax.nn.one_hot gives all zeros for values < 0 and >= self.dim_per_shard.
        gt_onehot = jax.nn.one_hot(targets - shard_start_index, self.dim_per_shard)
        # BJ the logits of the target indices:
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        # BJ predicted_logits (nsamples, T)
        # BJ For each timestep there is a predicted_logits value on each shard. Sum:
        predicted_logits = g_psum(predicted_logits)
        # BJ predicted_logits (nsamples, T)
        print(f"predicted_logits: {predicted_logits.shape}")

        exp_logits = jnp.exp(logits)
        # BJ exp_logits (nsamples, T, vocab/nshards)

        sum_exp_logits = exp_logits.sum(axis=-1)
        # BJ sum_exp_logits (nsamples, T)
        sum_exp_logits = g_psum(sum_exp_logits)
        # BJ sum_exp_logits (nsamples, T)
        print(f"sum_exp_logits: {sum_exp_logits.shape}")

        # See derivation above: -log(softmax(logits)) = log(\sum_j exp(logit_j - C)) - (logit_pred - C)
        loss = jnp.log(sum_exp_logits) - predicted_logits
        # BJ loss (nsamples, T)

        # From mesh_tensorflow:
        #   https://github.com/tensorflow/mesh/blob/acf624736647a7914529b97c38cdd43f14969f0b/mesh_tensorflow/layers.py#L1111
        #   if z_loss is nonzero, we add a loss equal to z_loss*log(z)^2, where z is the
        #   partition function.  Example value: z_loss=1e-4.  Two uses of z_loss are:
        #   - To keep the logits from drifting too far from zero, which can cause
        #     unacceptable roundoff errors in bfloat16.
        #   - To encourage the logits to be normalized log-probabilities.
        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()

        # BJ: moved here after 4 twitter16 runs, 4 twitter15_64 runs (TODO remove comment)
        if DO_BACKGROUND:
            # Mask: no gradient on pad or style_vecs
            mask_background = np.ones(targets.shape[-1])
            mask_background[0:BACKGROUND_LEN-1] = 0
            loss *= mask_background
            # TODO remove comment: https://theaisummer.com/jax-transformer/
            mask_special = jnp.less(targets, MASKED_TOKENS_START)
            loss *= mask_special
            if TFRECORDS_FORMAT == "REPLIES":
                mask_replies = mask_exclude_between_markers(targets,
                    REPLY_START_TOKEN_ID, REPLY_END_TOKEN_ID)
                loss *= mask_replies

        correct = (0.0 == predicted_logits)

        return loss, correct


# Used by CausalTransformerV2
class Projection(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        out_dim = config["n_vocab"]

        self.dim = out_dim
        self.norm = hk.LayerNorm(-1, True, True)

        self.proj = hk.Linear(self.dim)

    def __call__(self, x):
        x = self.norm(x)
        return self.proj(x)

    def loss(self, x, targets, z_loss=1):
        x = self.norm(x)
        logits = self.proj(x)

        logits -= logits.max(-1, keepdims=True)

        gt_onehot = jax.nn.one_hot(targets, self.dim)
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        exp_logits = jnp.exp(logits)

        sum_exp_logits = exp_logits.sum(axis=-1)

        loss = jnp.log(sum_exp_logits) - predicted_logits

        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()
        correct = (0.0 == predicted_logits)
        return loss, correct
