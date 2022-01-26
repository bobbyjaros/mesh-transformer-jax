
import numpy as np

seq_len = 10
background_len = 4
style_vecs_len = 3
# Original:
causal_mask2 = np.zeros((seq_len, seq_len))
l0 = background_len + style_vecs_len
l1 = seq_len - background_len
causal_mask2[0:l0, 0:l0] = np.tril(np.ones((l0, l0)))
causal_mask2[background_len:, background_len:] = np.tril(np.ones((l1, l1)))

# More efficient version:
causal_mask = np.tril(np.ones((seq_len, seq_len)))
causal_mask[background_len+style_vecs_len:,0:background_len] = 0

assert((causal_mask == causal_mask2).all())