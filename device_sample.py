import argparse
import json
import time
import os

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
from mesh_transformer.layers import DO_BACKGROUND, BACKGROUND_LEN, STYLE_VECS_LEN, TFRECORDS_FORMAT
from tfrecord_loader import TFRecordNewInputs
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

## Utils:
def pprint(output):
    for j in range(16):
        print("#"*60)
        print(f"\nsample {j}:")
        for o in output[1][0][j, :, 0]:
            print(tokenizer.decode(o), end="", flush=True)

def print_and_save(f, text, newline = True):
    if newline:
        print(text)
        f.write(text + "\n")
    else:
        print(text, end="", flush=True)
        f.write(text)
    return

def savename():
    sample_save_dir = "samples"
    # if user_text is None:
    #     text_clean = "_"
    # else:
    #     text_clean = "-".join()
    # os.makedirs("{sample_save_dir}/{model_dir}", exist_ok=True)
    # fname = f"{sample_save_dir}/{model_dir}/{text_clean}-{np.random.randint(0,1000)}.txt"
    os.makedirs(sample_save_dir, exist_ok=True)
    fname = f"{sample_save_dir}/{model_dir}-{np.random.randint(0,1000)}.txt"
    return fname

"""Given a list of tokens comprising a sample, determine the username this sample corresponds to.

TODO: this give the first username, which is usually the correct one -- except if the first tweet is a response to
      someone else. Should be the most common username.
"""
def username_from_token_list(token_list, tokenizer):
    first_nonpad_i = np.where(token_list != 50257)[0][0]
    first_colon_i = np.where(token_list == 25)[0][0]
    username = ''.join([tokenizer.decode(o) for o in token_list[first_nonpad_i:first_colon_i]])
    return username

"""
To run as:
with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
    gen("How are you?")
"""
def gen_from_background(background_tokens, tokenizer, style_tokens, user_text = None):
    fname = savename()
    with open(fname, "w") as f:
        for j in range(16):
            start = time.time()
            this_background_tokens = background_tokens[j, :]
            username = username_from_token_list(this_background_tokens, tokenizer)
            print_and_save(f, "#"*60)
            print_and_save(f, f"username: {username}")
            # Display background:
            for o in this_background_tokens:
                print_and_save(f, tokenizer.decode(o), newline=False)
            print_and_save(f, "*"*30)
            if TFRECORDS_FORMAT == "KEYWORDS":
                full_user_text = f"{START_KEYWORDS_TOKEN}{user_text}{END_KEYWORDS_TOKEN}{username}: "
            else: # "REPLIES"
                if user_text and len(user_text) > 0:
                    if TFRECORDS_FORMAT == "REPLIES":
                        full_user_text = f"{REPLY_START_TOKEN}bobby: {user_text}\n\n{REPLY_END_TOKEN}{username}: "
                    else:
                        full_user_text = f"bobby: {user_text}\n\n{username}: "
                else:
                    full_user_text = f"{username}: "
            print_and_save(f, full_user_text)
            user_tokens = tokenizer.encode(full_user_text)
            style_token_ids = tokenizer.encode(''.join(style_tokens))
            # |-----------backgroundj-----------|--style--|-------predict---------|
            tokens = np.hstack([this_background_tokens, style_token_ids, user_tokens])
            tlen = tokens.shape[-1]
            tokens = tokens.reshape((1,tlen))
            length = np.ones(1) * tlen
            print(f"BJ gen: tokens.shape {tokens.shape}")
            gen_length = 256
            output = network.generate(tokens, length, gen_length,
                {"top_p": np.ones(1) * 0.9,
                "temp": np.ones(1) * 0.75})
            (final_state, outputs) = output
            print(f"outputs[1].shape: {outputs[1].shape}")
            end_of_text = np.where(outputs[0][0,:,0] == 50256)[0]
            if len(end_of_text) > 0:
                first_end_of_text = end_of_text[0]
            else:
                first_end_of_text = outputs[0].shape[1]
            mean_log_prob = outputs[1][0, 0:first_end_of_text, 0].mean()
            print(f"log_prob mean: {mean_log_prob}")
            for o in outputs[0][0, :, 0]:
                print_and_save(f, tokenizer.decode(o), newline=False)
            # for logp in outputs[1]
            print_and_save(f, f"\ncompletion done in {time.time() - start:06}s")

"""Generates from background, compute a whole batch at once.
"""
def gen_from_background_OLD(context):
    tokens = tokenizer.encode(context)

    start = time.time()

    # |--style--|-------predict---------|
    tokens = np.append(style_tokens,tokens)
    batched_tokens = np.array([tokens] * total_batch)
    # |-----------background0-----------|--style--|-------predict---------|
    # |-----------background1-----------|--style--|-------predict---------|
    # |-----------background2-----------|--style--|-------predict---------|
    batched_tokens = np.hstack([background_tokens, batched_tokens])
    length = np.ones(total_batch, dtype=np.uint32) * batched_tokens.shape[-1]

    print(f"BJ gen_batch: batched_tokens.shape {batched_tokens.shape}")
    # signature: CausalTransformer.generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
    gen_length = 256
    sampler_options = {
        "top_p": np.ones(total_batch) * 0.9,
        "temp": np.ones(total_batch) * 0.75
    }
    output = network.generate(batched_tokens, length, gen_length, sampler_options)

    print(f"completion done in {time.time() - start:06}s")
    return output

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint gs://{bucket}/{model_dir}/step_{ckpt_step}/")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica

    # BJ: Load background tokens (i.e. previous tweets for a user, to set context)
    if DO_BACKGROUND:
        total_batch = 16 # TODO
        val_set = list(params["val_set"].values())[0]
        index_path = f"data/{val_set}"
        train_dataset = TFRecordNewInputs(index_path,
                                        batch_size=(
                                            total_batch,
                                            1),
                                        sample_size=4096,
                                        restore_state=None)
        sample = next(train_dataset.sample_once())
        background_tokens = np.array(sample[0:total_batch,0,0:BACKGROUND_LEN])

    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        print(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        if DO_BACKGROUND:
            style_tokens = [f'<|style{i}|>' for i in range(STYLE_VECS_LEN)]
            additional_special_tokens = style_tokens
            if TFRECORDS_FORMAT == "KEYWORDS":
                START_KEYWORDS_TOKEN = "<|KEY|>"
                END_KEYWORDS_TOKEN = "<|GO|>"
                additional_special_tokens.extend([START_KEYWORDS_TOKEN, END_KEYWORDS_TOKEN])
            elif TFRECORDS_FORMAT == "REPLIES":
                REPLY_START_TOKEN = "<|reply|>"
                REPLY_END_TOKEN = "<|reply_end|>"
                additional_special_tokens.extend([REPLY_START_TOKEN, REPLY_END_TOKEN])

            tokenizer.add_special_tokens({
                'pad_token': '<|pad|>',
                'additional_special_tokens': additional_special_tokens,
            })

        while True:
            context = input("Type input:")
            tokens = tokenizer.encode(context)

            start = time.time()

            if DO_BACKGROUND:
                output = gen_from_background(background_tokens, tokenizer, style_tokens, user_text = context)
                continue

            provided_ctx = len(tokens)
            pad_amount = seq - provided_ctx

            # Right justify:
            padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
            batched_tokens = np.array([padded_tokens] * total_batch)
            length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

            # signature: CausalTransformer.generate(self, ctx, ctx_length, gen_length, sampler_options, return_logits=False):
            output = network.generate(batched_tokens, length, 512, {"top_p": np.ones(total_batch) * 0.9,
                                                                    "temp": np.ones(total_batch) * 0.75})

            for idx, o in enumerate(output[1][0][:, :, 0]):
                print(f"sample {idx}: {repr(tokenizer.decode(o))}")

            print(f"completion done in {time.time() - start:06}s")
