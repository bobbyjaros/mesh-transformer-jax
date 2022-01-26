
from tfrecord_loader import TFRecordNewInputs
import time

# for index_path in ["data/deleteme.index", "data/twitter9.train.index"]:
# for index_path in ["data/twitter9_LOCAL.train.index"]:
# for index_path in ["data/twitter14_LOCAL.train.index", "data/twitter14_LOCAL.val.index"]:
# for index_path in ["data/twitter16_LOCAL.train.index", "data/twitter16_LOCAL.val.index"]:
# for index_path in ["data/twitter16_LOCAL.val.index"]:
# for index_path in ["data/twitter18_64_LOCAL.val.index"]:
# for index_path in ["data/twitter21_64_LOCAL.train.index"]:
for index_path in ["data/twitter25_64_LOCAL.train.index"]:
    print(index_path)
    t0 = time.time()
    train_dataset = TFRecordNewInputs(index_path,
                                    batch_size=(
                                        256,
                                        2),
                                    sample_size=4096,
                                    restore_state=None)
    # This will produce samples of (256,1,2049)
    t1 = time.time()
    print(f"init time: {t1-t0} sec")
    samples = train_dataset.get_samples()
    for i,sample in enumerate(train_dataset.sample_once()):
        # print(sample.shape)
        # if i == 800:
        #     break
        if sample.shape != (256, 2, 2049):
            print(f"sample [{i}] is {sample.shape}")
            print([(i,z) for (i,z) in enumerate([x.shape[0] for [x,y] in sample]) if z != 2049])
            print([(i,z) for (i,z) in enumerate([y.shape[0] for [x,y] in sample]) if z != 2049])
        break
    break

# Sample once:
# sample = next(train_dataset.sample_once())
# import numpy as np
# x = np.array(sample[0:16,0,:])
# print(x.shape)


# TOKENIZER:
from mesh_transformer.layers import STYLE_VECS_LEN, TFRECORDS_FORMAT
import transformers
tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
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
tokenizer.decode(sample[0,0,1024+STYLE_VECS_LEN-4:1024+STYLE_VECS_LEN+40])

for i in range(3):
    for o in samples[i, 0, :]:
        print(tokenizer.decode(int(o)), end="", flush=True)
    x = input("Press enter: ")
