# Original creation of twitter tfrecord didn't have samples of even length (2049).
# This script adds the padding to correct that.

import os
from tfrecord_loader import TFRecordNewInputs
from transformers import GPT2TokenizerFast
from create_finetune_tfrecords import prep_and_tokenize_generator, write_tfrecord

len_total = 2049
len_style_vectors = 64
encoder = GPT2TokenizerFast.from_pretrained('gpt2')
encoder.add_special_tokens({
  'pad_token': '<|pad|>',
  'additional_special_tokens': [f'style{i}' for i in range(len_style_vectors)],
})

index_path = "data/twitter5.index"
print(index_path)
train_dataset = TFRecordNewInputs(index_path,
                                batch_size=(
                                    256,
                                    1),
                                sample_size=4096,
                                restore_state=None)
all_batches = []
for i,sample in enumerate(train_dataset.sample_once()):
  for x in sample:
    token_list = list(x[0])
    pad_tokens = [encoder.pad_token_id] * (len_total - len(token_list))
    token_list += pad_tokens
    all_batches.append(token_list)

tot_len = len(all_batches)
fp = os.path.join("../data/twitter/results", f"twitter5_{tot_len}_even.tfrecords")
write_tfrecord(all_batches, fp)
print(f"Wrote to {fp}")
