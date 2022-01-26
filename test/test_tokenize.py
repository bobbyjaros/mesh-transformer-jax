from transformers import GPT2TokenizerFast
from create_finetune_tfrecords import prep_and_tokenize_generator

len_style_vectors = 64
encoder = GPT2TokenizerFast.from_pretrained('gpt2')
encoder.add_special_tokens({
  'pad_token': '<|pad|>',
  'additional_special_tokens': [f'style{i}' for i in range(len_style_vectors)],
})

# tweet_text1 = "BarackObama: Happy 97th birthday, President Carter!\n\nWe love you!" # <|endoftext|>"
tweet_text1 = "Carter!\n\nWe love you!\n" # <|endoftext|>"
string_iterable = [tweet_text1]
token_gen = prep_and_tokenize_generator(string_iterable,
                                        encoder,
                                        normalize_with_ftfy=True,
                                        normalize_with_wikitext_detokenize=False
)
token_list = [x for x in token_gen][0]
print(token_list)