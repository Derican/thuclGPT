import os
import requests
import tiktoken
from transformers import AutoTokenizer
import numpy as np

input_train_file_path = os.path.join(
    os.path.dirname(__file__), 'lyric_train.txt')
input_valid_file_path = os.path.join(
    os.path.dirname(__file__), 'lyric_valid.txt')

with open(input_train_file_path, 'r') as f:
    data_train = f.read()
with open(input_valid_file_path, 'r') as f:
    data_valid = f.read()

data = data_train + '\n' + data_valid
n = len(data)
train_data = data_train
val_data = data_valid

# encode with tiktoken gpt2 bpe
enc = AutoTokenizer.from_pretrained("uer/gpt2-xlarge-chinese-cluecorpussmall")
train_ids = enc(train_data)["input_ids"]
val_ids = enc(val_data)["input_ids"]
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 68,851,733 tokens
# val has 1,720,282 tokens
