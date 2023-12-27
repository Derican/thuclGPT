"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import sentencepiece as spm
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
init_from = 'resume'
out_dir = 'out'  # ignored if init_from is not 'resume'
# or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
start = "FILE:lyric_test.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 20  # number of tokens generated in each sample
# 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
temperature = 0.8
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported(
) else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
# overrides from command line or config file
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
# for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
# older checkpoints might not have these...
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join(
        'data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    if "vocab_type" not in meta or meta["vocab_type"] == "classic":
        stoi, itos = meta['stoi'], meta['itos']
        def encode(s): return [stoi[c] for c in s if c in stoi]
        def decode(l): return ''.join([itos[i] for i in l])
    elif meta["vocab_type"] == "sp":
        model_type = meta["model_type"]
        vocab_size = meta["vocab_size"]
        sp_path = os.path.join(
            'data', checkpoint['config']['dataset'], f'{model_type}_{vocab_size}.model')
        sp = spm.SentencePieceProcessor(
            model_file=sp_path)
        def encode(s): return sp.Encode(s)
        def decode(l): return ''.join(sp.Decode(l))

else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = AutoTokenizer.from_pretrained(
        "uer/gpt2-xlarge-chinese-cluecorpussmall")
    def encode(s): return enc.encode(s)
    def decode(l): return enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
perp = []
losses = []
start_encodings = encode(start)
test_len = len(start_encodings)
block_size = gptconf.block_size
for begin_loc in tqdm(range(0, test_len, block_size)):
    end_loc = min(begin_loc + block_size, test_len)
    start_ids = start_encodings[begin_loc:end_loc]
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -1

    # run generation
    with torch.no_grad():
        with ctx:
            logits, loss = model.forward(x, targets=y)
            losses.append(loss.item())
            ppl = torch.exp(loss).item()
            perp.append(ppl)

print(f"Loss: {np.mean(losses)} ({np.std(losses)})")
print(f"Perplexity: {np.mean(perp)} ({np.std(perp)})")
