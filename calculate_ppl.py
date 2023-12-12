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

# -----------------------------------------------------------------------------
# either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
init_from = 'resume'
out_dir = 'out'  # ignored if init_from is not 'resume'
# or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
start = "FILE:prompt.txt"
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
    if meta["vocab_type"] == "classic":
        stoi, itos = meta['stoi'], meta['itos']
        def encode(s): return [stoi[c] for c in s]
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
    enc = tiktoken.get_encoding("gpt2")
    def encode(s): return enc.encode(s, allowed_special={"<|endoftext|>"})
    def decode(l): return enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
for st in start.split('\n'):
    if len(st) <= 0:
        continue
    start_ids = encode(st + '\n')
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            # y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # print(decode(y[0].tolist()))
            # print('---------------')
            logits, loss = model.forward(x, targets=x)
            ppl = torch.exp(loss/len(x))
            print(ppl.item())
