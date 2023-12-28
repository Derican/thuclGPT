import gradio as gr
import os
import json
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import sentencepiece as spm
from transformers import AutoTokenizer

max_new_tokens = 20  # number of tokens generated in each sample
# 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
temperature = 0.8
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported(
) else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster

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

MODELS = {
    "lyric_char": {
        "out_dir": "out-lyric-char_28"
    },
    "lyric_word": {
        "out_dir": "out-lyric-word_11"
    },
    "lyric_gpt2_ft": {
        "out_dir": "/root/autodl-tmp/out-lyric-ft"
    },
    "lyric_gpt2_ext": {
        "out_dir": "/root/autodl-tmp/out-lyric-ft-word"
    },
}

selected_model = None

with open("prompt.txt", "r") as f:
    pre_prompts = f.read().splitlines()

def load_model(out_dir):
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

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    load_meta = False


    # older checkpoints might not have these...
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
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
        enc = AutoTokenizer.from_pretrained(
            "models/gpt2-xlarge-chinese-cluecorpussmall")
        def encode(s): return enc.encode(s, allowed_special={"<|endoftext|>"})
        def decode(l): return enc.decode(l)
    
    return model, encode, decode


def predict(input, chatbot, max_new_tokens, top_k, temperature):
    if selected_model:
        out_dir = MODELS[selected_model]["out_dir"]
    else:
        gr.Warning("Please select model first!")
        return
    chatbot.append((input, ""))
    model, encode, decode = load_model(out_dir)
    input_ids = encode(input + '\n')
    x = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens,
                               temperature=temperature, top_k=top_k)
            response = decode(y[0].tolist())
            response = response[len(input):]
            chatbot[-1] = (input, response)
    
    return chatbot

def set_model(model_dropdown):
    global selected_model
    selected_model = model_dropdown

def set_user_input(user_input, pre_prompt):
    user_input = pre_prompt
    return user_input


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">thuclGPT</h1>
            <h3 align="center">教研院 蒋建骁 罗裕佳</h3>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                model_dropdown = gr.Dropdown(MODELS.keys(),
                                             label="Selected Model")
            with gr.Row():
                pre_prompt_dropdown = gr.Dropdown(pre_prompts,
                                                  label="Default Prompts")
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False,
                                        placeholder="Input...",
                                        lines=10,
                                        container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(0,
                                   100,
                                   value=20,
                                   step=1.0,
                                   label="Maximum new tokens",
                                   interactive=True)
            top_k = gr.Slider(0,
                              200,
                              value=200,
                              label="Top K",
                              interactive=True)
            temperature = gr.Slider(0,
                                    1,
                                    value=0.8,
                                    step=0.01,
                                    label="Temperature",
                                    interactive=True)

    submitBtn.click(predict,
                    [user_input, chatbot, max_new_tokens, top_k, temperature],
                    [chatbot],
                    show_progress=True)

    model_dropdown.change(set_model, [model_dropdown])
    pre_prompt_dropdown.change(set_user_input,
                               [user_input, pre_prompt_dropdown],
                               [user_input])

demo.queue().launch(share=False, inbrowser=True)
