
# thuclGPT

A simple, fast repository for training/finetuning medium-sized GPTs. It is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT), following a graduate course plan of Computing Language in Tsinghua University.

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm sentencepiece gradio
```

or

```
pip install -r requirements.txt
```

## quick start

The project mainly contains four training/finetuning models, listed as follows.

|model_id|description|
|-|-|
|out-lyric-char|character-level chinese lyric GPT|
|out-lyric-word|word-level(bpe) chinese lyric GPT|
|out-lyric-ft|character-level chinese lyric GPT finetuned from uer/gpt2-xlarge-chinese-cluecorpussmall|
|out-lyric-ft-word|word-level(bpe) embedding-extended chinese lyric GPT finetuned from uer/gpt2-xlarge-chinese-cluecorpussmall|

### data preparation

Dataset is created from a chinese lyric dataset, described as follows.

|dataset|length|
|-|-|
|train|34M|
|valid|864K|
|test|180K|

For data preparation, simply run the `prepare.py` in `data` folder. For example, for `out-lyric-char`, just run

```
python data/lyric_char/prepare.py
```

and the data bins will be ready.

### train/finetune

For training/finetuning, simply run `train.py` with the corresponding configuration python script in the `config` folder. For example, for `out-lyric-char`, just run

```
python train.py config/train_lyric_char.py
```

and feel free to customize parameters according to needs and gpu specs.

### complete/calculate perplexity

For completion, simply run `completion.py` with the corresponding foler containing the checkpoints. For example, for `out-lyric-char`, just run

```
python completion.py --out_dir=out-lyric-char
```

and feel free to customize `prompt.txt`.

For perplexity calculation, replace the code above with `calculate_ppl.py` and the mean and std of the test set will displayed.

### web demo

For web demo, simply run `python web_demo.py` and check the browser. The simple frontend powered by Gradio provides functions to choose models and chat with them.

### Results

|model|loss|val_loss|test_ppl|VRAM|
|-|-|-|-|-|
|out-lyric-char|1.60|1.92|7.71|17G|
|out-lyric-word|3.08|4.10|82.29|17G|
|out-lyric-ft|2.04|1.99|9.59|37G|
|out-lyric-ft-word|3.35|3.68|10228.98|37G|

## acknowledgements

All thuclGPT experiments are powered by GPUs on [AutoDL](https://www.autodl.com/).
