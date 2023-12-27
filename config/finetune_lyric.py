import time

out_dir = '/root/autodl-tmp/out-lyric-ft'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'lyric-ft'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'lyric'
init_from = 'models/gpt2-xlarge-chinese-cluecorpussmall'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 160
block_size = 1024

# finetune at constant LR
learning_rate = 5e-5
decay_lr = False
