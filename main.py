import multiprocessing
from pathlib import Path

# Ensure the 'spawn' start method is used
multiprocessing.set_start_method("spawn", force=True)


from gpt.dmm_comm import DMM, DMMConfig
from gpt.dmm_gnn import DMMGNN, DMMGNNConfig
from loguru import logger
from gpt.aggregated_data_loader import AggregatedMapfArrowDataset
import math
import os
import time
from contextlib import nullcontext
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
train_dir = "train_dir"
exp_dir_name = "trial"
exp_dir = Path(train_dir) / exp_dir_name

eval_interval = 500
ckpt_every_n_eval = 10
log_interval = 1
eval_iters = 40
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

gradient_accumulation_steps = 16  # used to simulate larger batch sizes
batch_size = 8  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_encoder_layer = 2
n_decoder_layer = 2

n_head = 2
n_embd = 80
latent_embd = 40
latent_tok_n = 32
action_msg_feats = 80
num_action_heads = 3
loss_weights = (4, 2, 1)

empty_connection_code = -1
n_comm_rounds = 2

field_of_view_size = 11*11
max_num_neighbors = 13

dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 30000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 30000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = ("cuda")  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
state_size = 32
empty_token_code = 0
meta_vocab_size = 97
if 'cuda' in device and not torch.cuda.is_available():
    device = 'cpu'
    logger.warning(f'Cuda is not available, switching to {device}')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster

# Architecture selection
architecture = "transformer"  # 'transformer' or 'gnn'
# -----------------------------------------------------------------------------

current_train_index = 0
current_valid_index = 0

train_data_files = ['dataset/mazes', 'dataset/random']
batch_sizes = [36, 4]

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# Ensure architecture is in config
if "architecture" not in config_keys:
    config_keys.append("architecture")
exec(open("gpt/configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

train_data = AggregatedMapfArrowDataset(train_data_files, device=device, batch_sizes=batch_sizes)

# train_data = MapfArrowDataset(train_data_file, device=device, batch_size=batch_size // state_size)
# val_data = MapfArrowDataset(valid_data_file, device=device, batch_size=batch_size // state_size)

# train_data = AggregatedMapfArrowDataset(train_data_files, device=device, batch_sizes=[batch_size * 9 // 10, batch_size - batch_size * 9 // 10])
train_data_iter = iter(train_data)

def calculate_epochs(max_iters, dataset_size, batch_size, gradient_accumulation_steps=1):
    # Effective batch size considering gradient accumulation
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    num_epochs = max_iters / steps_per_epoch

    return num_epochs


def human_readable_size(size):
    for unit in ["pairs", "K pairs", "M pairs", "B pairs"]:
        if size < 1000:
            return f"{size:.2f} {unit}"
        size /= 1000
    return f"{size:.2f} B pairs"


logger.info(f"Train set size: {human_readable_size(train_data.get_full_dataset_size())}")
# logger.info(f"Validation set size: {human_readable_size(val_data.get_full_dataset_size())}")

num_epochs = calculate_epochs(max_iters, train_data.get_full_dataset_size(), batch_size, gradient_accumulation_steps)

logger.info(f"Number of training epochs: {num_epochs:.2f}")

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    if exp_dir.exists() and init_from == "scratch":
        # Create a new exp_dir with an incremented suffix
        i = 1
        while (Path(train_dir) / f"{exp_dir_name}_{str(i).zfill(2)}").exists():
            i += 1
        exp_dir = Path(train_dir) / f"{exp_dir_name}_{str(i).zfill(2)}"
        logger.warning(f"Existing experiment directory found. Starting fresh in new directory: {exp_dir}")


    os.makedirs(exp_dir, exist_ok=True)

    log_file = exp_dir / "log.txt"
    logger.add(log_file, rotation="100 MB", enqueue=True)

    tb_logdir = exp_dir / "tb"
    tb_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_logdir))


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

def get_batch(data):
    observations, actions, agent_chat_ids, agents_rel_coords = next(data)
    # observations: [B, agents, seq_len] - FOV observations
    # actions: [B, agents, horizon] - target actions
    # agent_chat_ids: [B, agents, max_neighbors] - agent IDs for message routing
    # agents_rel_coords: [B, agents, max_neighbors*2] - relative coordinates (dx,dy pairs)
    return (observations.to(torch.int), 
            agent_chat_ids.to(torch.int), 
            actions.to(torch.long),
            agents_rel_coords.to(torch.long))

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    n_encoder_layer = n_encoder_layer,
    n_decoder_layer = n_decoder_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,

    action_msg_feats = action_msg_feats,
    empty_connection_code = empty_connection_code,
    n_comm_rounds = n_comm_rounds,
    latent_embd = latent_embd,
    latent_tok_n = latent_tok_n,
    field_of_view_size = field_of_view_size,
    max_num_neighbors = max_num_neighbors,
    empty_token_code = empty_token_code,
    num_action_heads = num_action_heads,
    loss_weights = loss_weights,
)  # start with model_args from command line

# Determine which architecture to use
use_gnn = architecture.lower() == "gnn"
if use_gnn:
    logger.info("Using GNN-based architecture (DMMGNN)")
else:
    logger.info("Using Transformer-based architecture (DMM)")

if init_from == "scratch":
    # init a new model from scratch
    logger.info("Initializing a new model from scratch")
    model_args["vocab_size"] = meta_vocab_size
    
    if use_gnn:
        # Convert DMM args to GNN args
        gnn_args = {
            "block_size": model_args.get("block_size", 128),
            "vocab_size": meta_vocab_size,
            "field_of_view_size": model_args.get("field_of_view_size", 121),
            "max_num_neighbors": model_args.get("max_num_neighbors", 13),
            "n_gnn_layers": model_args.get("n_encoder_layer", 2),  # Use encoder layers as GNN layers
            "gnn_heads": model_args.get("n_head", 2),
            "n_embd": model_args.get("n_embd", 16),
            "latent_embd": model_args.get("latent_embd", 8),
            "dropout": model_args.get("dropout", 0.0),
            "bias": model_args.get("bias", False),
            "empty_token_code": model_args.get("empty_token_code", 0),
            "action_msg_feats": model_args.get("action_msg_feats", 16),
            "empty_connection_code": model_args.get("empty_connection_code", -1),
            "n_comm_rounds": model_args.get("n_comm_rounds", 2),
            "num_action_heads": model_args.get("num_action_heads", 1),
            "loss_weights": model_args.get("loss_weights", None),
            "n_resnet_blocks": model_args.get("n_resnet_blocks", 2),  # ResNet blocks in spatial encoder
        }
        gptconf = DMMGNNConfig(**gnn_args)
        model = DMMGNN(gptconf)
    else:
        gptconf = DMMConfig(**model_args)
        model = DMM(gptconf)
        
elif init_from == "resume":
    logger.info(f"Resuming training from {exp_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(exp_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    
    # Check if checkpoint uses GNN or transformer
    checkpoint_arch = checkpoint_model_args.get("architecture", "transformer")
    use_checkpoint_gnn = checkpoint_arch.lower() == "gnn"
    
    if use_checkpoint_gnn != use_gnn:
        logger.warning(f"Checkpoint architecture ({checkpoint_arch}) differs from requested ({architecture}). Using checkpoint architecture.")
        use_gnn = use_checkpoint_gnn
    
    # force these config attributes to be equal otherwise we can't even resume training
    if use_gnn:
        for k in ["action_msg_feats", "empty_connection_code", "n_comm_rounds",
            "latent_embd", "field_of_view_size", "max_num_neighbors", "empty_token_code",
            "n_embd", "block_size", "bias", "vocab_size", "n_gnn_layers", "gnn_heads", "n_resnet_blocks"]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        # Convert to GNN args
        gnn_args = {
            "block_size": model_args.get("block_size", 128),
            "vocab_size": model_args.get("vocab_size", 97),
            "field_of_view_size": model_args.get("field_of_view_size", 121),
            "max_num_neighbors": model_args.get("max_num_neighbors", 13),
            "n_gnn_layers": model_args.get("n_gnn_layers", model_args.get("n_encoder_layer", 2)),
            "gnn_heads": model_args.get("gnn_heads", model_args.get("n_head", 2)),
            "n_embd": model_args.get("n_embd", 16),
            "latent_embd": model_args.get("latent_embd", 8),
            "dropout": model_args.get("dropout", 0.0),
            "bias": model_args.get("bias", False),
            "empty_token_code": model_args.get("empty_token_code", 0),
            "action_msg_feats": model_args.get("action_msg_feats", 16),
            "empty_connection_code": model_args.get("empty_connection_code", -1),
            "n_comm_rounds": model_args.get("n_comm_rounds", 2),
            "num_action_heads": model_args.get("num_action_heads", 1),
            "loss_weights": model_args.get("loss_weights", None),
            "n_resnet_blocks": model_args.get("n_resnet_blocks", model_args.get("n_resnet_blocks", 2)),
        }
        gptconf = DMMGNNConfig(**gnn_args)
        model = DMMGNN(gptconf)
    else:
        for k in ["n_encoder_layer", "n_decoder_layer",
            "action_msg_feats", "empty_connection_code", "n_comm_rounds",
            "latent_embd", "latent_tok_n", "field_of_view_size",
            "max_num_neighbors", "empty_token_code",
            "n_head", "n_embd", "block_size", "bias",
            "vocab_size"]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        gptconf = DMMConfig(**model_args)
        model = DMM(gptconf)
    
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)  # strict=False for architecture differences
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

logger.info("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))


model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

if compile and 'cuda' in device:
    logger.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    try:
        model = torch.compile(model)
    except AttributeError:
        logger.warning('torch compile(model) requires PyTorch >= 2.0')

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", ]:
    # for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                obs, agent_ids, actions, rel_coords = get_batch(train_data_iter)
            else:
                raise KeyError
            with ctx:
                # DMM forward: (observations, agent_chat_ids, target_actions, agents_rel_coords)
                logits, loss = model(obs, agent_ids, actions, rel_coords)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



# training loop
obs, agent_ids, actions, rel_coords = get_batch(train_data_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        losses = estimate_loss()
        model.train()
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}")

        writer.add_scalar("Loss/train", losses["train"], iter_num)
        writer.add_scalar("Learning Rate", lr, iter_num)


        if always_save_checkpoint:
            if iter_num > 0 and (iter_num // eval_interval % ckpt_every_n_eval == 0):
                # Add architecture to model_args for checkpoint
                checkpoint_model_args = model_args.copy()
                checkpoint_model_args["architecture"] = architecture
                
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": checkpoint_model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                logger.info(f"saving checkpoint to {exp_dir}")
                torch.save(checkpoint, os.path.join(exp_dir, f"ckpt_{iter_num}.pt"))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            # DMM forward: (observations, agent_chat_ids, target_actions, agents_rel_coords)
            logits, loss = model(obs, agent_ids, actions, rel_coords)
            loss = (
                    loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        obs, agent_ids, actions, rel_coords = get_batch(train_data_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.monotonic()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5:  # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms,  {iter_num} / {max_iters}")
        writer.add_scalar("Loss/step", lossf, iter_num)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if master_process:
    writer.close()

if ddp:
    destroy_process_group()
