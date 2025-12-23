# Training hyperparameters
max_iters = 1_000_000
lr_decay_iters = 1_000_000
min_lr = 1e-5
learning_rate = 1e-4
gradient_accumulation_steps = 1
batch_sizes = [18, 2]
train_data_files = ['dataset/mazes', 'dataset/random']
eval_interval = 1000
ckpt_every_n_eval = 100

# Architecture
architecture = "gnn"

# GNN Model Configuration
block_size: int = 128
vocab_size: int = 97
field_of_view_size: int = 121

# GNN-specific architecture parameters
n_gnn_layers: int = 3  # Number of GNN layers in encoder
gnn_heads: int = 3  # Number of attention heads in graph transformer
n_resnet_blocks: int = 3  # Number of ResNet blocks in spatial encoder

# Shared model parameters
n_embd: int = 64*3
latent_embd: int = 32*3
action_msg_feats: int = 32*3
n_comm_rounds: int = 3
num_action_heads: int = 3
loss_weights = (4, 2, 1)  # weights for 3 action heads: [a0, a1, a2]

# Optional parameters (with defaults)
dropout: float = 0.0
bias: bool = False
empty_token_code: int = 0
empty_connection_code: int = -1