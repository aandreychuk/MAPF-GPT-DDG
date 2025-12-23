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

# Input Configuration
block_size = 128
vocab_size = 97
field_of_view_size = 121

# GNN architecture parameters
n_gnn_layers = 3          # Number of GNN layers in encoder
gnn_heads = 4             # Number of attention heads in graph transformer
n_resnet_blocks = 2       # Number of ResNet blocks in spatial encoder

# Model dimensions
n_embd = 64 * 3           # Token/FOV embedding dimension
latent_embd = 32 * 4      # Latent node embedding dimension
action_msg_feats = 32 * 4 # Features for action prediction

# Communication and action prediction
n_comm_rounds = 1         # Message passing rounds in decoder
num_action_heads = 2      # Number of future actions to predict
loss_weights = (3, 1)  # Loss weights for each action head [a0, a1, a2]