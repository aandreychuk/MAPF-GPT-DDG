max_iters = 1_000_000
lr_decay_iters = 1_000_000
min_lr = 1e-5
learning_rate = 1e-4

block_size: int = 256
vocab_size: int = 67
field_of_view_size: int = 121
agent_info_size: int = 10
max_num_neighbors: int = 13

n_encoder_layer: int = 3 #
n_decoder_layer: int = 3 #
n_head: int = 3
n_embd: int = 64*3
latent_embd: int = 32*3
latent_tok_n: int = 32
action_msg_feats: int = 32*3
n_comm_rounds: int = 2
num_action_heads: int = 3
loss_weights = (4, 2, 1)  # weights for 3 action heads: [a0, a1, a2]

block_size = 256
gradient_accumulation_steps = 1
#batch_size = 32
batch_size = 40
batch_sizes = [32, 8]
train_data_files = ['dataset/mazes', 'dataset/random']

eval_interval = 500
ckpt_every_n_eval = 100