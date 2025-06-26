compile = True
max_iters = 30000
lr_decay_iters = 30000

batch_size = 4096
n_layer = 5
n_head = 5
n_embd = 160

block_size = 256
gradient_accumulation_steps = 16

# init_from = 'resume'

#DDG settings
dagger_type = "ddg"
device_id = 0 # i.e. MAPF-GPT during data collection is run on device cuda:0
num_workers = 8 # number of workers during DDG
file_size = 50 * 2 ** 11 # number of observation-action pairs collected by each worker during single DDG iteration
max_ratio = 0.25 # maximum ratio of DDG data in training data
train_data_files = ["dataset/train", f"dataset/{dagger_type}"] # to avoid overfitting on DDG data, we use both 1B MAPF-GPT dataset and DDG data for training
valid_data_file = "dataset/validation"