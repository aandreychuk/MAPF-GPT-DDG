import pyarrow as pa
import pyarrow.ipc as ipc
import os
import sys
from pathlib import Path
from multiprocessing import Pool, Manager
import argparse
from tqdm import tqdm
import threading
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lacam.inference import LacamInference, LacamInferenceConfig
from dataset_generator.instance_generator import make_pogema_maze_instance, make_pogema_random_instance, make_pogema_house_instance

def save_to_arrow_with_grid(gt_actions, grid, starts, targets, seeds, filepath):
    schema = pa.schema([
        ('gt_actions', pa.list_(pa.list_(pa.int8()))),
        ('grid', pa.list_(pa.list_(pa.int8()))),
        ('starts', pa.list_(pa.list_(pa.int8()))),
        ('targets', pa.list_(pa.list_(pa.int8()))),
        ('seeds', pa.int64())
    ])
    gt_actions_col = pa.array(gt_actions, type=pa.list_(pa.list_(pa.int8())))
    grid_col = pa.array(grid, type=pa.list_(pa.list_(pa.int8())))
    starts_col = pa.array(starts, type=pa.list_(pa.list_(pa.int8())))
    targets_col = pa.array(targets, type=pa.list_(pa.list_(pa.int8())))
    seeds_col = pa.array(seeds, type=pa.int64())
    table = pa.Table.from_arrays([gt_actions_col, grid_col, starts_col, targets_col, seeds_col], schema=schema)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write(table)

def get_solved_instance(env):
    expert_algo = LacamInference(LacamInferenceConfig(time_limit=10, timeouts=[10]))
    observations, *_ = env.reset()
    actions, grid, starts, targets = expert_algo.get_full_solution(observations, env.grid_config.max_episode_steps)
    return actions, grid, starts, targets

def run_worker(seeds, worker_id, instance_type, data_per_file, num_agents, dataset_path, files_per_worker, progress_dict=None, save_instances=False):
    all_gt_actions = []
    all_grids = []
    all_starts = []
    all_targets = []
    all_seeds = []
    total_generated = 0
    instances_processed = 0
    unsolved_instances = 0
    
    # Determine folder path based on instance type
    if instance_type == 'mazes':
        folder = f"{dataset_path}/mazes/num_agents_{num_agents}"
    elif instance_type == 'random':
        folder = f"{dataset_path}/random/num_agents_{num_agents}"
    elif instance_type == 'house':
        folder = f"{dataset_path}/house/num_agents_{num_agents}"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    
    # Initialize progress
    if progress_dict is not None:
        progress_dict[worker_id] = {
            'files': 0,
            'total_files': files_per_worker,
            'instances': 0,
            'unsolved_instances': 0
        }
    
    for seed in seeds:
        if instance_type == 'mazes':
            env = make_pogema_maze_instance(num_agents=num_agents, max_episode_steps=256, map_seed=seed, scenario_seed=seed, on_target='nothing')
        elif instance_type == 'random':
            env = make_pogema_random_instance(num_agents=num_agents, max_episode_steps=256, map_seed=seed, scenario_seed=seed, on_target='nothing')
        elif instance_type == 'house':
            env = make_pogema_house_instance(num_agents=num_agents, max_episode_steps=256, map_seed=seed, scenario_seed=seed, on_target='nothing')
        gt_actions, grid, starts, targets = get_solved_instance(env)
        instances_processed += 1
        
        if gt_actions is None:
            unsolved_instances += 1
            continue
        all_gt_actions.append(gt_actions)
        all_grids.append(grid)
        all_starts.append(starts)
        all_targets.append(targets)
        all_seeds.append(seed)
        if len(all_gt_actions) >= data_per_file:
            save_to_arrow_with_grid(all_gt_actions[:data_per_file], all_grids[:data_per_file], all_starts[:data_per_file], all_targets[:data_per_file], all_seeds[:data_per_file], f"{folder}/part_{worker_id}_{total_generated}.arrow")
            all_gt_actions = all_gt_actions[data_per_file:]
            all_grids = all_grids[data_per_file:]
            all_starts = all_starts[data_per_file:]
            all_targets = all_targets[data_per_file:]
            all_seeds = all_seeds[data_per_file:]
            total_generated += 1
            
            # Update progress based on files generated
            if progress_dict is not None:
                progress_dict[worker_id] = {
                    'files': total_generated,
                    'total_files': files_per_worker,
                    'instances': instances_processed,
                    'unsolved_instances': unsolved_instances
                }

        if total_generated >= files_per_worker:
            break
    
    # Mark as completed
    if progress_dict is not None:
        progress_dict[worker_id] = {
            'files': total_generated,
            'total_files': files_per_worker,
            'instances': instances_processed,
            'unsolved_instances': unsolved_instances,
            'completed': True
        }
    
    return total_generated, instances_processed

def run_worker_wrapper(args_tuple):
    """Wrapper function to unpack arguments for imap"""
    return run_worker(*args_tuple)

def progress_monitor(progress_dict, num_workers, bars):
    """Monitor progress and update tqdm bars"""
    while True:
        all_completed = True
        for worker_id in range(num_workers):
            if worker_id in progress_dict:
                progress = progress_dict[worker_id]
                bars[worker_id].n = progress['files']
                bars[worker_id].set_postfix({
                    'instances': progress['instances'],
                    'unsolved': progress['unsolved_instances']
                })
                bars[worker_id].refresh()
                if not progress.get('completed', False):
                    all_completed = False
            else:
                # Worker hasn't started yet
                all_completed = False
        
        if all_completed and len(progress_dict) == num_workers:
            break
        time.sleep(0.1)  # Update every 100ms
    
    # Final update
    for worker_id in range(num_workers):
        if worker_id in progress_dict:
            progress = progress_dict[worker_id]
            bars[worker_id].n = progress['files']
            bars[worker_id].refresh()

def main():
    parser = argparse.ArgumentParser(description="Collect MAPF data.")
    parser.add_argument('--instance_type', choices=['mazes', 'random', 'house'], default='mazes', help='Type of instance to generate: mazes or random')
    parser.add_argument('--data_per_file', type=int, default=2**10, help='Number of data samples per file')
    parser.add_argument('--num_agents', type=int, default=32, help='Number of agents')
    parser.add_argument('--path', type=str, default='gt_data_mapf', help='Path to save the gt data')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--total_files', type=int, default=512, help='Total number of files to generate')
    args = parser.parse_args()

    files_per_worker = args.total_files // args.workers
    seeds = [i for i in range(128, 1000000*args.workers)]
    seeds_per_worker = len(seeds) // args.workers
    
    # Create shared progress dictionary
    manager = Manager()
    progress_dict = manager.dict()
    
    # Prepare worker arguments as tuples
    worker_args = [
        (seeds[i * seeds_per_worker:(i + 1) * seeds_per_worker], i, args.instance_type, args.data_per_file, args.num_agents, args.dataset_path, files_per_worker, progress_dict)
        for i in range(args.workers)
    ]
    
    # Create progress bars for each worker
    bars = []
    for worker_id in range(args.workers):
        bar = tqdm(
            total=files_per_worker,
            desc=f"Worker {worker_id}",
            position=worker_id,
            leave=True,
            unit="file"
        )
        bars.append(bar)
    
    # Start progress monitor thread
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_dict, args.workers, bars),
        daemon=True
    )
    monitor_thread.start()
    
    # Run workers
    with Pool(args.workers) as p:
        results = list(p.imap_unordered(run_worker_wrapper, worker_args))
    
    # Wait for monitor to finish
    monitor_thread.join(timeout=1.0)
    
    # Close all progress bars
    for bar in bars:
        bar.close()
    
    # Print summary
    total_files = sum(files for files, _ in results)
    total_instances = sum(instances for _, instances in results)
    print(f"\nCompleted: {total_files} files generated from {total_instances} instances processed")

if __name__ == "__main__":
    main()
