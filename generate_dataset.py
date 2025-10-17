import math
import pyarrow as pa
import pyarrow.ipc as ipc
import os
import numpy as np
from cppimport import import_hook
from gpt.observation_generator import ObservationGenerator
from pogema import GridConfig
import argparse
from multiprocessing import Pool

def save_to_arrow(inputs, gt_actions, filepath):
    # inputs: List[timestep][agent][int8]
    # gt_actions: List[timestep][agent][int8]
    schema = pa.schema([
        ('input_tensors', pa.list_(pa.list_(pa.int8()))),
        ('gt_actions', pa.list_(pa.list_(pa.int8())))
    ])

    input_tensors_col = pa.array(inputs, type=pa.list_(pa.list_(pa.int8())))
    gt_actions_col = pa.array(gt_actions, type=pa.list_(pa.list_(pa.int8())))
    table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write(table)

def filter_and_balance_timesteps(timestep_observations, timestep_actions, target_ratio=0.2):
    """
    Filter timesteps to reduce over-representation of zero actions.
    Only considers the first action (a0) in each tuple for balancing.

    - timestep_observations: List[timestep][agent][int8]
    - timestep_actions: List[timestep][agent][int8] where per-agent action is a horizon tuple (length 10)
    """
    if len(timestep_actions) == 0:
        return timestep_observations, timestep_actions

    # Compute totals and per-timestep zero counts (only considering first action a0)
    total_actions = 0
    total_zero_actions = 0
    per_timestep_stats = []  # (t_idx, zeros_in_t, total_in_t)

    for t_idx, actions_at_t in enumerate(timestep_actions):
        if len(actions_at_t) == 0:
            per_timestep_stats.append((t_idx, 0, 0))
            continue
        # Only consider the first action (a0) in each tuple
        first_actions = [action_tuple[0] for action_tuple in actions_at_t]  # Extract a0 from each agent
        zeros_in_t = int(sum(1 for a0 in first_actions if a0 == 0))
        total_in_t = len(first_actions)
        total_zero_actions += zeros_in_t
        total_actions += total_in_t
        per_timestep_stats.append((t_idx, zeros_in_t, total_in_t))

    if total_actions == 0:
        return timestep_observations, timestep_actions

    if total_zero_actions <= target_ratio * total_actions:
        return timestep_observations, timestep_actions

    # Sort timesteps by zero count descending; remove greedily until balanced
    removable = sorted(per_timestep_stats, key=lambda x: x[1], reverse=True)
    keep_mask = np.ones(len(timestep_actions), dtype=bool)
    total_removed = 0
    for t_idx, zeros_in_t, total_in_t in removable:
        if total_zero_actions <= target_ratio * total_actions:
            break
        # Remove this timestep
        if total_in_t <= 0:
            keep_mask[t_idx] = False
            continue
        keep_mask[t_idx] = False
        total_zero_actions -= zeros_in_t
        total_actions -= total_in_t
        total_removed += total_in_t
    filtered_obs = [timestep_observations[i] for i in range(len(timestep_observations)) if keep_mask[i]]
    filtered_actions = [timestep_actions[i] for i in range(len(timestep_actions)) if keep_mask[i]]

    '''
    print("out of ", total_actions+total_removed, " timesteps, ", total_removed, " were removed")
    action_counts = {}
    for actions_at_t in filtered_actions:
        for action_tuple in actions_at_t:
            if action_tuple: # Ensure action_tuple is not empty
                a0 = action_tuple[0]
                action_counts[a0] = action_counts.get(a0, 0) + 1

    total_filtered_actions = sum(action_counts.values())
    if total_filtered_actions > 0:
        print("Distribution of first actions (a0) in filtered dataset:")
        for action_val in sorted(action_counts.keys()):
            count = action_counts[action_val]
            percentage = (count / total_filtered_actions) * 100
            print(f"  Action {action_val}: {count} ({percentage:.2f}%)")
    else:
        print("No actions in filtered dataset.")
    '''
    return filtered_obs, filtered_actions

def get_data_from_file(file_path):
    with pa.memory_map(file_path) as source:
        table = pa.ipc.open_file(source).read_all()
        gt_actions = table["gt_actions"].to_pylist()
        grid = table["grid"].to_pylist()
        starts = table["starts"].to_pylist()
        targets = table["targets"].to_pylist()
        seeds = table["seeds"].to_pylist()
    return gt_actions, grid, starts, targets, seeds

def generate_observations(gt_actions, grid, starts, targets, seeds):
    observation_generator = ObservationGenerator(grid, 3, 64)
    observation_generator.create_agents(starts, targets)
    observation_generator.update_agents(starts, targets, [-1 for _ in range(len(starts))])
    # Accumulate per-timestep lists
    observations = []  # List[t][agent][...]
    actions = []       # List[t][agent][a0,a1,a2,...,a9] (10 actions)
    positions = starts
    moves = GridConfig().MOVES
    for i in range(len(gt_actions)):
        current_observations = observation_generator.generate_observations()
        observations.append(current_observations)
        num_agents = len(current_observations)
        timestep_actions = []
        for agent_idx in range(num_agents):
            # Create action tuple with 10 actions (a0 through a9)
            action_tuple = []
            for j in range(10):
                if i + j < len(gt_actions):
                    action_tuple.append(gt_actions[i + j][agent_idx])
                else:
                    action_tuple.append(0)  # Pad with 0 if beyond available timesteps
            timestep_actions.append(action_tuple)
        actions.append(timestep_actions)
        positions = [[positions[j][0] + moves[gt_actions[i][j]][0], positions[j][1] + moves[gt_actions[i][j]][1]] for j in range(len(positions))]
        observation_generator.update_agents(positions, targets, gt_actions[i])
    return observations, actions

def run_worker(files, log_path, dataset_path, samples_per_file, worker_id):
    # Buffers hold timesteps
    all_observations = []  # List[t][agent][...]
    all_actions = []       # List[t][agent][a0,a1,a2,...,a9] (10 actions)
    part_id = 0
    for file in files:
        file_path = os.path.join(log_path, file)
        gt_actions, grid, starts, targets, seeds = get_data_from_file(file_path)
        for i in range(len(gt_actions)):
            observations, actions = generate_observations(gt_actions[i], grid[i], starts[i], targets[i], seeds[i])
            # Append raw timesteps first; filter later when buffer big enough
            all_observations.extend(observations)
            all_actions.extend(actions)

            # Trigger filtering when buffer exceeds 2x samples_per_file timesteps
            if len(all_observations) >= 2 * samples_per_file:
                all_observations, all_actions = filter_and_balance_timesteps(all_observations, all_actions)
                # Save when enough timesteps available
                if len(all_observations) >= samples_per_file:
                    save_to_arrow(all_observations[:samples_per_file], all_actions[:samples_per_file], os.path.join(dataset_path, f"part_{worker_id}_{part_id}.arrow"))
                    all_observations = all_observations[samples_per_file:]
                    all_actions = all_actions[samples_per_file:]
                    part_id += 1
        print(f"{file} processed")

def __main__():
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('--log_path', type=str, default='logs_mapf/random/num_agents_32', help='Path to the logs')
    parser.add_argument('--dataset_path', type=str, default='dataset_mapf/random', help='Path to the dataset')
    parser.add_argument('--samples_per_file', type=int, default=10000, help='Number of samples per file')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    args = parser.parse_args()
    file_path = args.log_path
    files = os.listdir(file_path)
    files_per_worker = len(files) // args.workers
    with Pool(args.workers) as p:
        p.starmap(run_worker, [(files[i * files_per_worker:(i + 1) * files_per_worker], args.log_path, args.dataset_path, args.samples_per_file, i) for i in range(args.workers)])
    print("DONE")

if __name__ == "__main__":
    __main__()