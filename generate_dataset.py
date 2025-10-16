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
    schema = pa.schema([
        ('input_tensors', pa.list_(pa.int8())),
        ('gt_actions', pa.list_(pa.int8()))
    ])

    input_tensors_col = pa.array(inputs, type=pa.list_(pa.int8()))
    gt_actions_col = pa.array(gt_actions, type=pa.list_(pa.int8()))
    table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write(table)

def filter_and_balance_observations(observations, actions):
    actions_array = np.array(actions, dtype=np.int8)
    if actions_array.ndim == 1:
        actions_array = actions_array[:, None]

    horizon = actions_array.shape[1]
    total_actions = len(actions_array) * horizon
    zero_mask = actions_array == 0
    total_zero_actions = int(zero_mask.sum())

    zero_tuple_mask = np.all(zero_mask, axis=1)
    zero_tuple_indices = np.where(zero_tuple_mask)[0]
    zero_tuple_count = int(zero_tuple_indices.size)

    total_tuples = len(actions_array)
    target_ratio = 0.2

    if zero_tuple_count == 0 or total_zero_actions <= target_ratio * total_actions:
        filtered_indices = np.arange(total_tuples)
        filtered_observations = [observations[i] for i in filtered_indices]
        filtered_actions = actions_array[filtered_indices].tolist()
    else:
        numerator = total_zero_actions - target_ratio * total_actions
        remove_tuple_count = math.ceil(numerator / ((1 - target_ratio) * horizon))
        remove_tuple_count = min(max(remove_tuple_count, 0), zero_tuple_count)

        keep_mask = np.ones(total_tuples, dtype=bool)
        if remove_tuple_count > 0:
            remove_indices = zero_tuple_indices[:remove_tuple_count]
            keep_mask[remove_indices] = False

        filtered_indices = np.where(keep_mask)[0]
        filtered_observations = [observations[i] for i in filtered_indices]
        filtered_actions = actions_array[filtered_indices].tolist()
    final_observations = []
    final_actions = []
    check_obs = set()
    for i, obs in enumerate(filtered_observations):
        if tuple(obs) not in check_obs:
            check_obs.add(tuple(obs))
            final_observations.append(obs)
            final_actions.append(filtered_actions[i])

    return final_observations, final_actions

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
    observations = []
    actions = []
    positions = starts
    moves = GridConfig().MOVES
    for i in range(len(gt_actions)):
        current_observations = observation_generator.generate_observations()
        observations.extend(current_observations)
        num_agents = len(current_observations)
        for agent_idx in range(num_agents):
            a0 = gt_actions[i][agent_idx]
            a1 = gt_actions[i+1][agent_idx] if i + 1 < len(gt_actions) else 0
            a2 = gt_actions[i+2][agent_idx] if i + 2 < len(gt_actions) else 0
            actions.append([a0, a1, a2])
        positions = [[positions[j][0] + moves[gt_actions[i][j]][0], positions[j][1] + moves[gt_actions[i][j]][1]] for j in range(len(positions))]
        observation_generator.update_agents(positions, targets, gt_actions[i])
    return observations, actions

def run_worker(files, log_path, dataset_path, samples_per_file, worker_id):
    all_observations = []
    all_actions = []
    part_id = 0
    for file in files:
        file_path = os.path.join(log_path, file)
        gt_actions, grid, starts, targets, seeds = get_data_from_file(file_path)
        for i in range(len(gt_actions)):
            observations, actions = generate_observations(gt_actions[i], grid[i], starts[i], targets[i], seeds[i])
            filtered_observations, filtered_actions = filter_and_balance_observations(observations, actions)
            all_observations.extend(filtered_observations)
            all_actions.extend(filtered_actions)
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