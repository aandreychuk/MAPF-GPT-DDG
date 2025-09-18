# noinspection PyUnresolvedReferences
import cppimport.import_hook
from gpt.observation_generator import ObservationGenerator
import pyarrow as pa
import pyarrow.ipc as ipc
import os
from multiprocessing import Pool
import argparse

from lacam.inference import LacamInference, LacamInferenceConfig
import numpy as np

from pogema import pogema_v0
from pogema_toolbox.create_env import Environment
from pogema_toolbox.generators.maze_generator import MazeGenerator, MazeRangeSettings
from pogema_toolbox.generators.random_generator import MapRangeSettings as RandomRangeSettings, generate_map
from create_env import ProvideFutureTargetsWrapper
from house_generator import HouseGenerator, HouseRangeSettings

def make_pogema_house_instance(num_agents, max_episode_steps=256, width_min=10, width_max=20, height_min=10, height_max=20, obstacle_ratio_min=4, 
                               obstacle_ratio_max=8, remove_edge_ratio_min=4, remove_edge_ratio_max=10, on_target='nothing', map_seed=None, scenario_seed=None):
    rng = np.random.default_rng()
    settings_gen = HouseRangeSettings(width_min=width_min, width_max=width_max,
                                     height_min=height_min, height_max=height_max,
                                     obstacle_ratio_min=obstacle_ratio_min,
                                     obstacle_ratio_max=obstacle_ratio_max,
                                     remove_edge_ratio_min=remove_edge_ratio_min,
                                     remove_edge_ratio_max=remove_edge_ratio_max,
                                     )
    if map_seed is None:
        map_seed = rng.integers(np.iinfo(np.int64).max)
    house = HouseGenerator.generate(**settings_gen.sample(seed=map_seed))
    if scenario_seed is None:
        scenario_seed = rng.integers(np.iinfo(np.int64).max)
    env_cfg = Environment(
        num_agents=num_agents,
        observation_type="MAPF",
        max_episode_steps=max_episode_steps,
        map=house,
        with_animation=False,
        on_target=on_target,
        seed=scenario_seed,
        collision_system='soft'
    )
    env = pogema_v0(env_cfg)
    if on_target == 'restart':
        env = ProvideFutureTargetsWrapper(env)
    env_cfg.map_name = f'house-seed-{str(map_seed)}-scenario-{str(scenario_seed)}'
    return env

def make_pogema_maze_instance(num_agents, max_episode_steps=256, size_min=17, size_max=21, wall_components_min=4,
                              wall_components_max=8, on_target='nothing', map_seed=None, scenario_seed=None):
    rng = np.random.default_rng()

    settings_gen = MazeRangeSettings(width_min=size_min, width_max=size_max,
                                     height_min=size_max, height_max=size_max,
                                     wall_components_min=wall_components_min,
                                     wall_components_max=wall_components_max,
                                     )
    if map_seed is None:
        map_seed = rng.integers(np.iinfo(np.int64).max)
    maze = MazeGenerator.generate_maze(**settings_gen.sample(seed=map_seed))
    if scenario_seed is None:
        scenario_seed = rng.integers(np.iinfo(np.int64).max)
    env_cfg = Environment(
        num_agents=num_agents,
        observation_type="MAPF",
        max_episode_steps=max_episode_steps,
        map=maze,
        with_animation=False,
        on_target=on_target,
        seed=scenario_seed,
        collision_system='soft'
    )

    env = pogema_v0(env_cfg)
    if on_target == 'restart':
        env = ProvideFutureTargetsWrapper(env)
    env_cfg.map_name = f'maze-seed-{str(map_seed)}-scenario-{str(scenario_seed)}'
    return env


def make_pogema_random_instance(num_agents, max_episode_steps=256, size_min=17, size_max=21, on_target='nothing', map_seed=None,
                                scenario_seed=None):
    rng = np.random.default_rng()
    settings_gen = RandomRangeSettings(width_min=size_min, width_max=size_max, height_min=size_max, height_max=size_max)
    if map_seed is None:
        map_seed = rng.integers(np.iinfo(np.int64).max)
    maze = generate_map(settings_gen.sample(map_seed))
    if scenario_seed is None:
        scenario_seed = rng.integers(np.iinfo(np.int64).max)
    env_cfg = Environment(
        num_agents=num_agents,
        observation_type="MAPF",
        max_episode_steps=max_episode_steps,
        map=maze,
        with_animation=False,
        on_target=on_target,
        seed=scenario_seed,
        collision_system='soft'
    )

    env = pogema_v0(env_cfg)
    if on_target == 'restart':
        env = ProvideFutureTargetsWrapper(env)
    env_cfg.map_name = f'random-seed-{str(map_seed)}-scenario-{str(scenario_seed)}'
    return env

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

def save_to_arrow(inputs, gt_actions, filepath):
    schema = pa.schema([
        ('input_tensors', pa.list_(pa.int8())),
        ('gt_actions', pa.int8())
    ])

    input_tensors_col = pa.array(inputs, type=pa.list_(pa.int8()))
    gt_actions_col = pa.array(gt_actions, type=pa.int8())
    table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write(table)

def get_solved_instance(env):
    expert_algo = LacamInference(LacamInferenceConfig(time_limit=10, timeouts=[10]))
    observations, *_ = env.reset()
    actions, grid, starts, targets = expert_algo.get_full_solution(observations, env.grid_config.max_episode_steps)
    print(actions, '\n', grid, '\n', starts, '\n', targets)
    return actions, grid, starts, targets

def fill_actions_with_solver(env):
    expert_algo = LacamInference(LacamInferenceConfig(time_limit=10, timeouts=[10]))
    observations, *_ = env.reset()

    observation_generator = ObservationGenerator(observations[0]["global_obstacles"].copy().astype(int).tolist(), 5, 128)
    positions = [obs["global_xy"] for obs in observations]
    goals = [obs["global_target_xy"] for obs in observations]
    observation_generator.create_agents(positions, goals)
    observation_generator.update_agents(positions, goals, [-1 for _ in range(len(observations))])
    inputs = []
    gt_actions = []
    while True:
        input = observation_generator.generate_observations()
        actions = expert_algo.act(observations)
        observations, rew, terminated, truncated, infos = env.step(actions)
        #inputs.append(input)
        #gt_actions.append(actions.copy())
        inputs.extend(input)
        gt_actions.extend(actions.copy())
        
        positions = [obs["global_xy"] for obs in observations]
        goals = [obs["global_target_xy"] for obs in observations]
        observation_generator.update_agents(positions, goals, actions)
        if all(terminated) or all(truncated):
            break
    print(env.grid_config.map_name, env.grid_config.seed,infos[0]['metrics'])
    if infos[0]['metrics']['CSR'] == 0:
        return [], []
    return inputs, gt_actions

def run_worker(seeds, worker_id, instance_type, data_per_file, num_agents, dataset_path, files_per_worker, save_instances=False):
    all_inputs = []
    all_gt_actions = []
    all_grids = []
    all_starts = []
    all_targets = []
    all_seeds = []
    total_generated = 0
    for seed in seeds:
        if instance_type == 'mazes':
            env = make_pogema_maze_instance(num_agents=num_agents, max_episode_steps=256, map_seed=seed, scenario_seed=seed, on_target='nothing')
            folder = f"{dataset_path}/mazes/num_agents_{num_agents}"
        elif instance_type == 'random':
            env = make_pogema_random_instance(num_agents=num_agents, max_episode_steps=256, map_seed=seed, scenario_seed=seed, on_target='nothing')
            folder = f"{dataset_path}/random/num_agents_{num_agents}"
        elif instance_type == 'house':
            env = make_pogema_house_instance(num_agents=num_agents, max_episode_steps=256, map_seed=seed, scenario_seed=seed, on_target='nothing')
            folder = f"{dataset_path}/house/num_agents_{num_agents}"
        else:
            raise ValueError(f"Unknown instance_type: {instance_type}")
        if save_instances:
            gt_actions, grid, starts, targets = get_solved_instance(env)
            if gt_actions is None:
                continue
            all_gt_actions.append(gt_actions)
            all_grids.append(grid)
            all_starts.append(starts)
            all_targets.append(targets)
            all_seeds.append(seed)
            if len(all_gt_actions) > data_per_file:
                save_to_arrow_with_grid(all_gt_actions[:data_per_file], all_grids[:data_per_file], all_starts[:data_per_file], all_targets[:data_per_file], all_seeds[:data_per_file], f"{folder}/part_{worker_id}_{total_generated}.arrow")
                all_gt_actions = all_gt_actions[data_per_file:]
                all_grids = all_grids[data_per_file:]
                all_starts = all_starts[data_per_file:]
                all_targets = all_targets[data_per_file:]
                all_seeds = all_seeds[data_per_file:]
                total_generated += 1
        else:
            inputs, gt_actions = fill_actions_with_solver(env)
            if len(inputs) == 0:
                continue
            all_inputs.extend(inputs)
            all_gt_actions.extend(gt_actions)
            if len(all_inputs) > data_per_file:
                save_to_arrow(all_inputs[:data_per_file], all_gt_actions[:data_per_file], f"{folder}/part_{worker_id}_{total_generated}.arrow")
                all_inputs = all_inputs[data_per_file:]
                all_gt_actions = all_gt_actions[data_per_file:]
                total_generated += 1
        if total_generated >= files_per_worker:
            break

def main():
    parser = argparse.ArgumentParser(description="Collect MAPF data.")
    parser.add_argument('--instance_type', choices=['mazes', 'random', 'house'], default='mazes', help='Type of instance to generate: mazes or random')
    parser.add_argument('--data_per_file', type=int, default=2**10, help='Number of data samples per file')
    parser.add_argument('--num_agents', type=int, default=32, help='Number of agents')
    parser.add_argument('--dataset_path', type=str, default='dataset_mapf', help='Path to save the dataset')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--total_files', type=int, default=512, help='Total number of files to generate')
    parser.add_argument('--save_instances', type=bool, default=False, help='Save instances')
    args = parser.parse_args()

    files_per_worker = args.total_files // args.workers
    seeds = [i for i in range(128, 1000000*args.workers)]
    seeds_per_worker = len(seeds) // args.workers
    with Pool(args.workers) as p:
        p.starmap(run_worker, [
            (seeds[i * seeds_per_worker:(i + 1) * seeds_per_worker], i, args.instance_type, args.data_per_file, args.num_agents, args.dataset_path, files_per_worker, args.save_instances)
            for i in range(args.workers)
        ])

if __name__ == "__main__":
    main()
