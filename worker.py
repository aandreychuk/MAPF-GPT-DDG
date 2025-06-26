import argparse
import json
from pogema_toolbox.registry import ToolboxRegistry

import os
import sys
import yaml
from utils.data_utils import save_to_arrow
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from finetuning.data_aggregation_generator import data_aggregation, DataAggregationConfig
from finetuning.scenario_generators import make_pogema_maze_instance, make_pogema_random_instance, make_pogema_map_instance
from finetuning.delta_data_generator import fast_solver_delta, FastSolverDeltaConfig
from lacam.inference import LacamInference, LacamInferenceConfig

FAST_LACAM_TIMELIMIT = 2
EXPERT_LACAM_TIMELIMIT = 10
EPISODE_LENGTH = 256
NUM_ENVS = 8

def collect_data(seeds, env_generator, worker_id, device_id, actions_required, dagger_type, on_target, path_to_weights, num_agents, grid=None):
    inputs = []
    gt_actions = []
    actions_counters = [0 for _ in range(5)]
    learnable_algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=path_to_weights,
                                                             device=f"cuda:{device_id}"))
    fast_solver = LacamInference(LacamInferenceConfig(time_limit=FAST_LACAM_TIMELIMIT, timeouts=[FAST_LACAM_TIMELIMIT]))
    solver = LacamInference(LacamInferenceConfig(time_limit=EXPERT_LACAM_TIMELIMIT, timeouts=[EXPERT_LACAM_TIMELIMIT]))
    logs = []
    
    for seed_idx in range(0, len(seeds), NUM_ENVS):
        envs = []
        for i in range(NUM_ENVS):
            if grid is None:
                map_seed, scen_seed = seeds[seed_idx + i]
                env = env_generator(num_agents=num_agents[map_seed % len(num_agents)],
                                    max_episode_steps=EPISODE_LENGTH,
                                    map_seed=map_seed,
                                    scenario_seed=scen_seed,
                                    on_target=on_target)
            else:
                scenario_seed = seeds[seed_idx + i]
                env = env_generator(num_agents=num_agents[scenario_seed % len(num_agents)],
                                    max_episode_steps=EPISODE_LENGTH,
                                    map=grid,
                                    scenario_seed=scenario_seed,
                                    on_target=on_target)
            envs.append(env)
        if 'ddg' in dagger_type:
            data, log = fast_solver_delta(envs, learnable_algo, fast_solver, solver, FastSolverDeltaConfig(on_target=on_target))
            logs.extend(log)
        else:
            data = {'inputs': [], 'gt_actions': []}
            for env in envs:
                data_env, log = data_aggregation(env, learnable_algo, DataAggregationConfig(on_target=on_target))
                data['inputs'].extend(data_env['inputs'])
                data['gt_actions'].extend(data_env['gt_actions'])
                logs.append(log)
        if data:
            inputs.extend(data['inputs'])
            gt_actions.extend(data['gt_actions'])
            actions_counters = [actions_counters[i] + data['gt_actions'].count(i) for i in range(5)]
        ToolboxRegistry.debug(f"Collected {len(inputs)} pairs by worker {worker_id}, current min actions: {min(actions_counters)}, action with min count: {actions_counters.index(min(actions_counters))}, actions required: {actions_required}")
        if min(actions_counters) >= actions_required:
            break
    
    balanced_inputs = []
    balanced_gt_actions = []
    actions_counters = [0 for _ in range(5)]
    for input, gt_action in zip(inputs, gt_actions):
        if actions_counters[gt_action] < actions_required:
            actions_counters[gt_action] += 1
            balanced_inputs.append(input)
            balanced_gt_actions.append(gt_action)
    
    return balanced_inputs, balanced_gt_actions, logs

def main():
    parser = argparse.ArgumentParser(description='Dagger Worker')
    parser.add_argument('--worker_id', type=int, default=0, help='Worker ID (default: %(default)d)')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID (default: %(default)d)')
    parser.add_argument('--dagger_type', type=str, choices=['ddg', 'ddg_warehouse', 'dagger', 'dagger_warehouse'], default='ddg', help='Dagger type (default: %(default)s)')
    parser.add_argument('--num_agents', type=str, default='32', help='Comma-separated list of number of agents (default: %(default)s)')
    parser.add_argument('--map_seed', type=int, default=0, help='Starting map seed (default: %(default)d)')
    parser.add_argument('--scenario_seed', type=int, default=0, help='Starting scenario seed (default: %(default)d)')
    parser.add_argument('--seeds', type=int, default=1000, help='Number of seeds (default: %(default)d)')
    parser.add_argument('--map_path', type=str, default='eval_configs/03-warehouse/maps.yaml', help='Path to the map (default: %(default)s)')
    parser.add_argument('--map_name', type=str, default='wfi_warehouse', help='Map name (default: %(default)s)')
    parser.add_argument('--path_to_weights', type=str, default='out/ckpt.pt', help='Path to the weights (default: %(default)s)')
    parser.add_argument('--dataset_path', type=str, default='dataset/dagger', help='Folder to save the dataset (default: %(default)s)')
    parser.add_argument('--file_size', type=int, default=50 * 2 ** 11, help='File size (default: %(default)d)')
    args = parser.parse_args()
    all_logs = []
    num_agents = list(map(int, args.num_agents.split(',')))
    if args.dagger_type == 'ddg' or args.dagger_type == 'dagger':
        seeds = [(args.map_seed + i, args.scenario_seed) for i in range(args.seeds)]
        maze_inputs, maze_gt_actions, logs = collect_data(seeds=seeds, 
                                                    env_generator=make_pogema_maze_instance, 
                                                    worker_id=args.worker_id, 
                                                    device_id=args.device_id,
                                                    actions_required=args.file_size // 5 - args.file_size // 50, 
                                                    dagger_type=args.dagger_type,
                                                    on_target="nothing",
                                                    path_to_weights=args.path_to_weights,
                                                    num_agents=num_agents)
        all_logs.extend(logs)
        random_inputs, random_gt_actions, logs = collect_data(seeds=seeds, 
                                                        env_generator=make_pogema_random_instance, 
                                                        worker_id=args.worker_id, 
                                                        device_id=args.device_id,
                                                        actions_required=args.file_size // 50, 
                                                        dagger_type=args.dagger_type,
                                                        on_target="nothing",
                                                        path_to_weights=args.path_to_weights,
                                                        num_agents=num_agents)
        all_logs.extend(logs)
        balanced_inputs = maze_inputs + random_inputs
        balanced_gt_actions = maze_gt_actions + random_gt_actions
        
        filepath = f"{args.dataset_path}/data_{args.worker_id}_{args.map_seed}.arrow"
        save_to_arrow(balanced_inputs[:args.file_size], balanced_gt_actions[:args.file_size], filepath)
        ToolboxRegistry.info(f"Collected {len(balanced_inputs)} pairs by worker {args.worker_id}")
    elif args.dagger_type == 'ddg_warehouse' or args.dagger_type == 'dagger_warehouse':
        with open(args.map_path, 'r') as f:
            grid = yaml.safe_load(f)[args.map_name]
        seeds = [args.scenario_seed + i for i in range(args.seeds)]
        balanced_inputs, balanced_gt_actions, logs = collect_data(seeds=seeds, 
                                                            env_generator=make_pogema_map_instance, 
                                                            worker_id=args.worker_id, 
                                                            device_id=args.device_id,
                                                            actions_required=args.file_size // 5, 
                                                            dagger_type=args.dagger_type,
                                                            on_target="nothing",
                                                            path_to_weights=args.path_to_weights,
                                                            num_agents=num_agents, 
                                                            grid=grid)
        all_logs.extend(logs)
        filepath = f"{args.dataset_path}/data_{args.worker_id}_{args.scenario_seed}.arrow"
        save_to_arrow(balanced_inputs[:args.file_size], balanced_gt_actions[:args.file_size], filepath)
        ToolboxRegistry.info(f"Collected {len(balanced_inputs)} pairs by worker {args.worker_id}")
    log_file = f"{args.dataset_path}/logs_{args.worker_id}_{args.path_to_weights.split('/')[-1].split('.')[0]}.json"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            old_logs = json.load(f)
        old_logs.extend(all_logs)
        all_logs = old_logs
    with open(log_file, 'w') as f:
        json.dump(all_logs, f)

if __name__ == '__main__':
    ToolboxRegistry.setup_logger('DEBUG')
    main()
