
import pyarrow as pa
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from dataset_generator.instance_generator import make_pogema_maze_instance, make_pogema_random_instance, make_pogema_house_instance

def load_gt_data(filepath):
    with pa.memory_map(filepath) as source:
        table = pa.ipc.open_file(source).read_all()
        gt_actions = table["gt_actions"].to_pylist()
        grid = table["grid"].to_pylist()
        starts = table["starts"].to_pylist()
        targets = table["targets"].to_pylist()
        seeds = table["seeds"].to_pylist()
        return gt_actions, grid, starts, targets, seeds

def validate_gt_data(gt_actions, grid, starts, targets, seeds):
    for i in range(len(gt_actions)):
        env = make_pogema_maze_instance(num_agents=len(starts[i]), max_episode_steps=256, map_seed=seeds[i], scenario_seed=seeds[i], on_target='nothing')
        env.reset()
        for j in range(len(gt_actions[i])):
            obs, rew, terminated, truncated, infos = env.step(gt_actions[i][j])
            if all(terminated) or all(truncated):
                break
        print(infos[0]['metrics'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GT data.")
    parser.add_argument('--dataset_path', type=str, default='dataset_mapf', help='Path to the dataset')
    parser.add_argument('--instance_type', choices=['mazes', 'random', 'house'], default='mazes', help='Type of instance to generate: mazes or random')
    parser.add_argument('--num_agents', type=int, default=32, help='Number of agents')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    instance_type = args.instance_type
    num_agents = args.num_agents
    gt_actions, grid, starts, targets, seeds = load_gt_data(f"{dataset_path}/{instance_type}/num_agents_{num_agents}/part_0_0.arrow")
    validate_gt_data(gt_actions, grid, starts, targets, seeds)