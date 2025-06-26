from copy import deepcopy
from pathlib import Path

import yaml
from pogema import BatchAStarAgent

from pogema_toolbox.create_env import Environment

from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.run_episode import run_episode

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from utils.svg_utils import create_multi_animation


def run_episode_algos_to_svg(env, algos, filename='multi.svg'):
    histories = []
    for algo in algos:
        run_episode(env, algo)
        histories.append(env.decompress_history(env.get_history()))

    obstacles = env.get_obstacles(ignore_borders=False)
    grid_config = deepcopy(env.grid.config)
    create_multi_animation(obstacles, histories=histories, grid_config=grid_config, name=filename)


def main():
    env_cfg = Environment(
        observation_type="MAPF",
        on_target="nothing",
        map_name='validation-random-seed-001',
        max_episode_steps=32,
        num_agents=32,
        seed=42,
        obs_radius=5,
        collision_system="soft",
        with_animation=True
    )

    for maps_file in Path("eval_configs").rglob('maps.yaml'):
        with open(maps_file, 'r') as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

    env = create_eval_env(env_cfg)
    algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'../weights/model-2M.pt', device='cuda'))
    run_episode_algos_to_svg(env, [algo, BatchAStarAgent()], filename='out.svg')


if __name__ == "__main__":
    main()
