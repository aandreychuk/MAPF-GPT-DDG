import argparse
from pathlib import Path
import random
import time
from tqdm.auto import tqdm

import torch
import yaml
from pogema_toolbox.create_env import Environment

from pogema_toolbox.registry import ToolboxRegistry
from pogema import AnimationConfig
from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from pogema_toolbox.results_holder import ResultsHolder

def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()

    obs, _ = env.reset()#seed=env.grid_config.seed)
#    while True:
    for _ in tqdm(range(env.grid.config.max_episode_steps), desc="Running episode"):
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)

        if all(terminated) or all(truncated):
            break
    return results_holder.get_final()

def main():
    random.seed(42)
    start_xy = set()
    size = 2048
    while len(start_xy) < 2**20:
        start_xy.add((random.randint(0, size-1), random.randint(0, size-1)))
    start_xy = list(start_xy)
    
    delta = 64
    goal_xy = []
    used_goals = set()  # Track already used goal positions
    
    for sx, sy in start_xy:
        while True:
            dx = random.randint(-delta, delta)
            dy = random.randint(-(delta-abs(dx)), delta-abs(dx))

            gx, gy = sx + dx, sy + dy
            if 0 <= gx < size and 0 <= gy < size and (gx, gy) not in used_goals:
                goal_xy.append([gx, gy])
                used_goals.add((gx, gy))  # Mark this goal as used
                break

    for s in [32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576]:
        start_time = time.time()
        env_cfg = Environment(
            with_animation=False,
            observation_type="MAPF",
            on_target="nothing",
            size=size,
            density=0,
            agents_xy=start_xy[:s],
            targets_xy=goal_xy[:s],
            max_episode_steps=512,
            seed=0,
            obs_radius=5,
            collision_system="soft",
        )

        env = create_eval_env(env_cfg)
        create_time = time.time()
        algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'weights/model-2M-empty.pt', device='cuda'))
        algo.reset_states()
        results = run_episode(env, algo)
        end_time = time.time()
        env_time = end_time - create_time - results['runtime']
        total_time = end_time - start_time
        results['env_time'] = env_time
        results['total_time'] = total_time
        print(s, results) #, "Time to create env: ", create_time-start_time, "Time to run env: ", env_time, "Total time: ", total_time)



if __name__ == "__main__":
    main()
