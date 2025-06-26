import numpy as np

from pogema import pogema_v0
from pogema_toolbox.create_env import Environment
from pogema_toolbox.generators.maze_generator import MazeGenerator, MazeRangeSettings
from pogema_toolbox.generators.random_generator import MapRangeSettings as RandomRangeSettings, generate_map
from create_env import ProvideFutureTargetsWrapper

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

def make_pogema_map_instance(num_agents, map, max_episode_steps=256, on_target='nothing', scenario_seed=None):
    rng = np.random.default_rng()
    if scenario_seed is None:
        scenario_seed = rng.integers(np.iinfo(np.int64).max)
    env_cfg = Environment(
        num_agents=num_agents,
        observation_type="MAPF",
        max_episode_steps=max_episode_steps,
        map=map,
        with_animation=False,
        on_target=on_target,
        seed=scenario_seed,
        collision_system='soft'
    )
    env = pogema_v0(env_cfg)
    if on_target == 'restart':
        env = ProvideFutureTargetsWrapper(env)
    env_cfg.map_name = f'warehouse-scenario-{str(scenario_seed)}'
    return env


def main():
    maze_env = make_pogema_maze_instance(num_agents=32, map_seed=42, scenario_seed=42)
    maze_env.reset()
    maze_env.render()

    random_env = make_pogema_random_instance(num_agents=32, map_seed=42, scenario_seed=42)
    random_env.reset()
    random_env.render()

    grid = '''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !@@!@@!$$$$$$$$$$!$$$$$$$$$$!$$$$$$$$$$!@@!@@!
    !@@!@@!##########!##########!##########!@@!@@!
    !@@!@@!$$$$$$$$$$!$$$$$$$$$$!$$$$$$$$$$!@@!@@!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    warehouse_env = make_pogema_map_instance(num_agents=8, map=grid, scenario_seed=42)
    warehouse_env.reset()
    warehouse_env.render()

if __name__ == '__main__':
    main()
