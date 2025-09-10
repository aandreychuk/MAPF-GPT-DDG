from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

from copy import deepcopy
from pathlib import Path

from pogema import AnimationMonitor, AnimationConfig

from pogema_toolbox.run_episode import run_episode
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import BaseModel

from finetuning.filter_data import filter_data

from utils.data_collection import fill_actions_with_solver
from finetuning.scenario_generators import make_pogema_maze_instance

from utils.svg_utils import cut_history, create_multi_animation
from utils.wrappers import UnrollWrapper

from multiprocessing import Pool
from lacam.inference import LacamInference, LacamInferenceConfig
from pogema.wrappers.metrics import RuntimeMetricWrapper
from macro_env import PogemaMacroEnvironment, MAPFGPTObservationWrapper
from gpt.observation_generator import ObservationGenerator

class FastSolverDeltaConfig(BaseModel):
    steps_delta: int = 16
    steps_saved: int = 32
    save_debug_svg: bool = False
    diff_threshold = 3


def run_solver(env, unroll_steps, time_limit):
    env = deepcopy(env)
    solver = LacamInference(LacamInferenceConfig(time_limit=time_limit, timeouts=[time_limit]))
    env.set_unroll_steps(unroll_steps)
    results = run_episode(env, solver)
    results['step'] = unroll_steps
    results['map_name'] = env.grid.config.map_name
    return results

def run_episode_macro(env, algo):
    algo.reset_states()
    obs, _ = env.reset()
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        if all(terminated) or all(truncated):
            break
    return [info[0]['metrics'] for info in infos]

def run_expert(env, unroll_steps, steps_saved, time_limit):
    env = deepcopy(env)
    solver = LacamInference(LacamInferenceConfig(time_limit=time_limit, timeouts=[time_limit]))
    input, gt_action, metrics = fill_actions_with_solver(env, unroll_steps, steps_saved, solver)
    if metrics is not None:
        metrics['step'] = unroll_steps
        metrics['map_name'] = env.grid.config.map_name
    return input, gt_action, metrics

def fast_solver_delta(envs, learnable_algo, fast_solver, solver, cfg: FastSolverDeltaConfig):

    def create_svg(env, unroll_steps):
        obstacles = env.get_obstacles(ignore_borders=False)
        algo_history = env.get_full_history()
        fast_env = deepcopy(env)
        fast_env.set_unroll_steps(unroll_steps)
        run_episode(fast_env, fast_solver)
        fast_solver_history = fast_env.get_full_history()
        oracle_env = deepcopy(env)
        oracle_env.set_unroll_steps(unroll_steps)
        run_episode(oracle_env, solver)
        oracle_history = oracle_env.get_full_history()
        histories = [algo_history, fast_solver_history, oracle_history]
        ToolboxRegistry.debug('Histories sizes: ' + str([len(x[0]) for x in histories]))
        cut_histories = [cut_history(x, start=unroll_steps, finish=unroll_steps + cfg.steps_saved) for x in histories]
        ToolboxRegistry.debug('Cut histories sizes: ' + str([len(x[0]) for x in cut_histories]))

        svg_path = f'renders/seed-{env.grid.config.map_name}-step-{unroll_steps}.svg'
        Path(svg_path).parent.mkdir(exist_ok=True)
        create_multi_animation(obstacles, cut_histories, env.grid.config, name=svg_path)
        ToolboxRegistry.debug(f'Saved svg to: {svg_path}')

    inputs = []
    gt_actions = []
    gpt_envs = []
    for env in envs:
        env = RuntimeMetricWrapper(env)
        if cfg.save_debug_svg:
            env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
        obs, _ = env.reset(seed=env.grid_config.seed)
        obs_generator = ObservationGenerator(obs[0]["global_obstacles"].copy().astype(int).tolist(), 5, 128)
        obs_generator.create_agents([o["global_xy"] for o in obs], [o["global_target_xy"] for o in obs])
        env = UnrollWrapper(env)
        env = MAPFGPTObservationWrapper(env, obs_generator)
        gpt_envs.append(env)
    macro_env = PogemaMacroEnvironment(gpt_envs)
    gpt_results = run_episode_macro(macro_env, learnable_algo)

    envs = [env.get_inner_env() for env in macro_env.environments]

    unroll_steps_lists = []
    for gpt_result in gpt_results:
        unroll_steps_list = range(0, gpt_result['ep_length'], cfg.steps_delta)
        unroll_steps_lists.append(unroll_steps_list)
    
    with Pool(processes=8) as pool:
        fast_solver_results = pool.starmap(run_solver, 
            [(env, unroll_steps, 2) for env, unroll_steps_list in zip(envs, unroll_steps_lists) for unroll_steps in unroll_steps_list])

    fast_solver_results_by_map = {}
    for result in fast_solver_results:
        if result['map_name'] not in fast_solver_results_by_map:
            fast_solver_results_by_map[result['map_name']] = {}
        fast_solver_results_by_map[result['map_name']][result['step']] = result
    

    diffs_by_map = {}
    for map_name, results in fast_solver_results_by_map.items():
        unroll_steps = sorted(results.keys())
        diffs = []
        for i in range(1, len(unroll_steps)):
            prev_step = unroll_steps[i - 1]
            curr_step = unroll_steps[i]
            diff = results[curr_step]['makespan'] - results[prev_step]['makespan']
            diffs.append(diff)
        diffs_by_map[map_name] = diffs

    max_diff_indices = {map_name: diffs.index(max(diffs)) for map_name, diffs in diffs_by_map.items()}

    envs_with_positive_diffs = []
    for env in envs:
        if diffs_by_map[env.grid.config.map_name][max_diff_indices[env.grid.config.map_name]] > cfg.diff_threshold:
            env.set_unroll_steps(cfg.steps_delta*max_diff_indices[env.grid.config.map_name])
            envs_with_positive_diffs.append((env, cfg.steps_delta*max_diff_indices[env.grid.config.map_name]))
    ToolboxRegistry.debug(f'Makespan difference: {diffs_by_map}')
    with Pool(processes=8) as pool:
        expert_results = pool.starmap(run_expert, 
            [(env, unroll_steps, cfg.steps_saved, 10) for env, unroll_steps in envs_with_positive_diffs])
        
    inputs = []
    gt_actions = []
    expert_logs = {}
    for result in expert_results:
        if result[0] is not None:
            filtered_data = filter_data(result[0], result[1])
            inputs.extend(filtered_data['inputs'])
            gt_actions.extend(filtered_data['gt_actions'])
            expert_logs[result[2]['map_name']] = result[2]
        else:
            ToolboxRegistry.debug('No expert results for env', env.grid.config.map_name)
    if cfg.save_debug_svg:
        for env, unroll_steps in envs_with_positive_diffs:
            create_svg(env, unroll_steps)
    logs = [{'map_name': envs[i].grid.config.map_name,
             'gpt_results': gpt_results[i], 
             'fast_expert_results': fast_solver_results_by_map[envs[i].grid.config.map_name], 
             'expert_results': expert_logs[envs[i].grid.config.map_name] if envs[i].grid.config.map_name in expert_logs else "Diff threshold not reached"} for i in range(len(envs))]
    return {'inputs': inputs, 'gt_actions': gt_actions}, logs


def main():
    ToolboxRegistry.setup_logger('DEBUG')

    learnable_algo = MAPFGPTInference(MAPFGPTInferenceConfig(device='cuda', path_to_weights='../weights/model-2M.pt'))
    fast_time_limit = 2
    slow_time_limit = 10
    lacam_lib_path = "../lacam/liblacam.so"
    fast_solver = LacamInference(
        LacamInferenceConfig(time_limit=fast_time_limit, timeouts=[fast_time_limit], lacam_lib_path=lacam_lib_path), )
    solver = LacamInference(
        LacamInferenceConfig(time_limit=slow_time_limit, timeouts=[slow_time_limit], lacam_lib_path=lacam_lib_path))

    env = make_pogema_maze_instance(num_agents=32,
                                    max_episode_steps=256,
                                    map_seed=45,
                                    scenario_seed=45)

    fast_solver_delta(env=env, learnable_algo=learnable_algo, fast_solver=fast_solver, solver=solver,
                      cfg=FastSolverDeltaConfig(save_debug_svg=True))


if __name__ == '__main__':
    main()
