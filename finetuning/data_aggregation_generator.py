from copy import deepcopy
from multiprocessing import Pool

from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from lacam.inference import LacamInference, LacamInferenceConfig
#from rhcr_cpp.rhcr import RHCRInference, RHCRConfig
from pogema_toolbox.run_episode import run_episode
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import BaseModel

from finetuning.filter_data import filter_data

from utils.data_collection import fill_actions_with_solver
from finetuning.scenario_generators import make_pogema_maze_instance

from utils.wrappers import UnrollWrapper
from pogema.wrappers.metrics import RuntimeMetricWrapper


class DataAggregationConfig(BaseModel):
    steps_delta: int = 8
    steps_saved: int = 8

def run_solver(env, unroll_steps, steps_saved):
    env = deepcopy(env)
    solver = LacamInference(LacamInferenceConfig(time_limit=10, timeouts=[10]))
    env.set_unroll_steps(unroll_steps)
    chosen_agents = list(range(env.grid_config.num_agents))
    ToolboxRegistry.debug(f'Collecting data from step {unroll_steps} to {unroll_steps + steps_saved}')
    input, gt_action, metrics = fill_actions_with_solver(env, unroll_steps, steps_saved, chosen_agents, solver)
    return input, gt_action, metrics

def data_aggregation(env, learnable_algo, cfg: DataAggregationConfig):
    env = UnrollWrapper(env)
    env = RuntimeMetricWrapper(env)

    inputs = []
    gt_actions = []
    gpt_results = run_episode(env, learnable_algo)
    logs = {'map_name': env.grid.config.map_name, 'gpt_results': gpt_results, 'expert_results': []}
    episode_length = gpt_results['ep_length']

    with Pool(processes=8) as pool: # limit the number of workers to avoid overloading the CPU
        unroll_steps_list = range(0, episode_length, cfg.steps_delta)
        results = pool.starmap(run_solver, [(env, unroll_steps, cfg.steps_saved, cfg.on_target) for unroll_steps in unroll_steps_list])

    for input, gt_action, metrics in results:
        if input is not None:
            inputs.extend(input)
            gt_actions.extend(gt_action)
        if metrics is not None:
            logs['expert_results'].append(metrics)
    return filter_data(inputs, gt_actions), logs


def main():
    ToolboxRegistry.setup_logger('DEBUG')

    learnable_algo = MAPFGPTInference(MAPFGPTInferenceConfig(device='cuda', path_to_weights='../weights/model-2M.pt'))
    slow_time_limit = 10
    lacam_lib_path = "../lacam/liblacam.so"
    solver = LacamInference(
        LacamInferenceConfig(time_limit=slow_time_limit, timeouts=[slow_time_limit], lacam_lib_path=lacam_lib_path))

    env = make_pogema_maze_instance(num_agents=32,
                                    max_episode_steps=128,
                                    map_seed=45,
                                    scenario_seed=45)

    data_aggregation(env=env, learnable_algo=learnable_algo, solver=solver,
                     cfg=DataAggregationConfig())


if __name__ == '__main__':
    main()
