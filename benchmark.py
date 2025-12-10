from pathlib import Path

import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from gpt.inference import LCMAPFInference, LCMAPFInferenceConfig, create_eval_env

PROJECT_NAME = "Benchmark"
BASE_PATH = Path("eval_configs")
EVAL_DIR = Path("eval_results")
EVAL_NAME = "exp_name" 


def ensure_weights(eval_config):
    for algo_name, algo_cfg in eval_config['algorithms'].items():
        ToolboxRegistry.create_algorithm(algo_cfg['name'], **algo_cfg)


def main(disable_wandb=True):
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_eval_env, Environment)
    ToolboxRegistry.register_algorithm("LC-MAPF", LCMAPFInference, LCMAPFInferenceConfig)

    folder_names = [
         "01-random",
         "02-mazes",
         "03-warehouse",
         "04-movingai",
         "05-puzzles",
        # "06-pathfinding"
    ]

    evaluation_configs = {}

    for folder in folder_names:
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        evaluation_configs[folder] = evaluation_config
        
    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        with open(maps_path, "r") as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

        evaluation_config = evaluation_configs[folder]

        # ensuring model weights are downloaded
        ensure_weights(evaluation_config)

        #eval_dir = BASE_PATH / folder / EVAL_NAME
        eval_dir = EVAL_DIR / folder / EVAL_NAME
        eval_dir.mkdir(parents=True, exist_ok=True)

        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)


if __name__ == "__main__":
    main()
