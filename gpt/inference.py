from pathlib import Path
from typing import Literal, Optional

import torch
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.run_episode import run_episode
from pydantic import Extra

#from gpt.model import LCMAPF, Config
from gpt.dmm_gnn import DMMGNN, DMMGNNConfig
from gpt.dmm_comm import DMM, DMMConfig
import cppimport.import_hook
from gpt.observation_generator import InputParameters, ObservationGenerator
from gpt.observation_generator_directions import ObservationGeneratorDirections
from pogema import pogema_v0, AnimationMonitor, AnimationConfig
from pogema.wrappers.metrics import RuntimeMetricWrapper, Wrapper
from pogema_toolbox.create_env import MultiMapWrapper


class LCMAPFInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["LC-MAPF"] = "LC-MAPF"
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/ckpt.pt"
    device: str = "cuda"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = ''
    grid_step: int = 64
    save_cost2go: bool = False
    num_rounds: int = 2
    observation_type = 'cost2go'
    model_type: Optional[Literal["dmm", "gnn"]] = None  # None = auto-detect from checkpoint


def strip_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """
    strips the given prefix from the keys in the state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class LCMAPFInference:
    def __init__(self, cfg: LCMAPFInferenceConfig):
        self.cfg: LCMAPFInferenceConfig = cfg
        self.input_parameters = InputParameters(
            cfg.cost2go_value_limit,
            cfg.num_agents,
            cfg.num_previous_actions,
            cfg.context_size,
            cfg.cost2go_radius,
            cfg.agents_radius
        )
        self.observation_generator = None
        self.last_actions = None
        path_to_weights = Path(self.cfg.path_to_weights)

        if self.cfg.device in ['mps',
                               'cuda'] and not torch.cuda.is_available() if self.cfg.device == 'cuda' else not torch.backends.mps.is_available():
            ToolboxRegistry.warning(f'{self.cfg.device} is not available, using cpu instead!')
            self.cfg.device = 'cpu'

        self.torch_generator = torch.Generator(device=self.cfg.device)
        self.torch_generator.manual_seed(0)

        checkpoint = torch.load(Path(self.cfg.path_to_weights),
                                map_location=self.cfg.device,
                                weights_only=False)
       
        model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
        config_dict = checkpoint.get("model_args")
        config_dict["n_comm_rounds"] = cfg.num_rounds
        
        # Determine model type: use config if provided, otherwise auto-detect from checkpoint
        model_type = cfg.model_type
        if model_type is None:
            # Auto-detect: check if checkpoint has GNN-specific keys
            has_gnn_keys = any("graph_encoder" in k or "graph_decoder" in k or "gnn_layers" in k 
                              for k in model_state_dict.keys())
            model_type = "gnn" if has_gnn_keys else "dmm"
        
        # Instantiate appropriate model
        if model_type == "gnn":
            config = DMMGNNConfig(**config_dict)
            self.net = DMMGNN(config)
        else:  # model_type == "dmm"
            config = DMMConfig(**config_dict)
            self.net = DMM(config)
        self.net.load_state_dict(model_state_dict, strict=False)
        self.net.to(self.cfg.device)
        self.net.eval()

    def act(self, observations, custom_num_agents=None, infos=None):
        positions = [obs["global_xy"] for obs in observations]
        goals = [obs["global_target_xy"] for obs in observations]

        state_size = len(positions)
        if self.observation_generator is None:
            if self.cfg.observation_type == 'cost2go':
                #import cppimport.import_hook
                #from gpt.observation_generator import ObservationGenerator
                self.observation_generator = ObservationGenerator(
                    observations[0]["global_obstacles"].copy().astype(int).tolist(),
                    self.input_parameters
                )
            else:
                #import cppimport.import_hook
                #from gpt.observation_generator_directions import ObservationGeneratorDirections
                self.observation_generator = ObservationGeneratorDirections(
                    observations[0]['global_obstacles'].copy().astype(int).tolist(), 
                    5,  # obs_radius 
                    self.cfg.context_size
                )
            self.observation_generator.create_agents(positions, goals)
            self.last_actions = [-1 for _ in range(len(observations))]
        self.observation_generator.update_agents(positions, goals, self.last_actions)
        inputs = self.observation_generator.generate_observations()
        tensor_obs = torch.tensor(inputs, dtype=torch.long, device=self.cfg.device)
        tensor_obs = tensor_obs[None, :, :]

        agent_chat_ids = self.observation_generator.get_agents_ids_in_observations()
        agent_chat_ids = torch.tensor(agent_chat_ids, dtype=torch.long, device=self.cfg.device)
        agent_chat_ids = agent_chat_ids[None, :, :]

        # Get relative coordinates for message-to-FOV alignment (Directions observation)
        agents_rel_coords = None
        if self.cfg.observation_type != 'cost2go':
            rel_coords = self.observation_generator.get_agents_relative_coords()
            agents_rel_coords = torch.tensor(rel_coords, dtype=torch.long, device=self.cfg.device)
            agents_rel_coords = agents_rel_coords[None, :, :]

        actions = torch.squeeze(self.net.act(
            obs=tensor_obs, 
            agent_chat_ids=agent_chat_ids,
            agents_rel_coords=agents_rel_coords
        )).tolist()

        if not isinstance(actions, list):
            actions = [actions]
        self.last_actions = actions.copy()
        return actions

    def reset_states(self):
        self.observation_generator = None
        self.torch_generator.manual_seed(0)


class CollisionsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agent_collisions = 0
        self.obstacle_collisions = 0

    def step(self, actions):
        desired_actions = actions.copy()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        for i in range(len(observations)):
            if actions[i] != desired_actions[i]:
                x = self.env.grid.positions_xy[i][0] + self.env.grid_config.MOVES[desired_actions[i]][0]
                y = self.env.grid.positions_xy[i][1] + self.env.grid_config.MOVES[desired_actions[i]][1]
                if self.env.grid.obstacles[x][y] == 1:
                    self.obstacle_collisions += 1
                else:
                    self.agent_collisions += 1
        if all(terminated) or all(truncated):
            infos[0]["metrics"]["a_collisions"] = self.agent_collisions
            infos[0]["metrics"]["o_collisions"] = self.obstacle_collisions
        return observations, rewards, terminated, truncated, infos
    
    def reset(self, **kwargs):
        self.agent_collisions = 0
        self.obstacle_collisions = 0
        observations, info = self.env.reset(**kwargs)
        return observations, info


def create_eval_env(config):
    env = pogema_v0(grid_config=config)
    env = MultiMapWrapper(env)
    env = CollisionsWrapper(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
    env = RuntimeMetricWrapper(env)
    return env


def main():
    from pogema_toolbox.create_env import Environment

    env = create_eval_env(Environment(size=8, num_agents=2, seed=42, observation_type='MAPF'))
    algo = LCMAPFInference(LCMAPFInferenceConfig(path_to_weights='../weights/ckpt.pt'))
    results = run_episode(env, algo)
    print(results)


if __name__ == '__main__':
    main()
