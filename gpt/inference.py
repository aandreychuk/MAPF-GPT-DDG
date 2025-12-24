from pathlib import Path
from typing import Literal, Optional
from dataclasses import fields

import torch
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.run_episode import run_episode
from pydantic import Extra

from gpt.dmm_gnn import DMMGNN, DMMGNNConfig
import cppimport.import_hook
from gpt.observation_generator_directions import ObservationGeneratorDirections
from pogema import pogema_v0, AnimationMonitor, AnimationConfig
from pogema.wrappers.metrics import RuntimeMetricWrapper, Wrapper
from pogema_toolbox.create_env import MultiMapWrapper
from gpt.cs_pibt import PIBTCollisionShielding

class LCMAPFInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["LC-MAPF"] = "LC-MAPF"
    path_to_weights: Optional[str] = "weights/ckpt.pt"
    device: str = "cuda"
    context_size: int = 121
    num_rounds: int = 2
    use_cs_pibt: bool = True


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
        self.observation_generator = None
        self.last_actions = None
        self.pibt_collision_shielding = None
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
        config_dict = checkpoint.get("model_args", {})
        
        # Override n_comm_rounds with the value from inference config
        config_dict["n_comm_rounds"] = cfg.num_rounds
        
        # Filter to only include valid GNN config fields (DMMGNNConfig will use defaults for missing values)
        valid_fields = {f.name for f in fields(DMMGNNConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        config = DMMGNNConfig(**filtered_config)
        self.net = DMMGNN(config)
        self.net.load_state_dict(model_state_dict, strict=False)
        self.net.to(self.cfg.device)
        self.net.eval()

    def act(self, observations, custom_num_agents=None, infos=None):
        positions = [obs["global_xy"] for obs in observations]
        goals = [obs["global_target_xy"] for obs in observations]
        if self.pibt_collision_shielding is None:
            self.pibt_collision_shielding = PIBTCollisionShielding(
                obstacles=observations[0]['global_obstacles'].copy().astype(int),
                starts=positions,
                goals=goals,
                seed=0,
                do_sample=True,
                dist_priorities=True
            )
        if self.observation_generator is None:
            self.observation_generator = ObservationGeneratorDirections(observations[0]['global_obstacles'].copy().astype(int).tolist(), self.cfg.context_size)
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
        rel_coords = self.observation_generator.get_agents_relative_coords()
        agents_rel_coords = torch.tensor(rel_coords, dtype=torch.long, device=self.cfg.device)
        agents_rel_coords = agents_rel_coords[None, :, :]

        if self.cfg.use_cs_pibt:
            action_probs = self.net.get_action_probs(tensor_obs, agent_chat_ids, agents_rel_coords)
            actions = self.pibt_collision_shielding(action_probs)
        else:
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
        self.pibt_collision_shielding = None


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
