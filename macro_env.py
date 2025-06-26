import numpy as np
from gymnasium import Wrapper

class MAPFGPTObservationWrapper(Wrapper):
    def __init__(self, env, observation_generator):
        super().__init__(env)
        self.observation_generator = observation_generator
        

    def reset(self):
        observations, infos = self.env.reset()
        self.observation_generator.create_agents([o["global_xy"] for o in observations], [o["global_target_xy"] for o in observations])
        return self.observation_generator.generate_observations(), infos
    
    def step(self, actions):
        desired_actions = actions.copy()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self.observation_generator.update_agents([o["global_xy"] for o in observations], [o["global_target_xy"] for o in observations], desired_actions)
        observations = self.observation_generator.generate_observations()
        return observations, rewards, terminated, truncated, infos
    
    def get_inner_env(self):
        return self.env

class PogemaMacroEnvironment:
    def __init__(self, environments):
        self.environments = environments
        self.num_agents_per_env = None
        self.active_status = [True] * len(environments)  # Tracks which environments are still active
        self.last_observations = [None] * len(environments)  # Stores the last observations for inactive environments
        self.metrics_info = [{} for _ in environments]
        

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = [], [], [], [], []

        start_idx = 0
        for i, (env, num_agents) in enumerate(zip(self.environments, self.num_agents_per_env)):
            if self.active_status[i]:  # Process only active environments
                env_actions = actions[start_idx:start_idx + num_agents]
                obs, reward, term, trunc, info = env.step(env_actions)
                self.active_status[i] = not (all(term) or all(trunc))  # Mark inactive if terminated or truncated
                if all(term) or all(trunc):
                    info[0]['metrics']['map_name'] = env.grid_config.map_name
                    self.metrics_info[i] = info
                self.last_observations[i] = obs  # Store the observation for inactive reuse
            else:
                # Use last observation and set reward, terminated, and truncated to default values
                obs = self.last_observations[i]
                reward = np.zeros(num_agents)
                term = [True] * num_agents
                trunc = [True] * num_agents
                info = self.metrics_info[i]
                

            start_idx += num_agents

            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)

        return (
            np.concatenate(observations),
            np.concatenate(rewards),
            np.concatenate(terminated),
            np.concatenate(truncated),
            infos,
        )

    def reset(self):
        observations = []
        self.active_status = [True] * len(self.environments)  # Reset all environments to active
        for i, env in enumerate(self.environments):
            obs, info = env.reset()
            self.last_observations[i] = obs  # Store the initial observation for reuse
            observations.append(obs)
        self.num_agents_per_env = [env.grid.config.num_agents for env in self.environments]

        return np.concatenate(observations), {}