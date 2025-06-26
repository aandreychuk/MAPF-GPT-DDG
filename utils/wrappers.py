from gymnasium import Wrapper


class UnrollWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self._unroll_steps = None
        self._recorded_actions = []
        self._recording_episode = None

    def step(self, action):
        if self._recording_episode:
            self._recorded_actions.append(action)
        return self.env.step(action)

    def get_actions_at_step(self, step):
        if step < 0:
            return [-1 for _ in range(self.env.num_agents)]
        elif step < len(self._recorded_actions):
            return self._recorded_actions[step]
        else:
            raise ValueError(f'Step {step} is out of range')

    def set_unroll_steps(self, num_steps):
        self._unroll_steps = num_steps

    def reset(self, seed=None, **kwargs):
        self._recording_episode = True if self._recording_episode is None else False
        if seed is None:
            seed = self.env.grid_config.seed
        obs, infos = self.env.reset(seed=seed)
        if self.env.grid_config.on_target == "restart":
            targets_xy = [o['global_lifelong_targets_xy'] for o in obs]
            max_episode_steps = obs[0]['max_episode_steps']
        if self._unroll_steps and self._recorded_actions:
            for idx in range(self._unroll_steps):
                obs, rew, terminated, truncated, infos = self.env.step(self._recorded_actions[idx])
                if self.env.grid_config.on_target == "restart":
                    obs[0]['max_episode_steps'] = max_episode_steps
                    for i in range(len(obs)):
                        if obs[i]['global_target_xy'] != targets_xy[i][0]:
                            targets_xy[i] = targets_xy[i][1:]
                        obs[i]['global_lifelong_targets_xy'] = targets_xy[i]
        return obs, infos
