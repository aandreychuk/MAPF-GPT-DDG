# noinspection PyUnresolvedReferences
from gpt.observation_generator import ObservationGenerator, InputParameters
from pogema_toolbox.registry import ToolboxRegistry


def fill_actions_with_solver(env, start_step, steps_to_collect, chosen_agents, expert_algo=None):
    if expert_algo is not None:
        expert_algo.reset_states()
    observations, *_ = env.reset()
    observation_generator = ObservationGenerator(observations[0]["global_obstacles"].copy().astype(int).tolist(),
                                                 InputParameters(20, 13, 5, 256, 5, 5, 64, False))
    positions = [obs["global_xy"] for obs in observations]
    goals = [obs["global_target_xy"] for obs in observations]
    observation_generator.create_agents(positions, goals)
    for i in range(5):
        observation_generator.update_agents(positions, goals, env.get_actions_at_step(start_step - 5 + i))
    inputs = []
    gt_actions = []
    for i in range(steps_to_collect):
        input = observation_generator.generate_observations()
        if expert_algo is not None:
            actions = expert_algo.act(observations)
            if not expert_algo.solved:
                return None, None, {'ISR': 0.0, 'CSR': 0.0, 'ep_length': 256, 'SoC': -1, 'makespan': 256, 'runtime': 10} # placeholder metrics if expert algo is failed
        else:
            actions = env.get_actions_at_step(start_step + i) # if no expert algo => use the actions from MAPF-GPT
        for agent_idx in chosen_agents:
            inputs.append(input[agent_idx])
            gt_actions.append(actions[agent_idx])
        observations, rew, terminated, truncated, infos = env.step(actions)
        if all(terminated) or all(truncated):
            break

        positions = [obs["global_xy"] for obs in observations]
        goals = [obs["global_target_xy"] for obs in observations]
        observation_generator.update_agents(positions, goals, actions)
    ToolboxRegistry.debug(f'Tagged {len(inputs)} steps with expert data starting from step {start_step}')
    if expert_algo is not None and expert_algo.cfg.name == "LaCAM" and not (all(terminated) or all(truncated)):
        while True:
            input = observation_generator.generate_observations()
            actions = expert_algo.act(observations)
            observations, rew, terminated, truncated, infos = env.step(actions)
            if all(terminated) or all(truncated):
                break
    return inputs, gt_actions, infos[0]['metrics']