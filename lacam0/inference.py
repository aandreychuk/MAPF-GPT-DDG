from typing import Literal
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase
from pogema import GridConfig
import cppimport.import_hook
from lacam0.lacam_py import LaCAM


class LaCAM0InferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["LaCAM0"] = "LaCAM0"
    time_limit: float = 10
    multi_thread_dist_init: bool = False
    anytime: bool = True
    pibt_swap: bool = True
    pibt_hindrance: bool = True


class LaCAM0Inference:
    def __init__(self, cfg: LaCAM0InferenceConfig, net=None):
        self.cfg: LaCAM0InferenceConfig = cfg
        self.solution = None
        self.step = 0

    def act(self, observations):
        if self.solution is None:
            lacam = LaCAM()
            grid = observations[0]["global_obstacles"].copy().astype(int).tolist()
            starts = [obs["global_xy"] for obs in observations]
            goals = [obs["global_target_xy"] for obs in observations]
            lacam.init(grid, starts, goals,
                                       time_limit_sec=self.cfg.time_limit,
                                       multi_thread_dist_init=self.cfg.multi_thread_dist_init,
                                       anytime=self.cfg.anytime,
                                       pibt_swap=self.cfg.pibt_swap,
                                       pibt_hindrance=self.cfg.pibt_hindrance,
                                       verbose=3)
            self.solution = lacam.get_solution()
        moves = {tuple(move):i for i, move in enumerate(GridConfig().MOVES)}
        actions = []
        for i in range(len(observations)):
            if len(self.solution[i]) - 1 > self.step:
                p0 = self.solution[i][self.step]
                p1 = self.solution[i][self.step+1]
                actions.append(moves[p1[1] - p0[1], p1[0] - p0[0]])
            else:
                actions.append(0)
        self.step += 1
        return actions
            

    def reset_states(self):
        self.solution = None
        self.step = 0
