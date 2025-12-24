import numpy as np
from typing import TypeAlias
from dataclasses import dataclass, field
from collections import deque

# y, x
Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
Config: TypeAlias = list[Coord]
Configs: TypeAlias = list[Config]


def is_valid_coord(grid, coord):
    y, x = coord
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == 0

def get_neighbors(grid, coord):
    # coord: y, x
    neigh = []
    move_idx = []
    mask = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh, move_idx, mask

    y, x = coord
    moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(moves):
        if is_valid_coord(grid, (y + dy, x + dx)):
            neigh.append((y + dy, x + dx))
            move_idx.append(i)
            mask.append(True)
        else:
            mask.append(False)

    return neigh, move_idx, mask

@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    Q: deque = field(init=False)  # lazy distance evaluation
    table: np.ndarray = field(init=False)  # distance matrix

    def __post_init__(self):
        self.Q = deque([self.goal])
        self.table = np.full(self.grid.shape, self.grid.size, dtype=int)
        self.table[self.goal] = 0

    def get(self, target: Coord) -> int:
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if self.table[target] < self.table.size:
            return self.table[target]

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            n, _, _ = (get_neighbors(self.grid, u))
            for v in n:
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.grid.size


class PIBTInstance:
    def __init__(self, grid, starts, goals, sampling_method, seed=0):
        
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

        # Calculating initial priorities
        self.priorities: list[float] = []
        for i in range(self.N):
            self.priorities.append(
                self.dist_tables[i].get(self.starts[i]) / self.grid.size
            )

        self.state = self.starts
        self.reached_goals = False
        self.sampling_method = sampling_method

    def _update_priorities(self):
        flg_fin = True
        for i in range(self.N):
            if self.state[i] != self.goals[i]:
                flg_fin = False
                self.priorities[i] += 1
            else:
                self.priorities[i] -= np.floor(self.priorities[i])
        self.reached_goals = flg_fin

    def funcPIBT(
        self, Q_from, Q_to, i: int, transition_probabilities, pibt_ids
    ) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C, move_idx, mask = get_neighbors(self.grid, Q_from[i])

        if (pibt_ids is not None) and (pibt_ids[i] is not None):
            ids = pibt_ids[i]
        elif self.sampling_method == "deterministic":
            ids = np.arange(len(C))
            self.rng.shuffle(ids)  # tie-breaking, randomize
            ids = sorted(
                ids,
                key=lambda u: transition_probabilities[i][move_idx[u]],
                reverse=True,
            )
        elif self.sampling_method == "probabilistic":
            try:
                cur_trans_probs = transition_probabilities[i][mask]
                cur_trans_probs = cur_trans_probs / np.sum(cur_trans_probs)

                ids = np.arange(len(C))
                ids = self.rng.choice(
                    ids, size=len(C), replace=False, p=cur_trans_probs, shuffle=False
                )
            except:
                # Potential error due to zeroing of some probs
                cur_trans_probs = transition_probabilities[i][mask]
                EPSILON = 1e-6

                cur_trans_probs = cur_trans_probs + EPSILON
                cur_trans_probs = cur_trans_probs / np.sum(cur_trans_probs)

                ids = np.arange(len(C))
                ids = self.rng.choice(
                    ids, size=len(C), replace=False, p=cur_trans_probs, shuffle=False
                )
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}.")

        # vertex assignment
        for id in ids:
            v = C[id]
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.actions[i] = move_idx[id]
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j, transition_probabilities, pibt_ids))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.actions[i] = 0
        self.occupied_nxt[Q_from[i]] = i
        return False

    def _step(self, Q_from, priorities, transition_probabilities, pibt_ids=None):
        # setup
        N = len(Q_from)
        Q_to = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i, transition_probabilities, pibt_ids)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def step(self, transition_probabilities, pibt_ids=None):
        self.actions = np.zeros(self.N, dtype=np.int)
        if self.reached_goals:
            return self.actions
        self.state = self._step(
            self.state, self.priorities, transition_probabilities, pibt_ids
        )
        self._update_priorities()
        return self.actions

    def run(self, max_timestep=1000):
        raise AssertionError("This method should not be run.")


class PIBTInstanceDist(PIBTInstance):
    def __init__(self, grid, starts, goals, sampling_method, seed=0):
        super().__init__(grid, starts, goals, sampling_method, seed)
        self._update_priorities()

    def _update_priorities(self):
        # Setting priorities based on distance to goal
        for i in range(self.N):
            sx, sy = self.state[i]
            gx, gy = self.goals[i]
            self.priorities[i] = abs(gx - sx) + abs(gy - sy)


class PIBTCollisionShielding:
    def __init__(
        self,
        obstacles,
        starts,
        goals,
        seed=0,
        do_sample=True,
        dist_priorities=False,
    ):
        sampling_method = "probabilistic"
        if not do_sample:
            sampling_method = "deterministic"
        self.sampling_method = sampling_method

        starts = [tuple(s) for s in starts]
        goals = [tuple(g) for g in goals]

        if dist_priorities:
            self.pibt_instance = PIBTInstanceDist(
                grid=obstacles,
                starts=starts,
                goals=goals,
                seed=seed,
                sampling_method=sampling_method,
            )
        else:
            self.pibt_instance = PIBTInstance(
                grid=obstacles,
                starts=starts,
                goals=goals,
                seed=seed,
                sampling_method=sampling_method,
            )

    def __call__(self, actions):
        if self.sampling_method == "probabilistic":
            # actions = torch.nn.functional.softmax(actions, dim=-1)
            actions = actions.detach().cpu().numpy()
        actions = self.pibt_instance.step(actions)
        return actions