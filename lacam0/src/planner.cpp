#include "../include/planner.hpp"

Solution solve(const Instance &ins, int verbose, const Deadline *deadline,
               int seed)
{
  // distance table
  auto D = DistTable(ins);
  info(1, verbose, deadline,
       "set distance table, multi-thread init: ", DistTable::MULTI_THREAD_INIT);

  // lacam
  auto lacam = LaCAM(&ins, &D, verbose, deadline, seed);
  info(1, verbose, deadline, "start lacam");
  return lacam.solve();
}

SolutionStats compute_solution_stats(const Instance &ins, const Solution &solution)
{
  SolutionStats out{0, 0};
  if (solution.empty()) return out;
  const size_t N = ins.N;
  // depth is number of transitions
  const int T = (int)solution.size();
  out.depth = T - 1;
  const auto &goals = ins.goals;

  // For each agent, find from the end the last time it is NOT at goal.
  // Then g_i = last_not_goal_index + 1 (first timestep afterward where it must be at goal and stay).
  // If the agent is at goal for entire horizon, g_i = 0.
  // If the agent is not at goal even at the end, g_i = T (i.e., never settled).
  for (size_t i = 0; i < N; ++i) {
    int last_not_goal = -1;
    for (int t = T - 1; t >= 0; --t) {
      if (solution[t][i] != goals[i]) { last_not_goal = t; break; }
    }
    int g_i = last_not_goal + 1;  // becomes 0 if always at goal; becomes T if never at goal at end
    out.g += g_i;
  }
  return out;
}
