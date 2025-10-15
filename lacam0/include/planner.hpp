#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "lacam.hpp"
#include "planner.hpp"
#include "post_processing.hpp"
#include "utils.hpp"

Solution solve(const Instance &ins, const int verbose = 0,
               const Deadline *deadline = nullptr, int seed = 0);

struct SolutionStats {
  int g;
  int depth;
};

// Compute g-value and depth for a given solution with respect to ins.goals
SolutionStats compute_solution_stats(const Instance &ins, const Solution &solution);
