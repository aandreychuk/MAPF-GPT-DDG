/*
 * Implementation of LaCAM*
 *
 * references:
 * LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. AAAI Conf. on Artificial Intelligence (AAAI). 2023.
 *
 * Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. Int. Joint Conf. on Artificial Intelligence (IJCAI). 2023.
 *
 * Engineering LaCAM*: Towards Real-Time, Large-Scale, and Near-Optimal
 * Multi-Agent Pathfinding. Keisuke Okumura. Proc. Int. Conf. on Autonomous
 * Agents and Multiagent Systems. 2024.
 */
#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "utils.hpp"
#include <unordered_map>
#include <queue>
#include <vector>
#include <unordered_set>

// low-level search node
struct LNode {
  std::vector<int> who;
  Vertices where;
  const uint depth;
  LNode();
  LNode(LNode *parent, int i, Vertex *v);  // who and where
  ~LNode();
};

struct HNode;
struct CompareHNodePointers {  // for determinism
  bool operator()(const HNode *lhs, const HNode *rhs) const;
};

// high-level search node
struct HNode {
  const Config Q;
  HNode *parent;
  std::set<HNode *, CompareHNodePointers> neighbors;

  // cost
  int g;
  int h;
  int f;
  int depth;

  std::vector<int> g_values;
  std::vector<float> priorities;
  std::vector<int> order;
  std::queue<LNode *> search_tree;

  HNode(Config _C, DistTable *D, const Instance *ins, HNode *_parent = nullptr, int _g = 0,
        int _h = 0);
  ~HNode();
};
using HNodes = std::vector<HNode *>;

struct LaCAM {
  const Instance *ins;
  DistTable *D;
  const Deadline *deadline;
  const int seed;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution
  const int verbose;

  // solver utils
  PIBT pibt;
  HNode *H_goal;
  std::deque<HNode *> OPEN;
  int loop_cnt;
  
  // Run-level statistics
  uint64_t stats_generated = 0; // number of freshly generated nodes this run
  uint64_t stats_merged = 0;    // number of nodes rebuilt from cached subtree this run
  
  // Priority queue for f-value based expansion (used in solve_from_config)
  struct HNodeComparator {
    bool operator()(const HNode* a, const HNode* b) const {
      if (a->f != b->f) return a->f > b->f;  // Lower f-value first
      return a->h > b->h;  // Tie-break with lower h-value
    }
  };
  std::priority_queue<HNode*, std::vector<HNode*>, HNodeComparator> OPEN_PQ;

  // persistent cache (optional)
  bool reuse_cache = false;  // if true, EXPLORED/GC_HNodes persist across solves
  std::unordered_map<Config, HNode *, ConfigHasher> EXPLORED_CACHE;
  HNodes GC_HNodes_CACHE;
  const Instance *cache_ins = nullptr;  // to validate cache compatibility
  DistTable *cache_D = nullptr;

  // solution suffix cache (solution graph): for any config, store best successor towards goal
  std::unordered_map<Config, Config, ConfigHasher> BEST_SUCCESSOR_CACHE;
  std::unordered_map<Config, int, ConfigHasher> REMAINING_G_CACHE; // remaining g to goal from this config
  // full solution tree: all known successors from solutions (may have multiple)
  std::unordered_map<Config, std::vector<Config>, ConfigHasher> SOL_TREE_SUCCESSORS;
  // track current best solution path nodes and its cost
  std::unordered_set<Config, ConfigHasher> BEST_SOLUTION_SET;
  int BEST_SOLUTION_G = std::numeric_limits<int>::max();
  // solution cache versioning to avoid splicing early in the same run
  uint64_t current_run_version = 0;
  uint64_t best_solution_version = 0; // version of BEST_SOLUTION_SET

  // Hyperparameters
  static bool ANYTIME;
  static float RANDOM_INSERT_PROB1;
  static float RANDOM_INSERT_PROB2;

  LaCAM(const Instance *_ins, DistTable *_D, int _verbose = 0,
        const Deadline *_deadline = nullptr, int _seed = 0);
  ~LaCAM();
  Solution solve(const Config *custom_start = nullptr, float deadline_seconds = -1.0f);
  // Control cache reuse lifecycle
  void set_reuse_cache(bool enable) { reuse_cache = enable; }
  void clear_cache();
  // Reset performance counters for fresh timing
  void reset_counters();
  // Helper function to push nodes in prioritized order
  void push_prioritized(const std::vector<HNode*>& nodes);
  bool set_new_config(HNode *S, LNode *M, Config &Q_to);
  void rewrite(HNode *H_from, HNode *H_to);
  int get_g_val(HNode *H_parent, const Config &Q_to);
  int get_h_val(const Config &Q);
  int get_edge_cost(const Config &Q1, const Config &Q2);
  
  // Cache merging utilities
  void merge_cached_subtree(HNode *H_new, HNode *H_cached, 
                           std::unordered_map<Config, HNode *, ConfigHasher> &EXPLORED,
                           HNodes &GC_HNodes,
                           std::unordered_set<Config, ConfigHasher> &merged_configs);

  // Solution-suffix utilities
  void cache_solution_suffix_from_goal(HNode *H_goal);
  void build_spliced_solution_from_cached(HNode *prefix_end,
                                          Solution &out_solution);

  // utilities
  template <typename... Body>
  void solver_info(const int level, Body &&...body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  loop_cnt:" << std::setw(8) << loop_cnt << "\t";
    info(level, verbose, (body)...);
  }
};
