#include "../include/lacam.hpp"
#include <queue>
#include <unordered_set>
#include <chrono>

bool LaCAM::ANYTIME = false;
float LaCAM::RANDOM_INSERT_PROB1 = 0.001;
float LaCAM::RANDOM_INSERT_PROB2 = 0.001;

bool CompareHNodePointers::operator()(const HNode *l, const HNode *r) const
{
  const auto N = l->Q.size();
  for (size_t i = 0; i < N; ++i) {
    if (l->Q[i] != r->Q[i]) return l->Q[i]->id < r->Q[i]->id;
  }
  return false;
}

HNode::HNode(Config _Q, DistTable *D, const Instance *ins, HNode *_parent, int _g, int _h)
    : Q(_Q),
      parent(_parent),
      neighbors(),
      g(_g),
      h(_h),
      f(g + h),
      depth(parent == nullptr ? 0 : parent->depth + 1),
      g_values(Q.size()),
      priorities(Q.size()),
      order(Q.size(), 0),
      search_tree()
{
  if (parent != nullptr) parent->neighbors.insert(this);
  search_tree.push(new LNode());
  const int N = Q.size();
  for (int i = 0; i < N; ++i) {
    // set g_values - track when each agent reaches its goal
    if (parent == nullptr) {
      // initialize: all agents start with cost 0
      g_values[i] = 0;
    } else {
      // inherit parent's g_value
      g_values[i] = parent->g_values[i];
      
      // if agent is now at goal and wasn't before, set its cost to current depth
      if (Q[i] == ins->goals[i] && parent->Q[i] != ins->goals[i]) {
        g_values[i] = depth;
      }
      // if agent leaves goal, update its cost to current depth
      else if (Q[i] != ins->goals[i] && parent->Q[i] == ins->goals[i]) {
        g_values[i] = depth;
      }
    }
    
    // set priorities
    if (parent == nullptr) {
      // initialize
      priorities[i] = (float)D->get(i, Q[i]) / 10000;
    } else {
      // dynamic priorities, akin to PIBT
      if (D->get(i, Q[i]) != 0) {
        priorities[i] = parent->priorities[i] + 1;
      } else {
        priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
      }
    }
  }

  // set order
  auto cmp = [&](int i, int j) { return priorities[i] > priorities[j]; };
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), cmp);
}

HNode::~HNode()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}

LNode::LNode() : who(), where(), depth(0) {}

LNode::LNode(LNode *parent, int i, Vertex *v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
{
  who.push_back(i);
  where.push_back(v);
}

LNode::~LNode(){};

LaCAM::LaCAM(const Instance *_ins, DistTable *_D, int _verbose,
             const Deadline *_deadline, int _seed)
    : ins(_ins),
      D(_D),
      deadline(_deadline),
      seed(_seed),
      MT(seed),
      rrd(0, 1),
      verbose(_verbose),
      pibt(ins, D, seed),
      H_goal(nullptr),
      OPEN(),
      loop_cnt(0)
{
}

LaCAM::~LaCAM() {}

Solution LaCAM::solve(const Config *custom_start, float deadline_seconds)
{
  solver_info(1, "LaCAM begins");
  // Fresh counters/state per run
  reset_counters();
  H_goal = nullptr;
  stats_generated = 0;
  stats_merged = 0;
  ++current_run_version;
  const bool use_solution_cache = !BEST_SUCCESSOR_CACHE.empty();
  const bool accumulate_solution_cache = !use_solution_cache; // only the first run accumulates
  
  // Use custom start if provided, otherwise use instance start
  const Config &start_config = (custom_start != nullptr) ? *custom_start : ins->starts;
  
  // Enable anytime mode if deadline is provided and > 0
  if (deadline_seconds > 0.0f) {
    ANYTIME = true;
  } else {
    ANYTIME = false;
  }

  // setup search
  auto EXPLORED = std::unordered_map<Config, HNode *, ConfigHasher>();
  HNodes GC_HNodes;
  // Track which cached configurations have been merged to avoid duplicate merging in this run
  std::unordered_set<Config, ConfigHasher> merged_from_cache_configs;

  // seed from cache if enabled and compatible
  const bool can_reuse = reuse_cache && cache_ins == ins && cache_D == D;
  if (!can_reuse && reuse_cache) {
    // reset cache context if reuse requested but incompatible
    EXPLORED_CACHE.clear();
    GC_HNodes_CACHE.clear();
  }
  cache_ins = ins;
  cache_D = D;

  // insert initial node
  HNode *H_init = nullptr;
  auto it_cached = EXPLORED.find(start_config);
  if (it_cached != EXPLORED.end()) {
    H_init = it_cached->second;
  } else {
    H_init = new HNode(start_config, D, ins);
    EXPLORED[H_init->Q] = H_init;
    GC_HNodes.push_back(H_init);
  }
  OPEN.clear();
  OPEN.push_front(H_init);

  // search loop
  solver_info(2, "search iteration begins");
  // Use deadline if provided, otherwise use instance deadline
  auto start_time = std::chrono::high_resolution_clock::now();
  while (!OPEN.empty() && (deadline_seconds <= 0.0f || 
         std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start_time).count() < deadline_seconds)) {
    ++loop_cnt;

    // random insert
    if (H_goal != nullptr) {
      auto r = rrd(MT);
      if (r < RANDOM_INSERT_PROB2 / 2) {
        OPEN.push_front(H_init);
      } else if (r < RANDOM_INSERT_PROB2) {
        auto H = OPEN[get_random_int(MT, 0, OPEN.size() - 1)];
        OPEN.push_front(H);
      }
    }

    // do not pop here!
    auto H = OPEN.front();  // high-level node
    //std::cout << "H->Q: " << H->Q << std::endl;
    // check uppwer bounds
    if (H_goal != nullptr && H->g >= H_goal->g) {
      OPEN.pop_front();
      solver_info(5, "prune, g=", H->g, " >= ", H_goal->g);
      OPEN.push_front(H_init);
      continue;
    }

    // check goal condition
    if (H_goal == nullptr && is_same_config(H->Q, ins->goals)) {
      H_goal = H;
      solver_info(2, "found solution, g=", H->g, ", depth=", H->depth);
      // Immediately cache this solution path (suffix graph and best tracking) only on first run
      if (accumulate_solution_cache) cache_solution_suffix_from_goal(H_goal);
      if (!ANYTIME) break;
      // In anytime mode, continue searching for better solutions (don't break)
      continue;
    }

    // extract constraints
    if (H->search_tree.empty()) {
      // With priority queue OPEN_PQ, just skip if no low-level expansions
      continue;
    }
    auto L = H->search_tree.front();
    H->search_tree.pop();

    // low level search
    if (L->depth < H->Q.size()) {
      const auto i = H->order[L->depth];
      auto &&C = H->Q[i]->actions;
      std::shuffle(C.begin(), C.end(), MT);  // randomize
      for (auto u : C) H->search_tree.push(new LNode(L, i, u));
    }

    // create successors at the high-level search
    auto Q_to = Config(ins->N, nullptr);
    auto res = set_new_config(H, L, Q_to);
    delete L;
    if (!res) continue;

    // check if we reached a cached solution suffix
    if (use_solution_cache && BEST_SUCCESSOR_CACHE.find(Q_to) != BEST_SUCCESSOR_CACHE.end()) {
      // If Q_to is on the best known solution, we can splice and return immediately
      if (BEST_SOLUTION_SET.find(Q_to) != BEST_SOLUTION_SET.end() && best_solution_version < current_run_version) {
        // Debug: report splice details and estimated costs
        int prefix_g = H->g;
        int remaining_g = -1;
        auto rem_it = REMAINING_G_CACHE.find(H->Q);
        if (rem_it != REMAINING_G_CACHE.end()) remaining_g = rem_it->second;
        solver_info(2, "splice from cached best: prefix_g=", prefix_g,
                    ", remaining_g=", remaining_g,
                    ", total=", (remaining_g >= 0 ? prefix_g + remaining_g : -1),
                    ", run_ver=", current_run_version,
                    ", best_ver=", best_solution_version);
        Solution spliced;
        build_spliced_solution_from_cached(H, spliced);
        solver_info(2, "spliced solution length=", (int)spliced.size());
        return spliced;
      }
      // Otherwise, merge nodes from solution tree into current OPEN to avoid regeneration
      // Push known successors of Q_to in prioritized order
      if (best_solution_version < current_run_version) {
        auto &succs = SOL_TREE_SUCCESSORS[Q_to];
        if (!succs.empty()) {
        std::vector<HNode*> to_push;
        to_push.reserve(succs.size());
        for (auto &Q_succ : succs) {
          // Create a fresh node tied to current parent H
          auto H_new = new HNode(Q_succ, D, ins, H, get_g_val(H, Q_succ), get_h_val(Q_succ));
          EXPLORED[H_new->Q] = H_new;
          GC_HNodes.push_back(H_new);
          to_push.push_back(H_new);
          ++stats_merged;
        }
        if (!to_push.empty()) {
          solver_info(3, "inject successors from solution tree: ", (int)to_push.size());
          push_prioritized(to_push);
        }
        }
      }
    }

    // check explored list
    auto iter = EXPLORED.find(Q_to);
    if (iter == EXPLORED.end()) {
      // new one -> insert
      auto H_new = new HNode(Q_to, D, ins, H, get_g_val(H, Q_to), get_h_val(Q_to));
      OPEN.push_front(H_new);
      EXPLORED[H_new->Q] = H_new;
      GC_HNodes.push_back(H_new);
      ++stats_generated;

      // If this configuration exists in cache, merge its subtree with explicit recomputation
      if (can_reuse) {
        auto cache_iter = EXPLORED_CACHE.find(Q_to);
        if (cache_iter != EXPLORED_CACHE.end()) {
          // Avoid redundant merges of the same root config
          if (merged_from_cache_configs.find(Q_to) == merged_from_cache_configs.end()) {
            auto H_cached = cache_iter->second;
            merge_cached_subtree(H_new, H_cached, EXPLORED, GC_HNodes, merged_from_cache_configs);
            merged_from_cache_configs.insert(Q_to);
            //solver_info(3, "merged cached configuration and subtree");
          }
        }
      }
    } else {
      // known configuration
      auto H_known = iter->second;
      rewrite(H, H_known);

      if (rrd(MT) >= RANDOM_INSERT_PROB1) {
        OPEN.push_front(iter->second);  // usual
      } else {
        solver_info(3, "random restart");
        OPEN.push_front(H_init);  // sometimes
      }
    }
  }

  // backtrack
  Solution solution;
  {
    auto H = H_goal;
    while (H != nullptr) {
      solution.push_back(H->Q);
      H = H->parent;
    }
    std::reverse(solution.begin(), solution.end());
  }
  // Cache solution suffix graph if a goal was found
  if (!solution.empty()) {
    cache_solution_suffix_from_goal(H_goal);
  }

  // solution
  if (solution.empty()) {
    if (OPEN.empty()) {
      solver_info(2, "fin. unsolvable instance");
    } else {
      solver_info(2, "fin. reach time limit");
    }
  } else {
    if (OPEN.empty()) {
      solver_info(2, "fin. optimal solution, g=", H_goal->g,
                  ", depth=", H_goal->depth);
    } else {
      solver_info(2, "fin. suboptimal solution, g=", H_goal->g,
                  ", depth=", H_goal->depth);
    }
  }

  // end processing
  if (reuse_cache) {
    // persist cache for next runs, but only if configs don't already exist
    for (auto &pair : EXPLORED) {
      if (EXPLORED_CACHE.find(pair.first) == EXPLORED_CACHE.end()) {
        EXPLORED_CACHE[pair.first] = pair.second;
      }
    }
    for (auto H : GC_HNodes) {
      // Only add to cache if this config wasn't already cached
      if (EXPLORED_CACHE.find(H->Q) == EXPLORED_CACHE.end()) {
        GC_HNodes_CACHE.push_back(H);
      }
    }
  } else {
    for (auto &&H : GC_HNodes) delete H;  // memory management
  }

  // Print run-level statistics
  solver_info(2, "stats: generated=", stats_generated, ", merged=", stats_merged);

  return solution;
}

void LaCAM::cache_solution_suffix_from_goal(HNode *goal)
{
  // Build suffix pointers from each config on the path to its next config
  std::vector<HNode*> path;
  auto H = goal;
  while (H != nullptr) {
    path.push_back(H);
    H = H->parent;
  }
  std::reverse(path.begin(), path.end());
  // Compute remaining g from each node to goal
  int remaining = 0;
  for (int i = (int)path.size() - 1; i >= 0; --i) {
    REMAINING_G_CACHE[path[i]->Q] = remaining;
    if (i > 0) {
      // edge cost equals sum of agent deltas defined by get_edge_cost
      remaining += get_edge_cost(path[i-1]->Q, path[i]->Q);
    }
  }
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    BEST_SUCCESSOR_CACHE[path[i]->Q] = path[i+1]->Q;
    SOL_TREE_SUCCESSORS[path[i]->Q].push_back(path[i+1]->Q);
  }
  // update best solution record
  int g_total = path.back()->g;
  if (g_total < BEST_SOLUTION_G) {
    BEST_SOLUTION_G = g_total;
    BEST_SOLUTION_SET.clear();
    for (auto *node : path) BEST_SOLUTION_SET.insert(node->Q);
    best_solution_version = current_run_version;
  }
}

void LaCAM::build_spliced_solution_from_cached(HNode *prefix_end,
                                               Solution &out_solution)
{
  // Build prefix
  std::vector<Config> prefix;
  auto H = prefix_end;
  while (H != nullptr) {
    prefix.push_back(H->Q);
    H = H->parent;
  }
  std::reverse(prefix.begin(), prefix.end());
  // Append cached suffix if exists starting from last prefix config
  if (!prefix.empty()) {
    auto cur = prefix.back();
    while (BEST_SUCCESSOR_CACHE.find(cur) != BEST_SUCCESSOR_CACHE.end()) {
      cur = BEST_SUCCESSOR_CACHE[cur];
      prefix.push_back(cur);
      // Stop if we loop defensively (shouldn't happen if cache is a tree)
      if (prefix.size() > 1000000) break;
    }
  }
  out_solution = std::move(prefix);
}

void LaCAM::clear_cache()
{
  for (auto &&H : GC_HNodes_CACHE) delete H;
  GC_HNodes_CACHE.clear();
  EXPLORED_CACHE.clear();
  cache_ins = nullptr;
  cache_D = nullptr;
}

void LaCAM::reset_counters()
{
  loop_cnt = 0;
  // Note: deadline is external and not reset here
  // If you need to reset timing, create a new LaCAM instance or pass a new deadline
}

// Helper function to push nodes in prioritized order (lower f-value first)
void LaCAM::push_prioritized(const std::vector<HNode*>& nodes)
{
  // Sort nodes by f-value (and h-value for ties)
  std::vector<HNode*> sorted_nodes = nodes;
  std::sort(sorted_nodes.begin(), sorted_nodes.end(), [](HNode* a, HNode* b) {
    if (a->f != b->f) return a->f < b->f;
    return a->h < b->h;
  });
  
  // Push all nodes to deque (best first)
  for (auto H : sorted_nodes) {
    OPEN.push_front(H);
  }
}

void LaCAM::merge_cached_subtree(HNode *H_new, HNode *H_cached, 
                                std::unordered_map<Config, HNode *, ConfigHasher> &EXPLORED,
                                HNodes &GC_HNodes,
                                std::unordered_set<Config, ConfigHasher> &merged_configs)
{
  // Do not attach cached neighbor pointers directly; we'll rebuild nodes explicitly
  // Track visited cached nodes to avoid processing duplicates
  std::unordered_set<HNode*> visited_cached;
  
  // Create a mapping from cached nodes to new nodes
  std::unordered_map<HNode*, HNode*> cached_to_new;
  cached_to_new[H_cached] = H_new;
  
  // Collect all new nodes to push in prioritized order
  std::vector<HNode*> nodes_to_push;
  
  // Recursively merge all descendants of the cached node
  // Only follow true successors (parent-child relationships), not cross-edges from rewrite
  std::queue<HNode*> to_process;
  for (auto neighbor : H_cached->neighbors) {
    if (neighbor->parent == H_cached) {
      to_process.push(neighbor);
    }
  }
  
  while (!to_process.empty()) {
    auto H_descendant = to_process.front();
    to_process.pop();
    if (visited_cached.count(H_descendant)) continue;
    visited_cached.insert(H_descendant);
    
    // Mark this configuration as merged to avoid re-merge elsewhere
    merged_configs.insert(H_descendant->Q);

    // Check if this descendant is already in the new search tree
    auto iter = EXPLORED.find(H_descendant->Q);
    if (iter == EXPLORED.end()) {
      // Find the correct parent in the new tree
      HNode *parent_in_new = nullptr;
      if (H_descendant->parent != nullptr) {
        auto parent_iter = cached_to_new.find(H_descendant->parent);
        if (parent_iter != cached_to_new.end()) {
          parent_in_new = parent_iter->second;
        }
      }
      
      // EXPLICITLY calculate depth, g, and f values for the new node
      int new_depth = (parent_in_new != nullptr) ? parent_in_new->depth + 1 : 0;
      int new_h = get_h_val(H_descendant->Q);
      /*if (parent_in_new != nullptr) {
        std::cout<<"new_depth: "<<new_depth<< " "<<H_descendant->Q<<" "<<parent_in_new->Q<<std::endl;
      } else {
        std::cout<<"new_depth: "<<new_depth<< " "<<H_descendant->Q<<" [null-parent]"<<std::endl;
      }*/
      // Create new node first
      auto H_new_desc = new HNode(H_descendant->Q, D, ins, parent_in_new, 0, new_h);
      
      // EXPLICITLY calculate and set g_values vector with transition rules
      std::vector<int> new_g_values(ins->N);
      if (parent_in_new != nullptr) {
        // Inherit parent's g_values
        for (size_t i = 0; i < ins->N; ++i) {
          new_g_values[i] = parent_in_new->g_values[i];
        }
        // Apply transitions relative to parent_in_new
        for (size_t i = 0; i < ins->N; ++i) {
          const bool parent_at_goal = (parent_in_new->Q[i] == ins->goals[i]);
          const bool child_at_goal = (H_descendant->Q[i] == ins->goals[i]);
          if ((child_at_goal && !parent_at_goal) || (!child_at_goal && parent_at_goal)) {
            new_g_values[i] = new_depth;
          }
        }
      } else {
        // Root node - all agents start with cost 0
        for (size_t i = 0; i < ins->N; ++i) {
          new_g_values[i] = 0;
        }
      }
      
      // Calculate total g-value from g_values
      int new_g = 0;
      for (size_t i = 0; i < ins->N; ++i) {
        new_g += new_g_values[i];
      }
      int new_f = new_g + new_h;
      //std::cout<<"H_new_desc->Q: "<<H_new_desc->Q<<" "<<new_depth<<" "<<new_g<<" "<<new_h<<" "<<new_f<<std::endl;
      // EXPLICITLY override all calculated values
      H_new_desc->depth = new_depth;
      H_new_desc->g_values = new_g_values;
      H_new_desc->g = new_g;
      H_new_desc->f = new_f;

      // Link to parent in new tree
      H_new_desc->parent = parent_in_new;
      
      EXPLORED[H_new_desc->Q] = H_new_desc;
      GC_HNodes.push_back(H_new_desc);
      cached_to_new[H_descendant] = H_new_desc;
      ++stats_merged;
      
      // Collect node for prioritized pushing
      nodes_to_push.push_back(H_new_desc);
      
      // Enqueue only true successors (parent-child relationships) to process
      for (auto neighbor : H_descendant->neighbors) {
        if (neighbor->parent == H_descendant) {
          to_process.push(neighbor);
        }
      }
      
      solver_info(4, "merged cached descendant with recalculated values");
    }
  }
  
  // Push all collected nodes in prioritized order
  if (!nodes_to_push.empty()) {
    push_prioritized(nodes_to_push);
  }
}


bool LaCAM::set_new_config(HNode *H, LNode *L, Config &Q_to)
{
  for (uint d = 0; d < L->depth; ++d) Q_to[L->who[d]] = L->where[d];
  return pibt.set_new_config(H->Q, Q_to, H->order);
}

void LaCAM::rewrite(HNode *H_from, HNode *H_to)
{
  if (!ANYTIME) return;

  // update neighbors
  H_from->neighbors.insert(H_to);

  // Dijkstra
  std::queue<HNode *> Q({H_from});  // queue is sufficient
  while (!Q.empty()) {
    auto n_from = Q.front();
    Q.pop();
    for (auto n_to : n_from->neighbors) {
      auto g_val = get_g_val(n_from, n_to->Q);
      if (g_val < n_to->g) {
        if (n_to == H_goal) {
          solver_info(2, "cost update: g=", H_goal->g, " -> ", g_val,
                      ", depth=", H_goal->depth, " -> ", n_from->depth + 1);
        }
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
        n_to->depth = n_from->depth + 1;
        
        // Update g_values for the rewritten node
        for (size_t i = 0; i < ins->N; ++i) {
          n_to->g_values[i] = n_from->g_values[i];
          
          // if agent is now at goal and wasn't before, set its cost to current depth
          if (n_to->Q[i] == ins->goals[i] && n_from->Q[i] != ins->goals[i]) {
            n_to->g_values[i] = n_to->depth;
          }
          // if agent leaves goal, update its cost to current depth
          else if (n_to->Q[i] != ins->goals[i] && n_from->Q[i] == ins->goals[i]) {
            n_to->g_values[i] = n_to->depth;
          }
        }
        
        Q.push(n_to);
        // If goal improved, refresh cached solution suffix
        if (n_to == H_goal && best_solution_version < current_run_version) {
          // Only refresh cached suffix if it belongs to a previous run (never during current run)
          cache_solution_suffix_from_goal(H_goal);
        }
        if (H_goal != nullptr && n_to->f < H_goal->f) {
          OPEN.push_front(n_to);
          solver_info(4, "reinsert: g=", n_to->g, " < ", H_goal->g);
        }
      }
    }
  }
}

int LaCAM::get_g_val(HNode *H_parent, const Config &Q_to)
{
  // Calculate g-value as sum of individual agent costs
  auto g_val = 0;
  for (size_t i = 0; i < ins->N; ++i) {
    // inherit parent's g_value for this agent
    auto agent_g = H_parent->g_values[i];
    
    // if agent is now at goal and wasn't before, set its cost to current depth
    if (Q_to[i] == ins->goals[i] && H_parent->Q[i] != ins->goals[i]) {
      agent_g = H_parent->depth + 1;
    }
    // if agent leaves goal, update its cost to current depth
    else if (Q_to[i] != ins->goals[i] && H_parent->Q[i] == ins->goals[i]) {
      agent_g = H_parent->depth + 1;
    }
    
    g_val += agent_g;
  }
  return g_val;
}

int LaCAM::get_h_val(const Config &Q)
{
  auto c = 0;
  for (size_t i = 0; i < ins->N; ++i) c += D->get(i, Q[i]);
  return c;
}

int LaCAM::get_edge_cost(const Config &Q1, const Config &Q2)
{
  auto cost = 0;
  for (size_t i = 0; i < ins->N; ++i) {
    if (Q1[i] != ins->goals[i] || Q2[i] != ins->goals[i]) {
      cost += 1;
    }
  }
  return cost;
}
