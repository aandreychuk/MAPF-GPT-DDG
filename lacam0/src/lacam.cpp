#include "../include/lacam.hpp"
#include <queue>
#include <unordered_set>

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

Solution LaCAM::solve()
{
  solver_info(1, "LaCAM begins");

  // setup search
  auto EXPLORED = std::unordered_map<Config, HNode *, ConfigHasher>();
  HNodes GC_HNodes;

  // seed from cache if enabled and compatible
  const bool can_reuse = reuse_cache && cache_ins == ins && cache_D == D;
  if (can_reuse) {
    EXPLORED = EXPLORED_CACHE;
    GC_HNodes = GC_HNodes_CACHE;
  } else {
    // reset cache context if reuse requested but incompatible
    if (reuse_cache) {
      EXPLORED_CACHE.clear();
      GC_HNodes_CACHE.clear();
    }
    cache_ins = ins;
    cache_D = D;
  }

  // insert initial node
  HNode *H_init = nullptr;
  auto it_cached = EXPLORED.find(ins->starts);
  if (it_cached != EXPLORED.end()) {
    H_init = it_cached->second;
  } else {
    H_init = new HNode(ins->starts, D, ins);
    EXPLORED[H_init->Q] = H_init;
    GC_HNodes.push_back(H_init);
  }
  OPEN.push_front(H_init);

  // search loop
  solver_info(2, "search iteration begins");
  while (!OPEN.empty() && !is_expired(deadline)) {
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
    std::cout << "H->Q: " << H->Q << std::endl;
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
      if (!ANYTIME) break;
      continue;
    }

    // extract constraints
    if (H->search_tree.empty()) {
      OPEN.pop_front();
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

    // check explored list
    auto iter = EXPLORED.find(Q_to);
    if (iter == EXPLORED.end()) {
      // new one -> insert
      auto H_new = new HNode(Q_to, D, ins, H, get_g_val(H, Q_to), get_h_val(Q_to));
      OPEN.push_front(H_new);
      EXPLORED[H_new->Q] = H_new;
      GC_HNodes.push_back(H_new);
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
    // persist cache for next runs
    EXPLORED_CACHE = EXPLORED;
    GC_HNodes_CACHE = GC_HNodes;
  } else {
    for (auto &&H : GC_HNodes) delete H;  // memory management
  }

  return solution;
}

Solution LaCAM::solve_from_config(const Config &start)
{
  solver_info(1, "LaCAM begins (from config)");

  // Always start fresh search tree for new start configuration
  auto EXPLORED = std::unordered_map<Config, HNode *, ConfigHasher>();
  HNodes GC_HNodes;
  
  // Track which cached configurations have been merged to avoid duplicate merging
  std::unordered_set<Config, ConfigHasher> merged_from_cache;

  // Check if we can reuse cache
  const bool can_reuse = reuse_cache && cache_ins == ins && cache_D == D;
  if (!can_reuse && reuse_cache) {
    EXPLORED_CACHE.clear();
    GC_HNodes_CACHE.clear();
    cache_ins = ins;
    cache_D = D;
  }

  // Always create fresh start node for new search tree
  auto H_init = new HNode(start, D, ins);
  EXPLORED[H_init->Q] = H_init;
  GC_HNodes.push_back(H_init);
  
  // Use priority queue for f-value based expansion
  OPEN_PQ = std::priority_queue<HNode*, std::vector<HNode*>, HNodeComparator>();
  OPEN_PQ.push(H_init);
  H_goal = nullptr;

  // search loop with f-value based expansion
  solver_info(2, "search iteration begins (f-value based)");
  while (!OPEN_PQ.empty() && !is_expired(deadline)) {
    ++loop_cnt;

    // Random insertions for diversification (less frequent with f-value ordering)
    if (H_goal != nullptr) {
      auto r = rrd(MT);
      if (r < RANDOM_INSERT_PROB2 / 4) {  // Reduced probability
        OPEN_PQ.push(H_init);
      } else if (r < RANDOM_INSERT_PROB2 / 2) {  // Reduced probability
        // For priority queue, we can't easily get random element, so skip this case
        // or implement a more complex random selection
      }
    }

    // Select node with lowest f-value (and lowest h-value for ties)
    auto H = OPEN_PQ.top();
    OPEN_PQ.pop();
    std::cout << "H->Q: " << H->Q << " f=" << H->f << " h=" << H->h << " g=" << H->g << std::endl;

    // Pruning based on f-values
    if (H_goal != nullptr && H->f >= H_goal->f) {
      solver_info(5, "prune, f=", H->f, " >= ", H_goal->f);
      continue;
    }

    if (H_goal == nullptr && is_same_config(H->Q, ins->goals)) {
      H_goal = H;
      solver_info(2, "found solution, g=", H->g, ", depth=", H->depth);
      if (!ANYTIME) break;
      continue;
    }

    if (H->search_tree.empty()) {
      OPEN.pop_front();
      continue;
    }
    auto L = H->search_tree.front();
    H->search_tree.pop();

    if (L->depth < H->Q.size()) {
      const auto i = H->order[L->depth];
      auto &&C = H->Q[i]->actions;
      std::shuffle(C.begin(), C.end(), MT);
      for (auto u : C) H->search_tree.push(new LNode(L, i, u));
    }

    auto Q_to = Config(ins->N, nullptr);
    auto res = set_new_config(H, L, Q_to);
    delete L;
    if (!res) continue;

    auto iter = EXPLORED.find(Q_to);
    if (iter == EXPLORED.end()) {
      // Check if this configuration exists in cache
      HNode *H_cached = nullptr;
      if (can_reuse) {
        auto cache_iter = EXPLORED_CACHE.find(Q_to);
        if (cache_iter != EXPLORED_CACHE.end()) {
          H_cached = cache_iter->second;
          std::cout << "H_cached->Q: " << H_cached->Q << " FOUND IN CACHE" << std::endl;
        }
      }
      
      if (H_cached != nullptr) {
        // Check if this cached configuration was already merged
        if (merged_from_cache.find(Q_to) == merged_from_cache.end()) {
          // Merge cached node into new search tree
          auto H_new = new HNode(Q_to, D, ins, H, get_g_val(H, Q_to), get_h_val(Q_to));
          OPEN_PQ.push(H_new);
          EXPLORED[H_new->Q] = H_new;
          GC_HNodes.push_back(H_new);
          
          // Mark as merged and merge entire cached subtree
          merged_from_cache.insert(Q_to);
          merge_cached_subtree(H_new, H_cached, EXPLORED, GC_HNodes);
          
          solver_info(3, "merged cached configuration and subtree");
        } else {
          // Configuration already merged, just create a regular new node
          auto H_new = new HNode(Q_to, D, ins, H, get_g_val(H, Q_to), get_h_val(Q_to));
          OPEN_PQ.push(H_new);
          EXPLORED[H_new->Q] = H_new;
          GC_HNodes.push_back(H_new);
          
          solver_info(3, "cached configuration already merged, created regular node");
        }
      } else {
        // Completely new configuration
        auto H_new = new HNode(Q_to, D, ins, H, get_g_val(H, Q_to), get_h_val(Q_to));
        OPEN_PQ.push(H_new);
        EXPLORED[H_new->Q] = H_new;
        GC_HNodes.push_back(H_new);
      }
    } else {
      auto H_known = iter->second;
      rewrite(H, H_known);

      if (rrd(MT) >= RANDOM_INSERT_PROB1) {
        OPEN_PQ.push(iter->second);
      } else {
        solver_info(3, "random restart");
        OPEN_PQ.push(H_init);
      }
    }
  }

  Solution solution;
  {
    auto H = H_goal;
    while (H != nullptr) {
      solution.push_back(H->Q);
      H = H->parent;
    }
    std::reverse(solution.begin(), solution.end());
  }

  if (reuse_cache) {
    // Merge new search tree into cache
    for (auto &pair : EXPLORED) {
      EXPLORED_CACHE[pair.first] = pair.second;
    }
    for (auto H : GC_HNodes) {
      GC_HNodes_CACHE.push_back(H);
    }
  } else {
    for (auto &&H : GC_HNodes) delete H;
  }

  return solution;
}

void LaCAM::clear_cache()
{
  for (auto &&H : GC_HNodes_CACHE) delete H;
  GC_HNodes_CACHE.clear();
  EXPLORED_CACHE.clear();
  cache_ins = nullptr;
  cache_D = nullptr;
}

void LaCAM::merge_cached_subtree(HNode *H_new, HNode *H_cached, 
                                std::unordered_map<Config, HNode *, ConfigHasher> &EXPLORED,
                                HNodes &GC_HNodes)
{
  // Copy neighbor information from cached node
  for (auto neighbor : H_cached->neighbors) {
    H_new->neighbors.insert(neighbor);
  }
  
  // Create a mapping from cached nodes to new nodes
  std::unordered_map<HNode*, HNode*> cached_to_new;
  cached_to_new[H_cached] = H_new;
  
  // Recursively merge all descendants of the cached node
  std::queue<HNode*> to_process;
  for (auto neighbor : H_cached->neighbors) {
    to_process.push(neighbor);
  }
  
  while (!to_process.empty()) {
    auto H_descendant = to_process.front();
    to_process.pop();
    
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
      std::cout<<"new_depth: "<<new_depth<< " "<<H_descendant->Q<<" "<<parent_in_new->Q<<std::endl;
      // Create new node first
      auto H_new_desc = new HNode(H_descendant->Q, D, ins, parent_in_new, 0, new_h);
      
      // EXPLICITLY calculate and set g_values vector
      std::vector<int> new_g_values(ins->N);
      if (parent_in_new != nullptr) {
        // Inherit parent's g_values
        for (size_t i = 0; i < ins->N; ++i) {
          new_g_values[i] = parent_in_new->g_values[i];
        }
        
        // Update g_values based on agent goal status
        for (size_t i = 0; i < ins->N; ++i) {
          // If agent is not at goal, g-value equals current depth
          if (H_descendant->Q[i] != ins->goals[i]) {
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
      std::cout<<"H_new_desc->Q: "<<H_new_desc->Q<<" "<<new_depth<<" "<<new_g<<" "<<new_h<<" "<<new_f<<std::endl;
      // EXPLICITLY override all calculated values
      H_new_desc->depth = new_depth;
      H_new_desc->g_values = new_g_values;
      H_new_desc->g = new_g;
      H_new_desc->f = new_f;
      
      EXPLORED[H_new_desc->Q] = H_new_desc;
      GC_HNodes.push_back(H_new_desc);
      cached_to_new[H_descendant] = H_new_desc;
      
      // Add to priority queue for expansion
      OPEN_PQ.push(H_new_desc);
      
      // Copy neighbor information
      for (auto neighbor : H_descendant->neighbors) {
        H_new_desc->neighbors.insert(neighbor);
        to_process.push(neighbor);
      }
      
      solver_info(4, "merged cached descendant with recalculated values");
    }
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
