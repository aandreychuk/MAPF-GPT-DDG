// cppimport
#include "observation_generator.h"

std::vector<std::vector<uint16_t>> ObservationGenerator::compute_full_cost2go(const std::pair<int, int> &goal)
{
    // Check if already computed
    if (full_cost2go_cache.find(goal) != full_cost2go_cache.end())
    {
        return full_cost2go_cache[goal];
    }

    // Compute full cost2go matrix from goal to all cells using BFS
    std::vector<std::vector<uint16_t>> cost_matrix(grid.size(), std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max()));
    
    if (grid[goal.first][goal.second] != 0)
    {
        // Goal is on obstacle, return empty matrix
        full_cost2go_cache[goal] = cost_matrix;
        return cost_matrix;
    }

    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::queue<std::pair<int, int>> fringe;
    fringe.push(goal);
    cost_matrix[goal.first][goal.second] = 0;

    while (!fringe.empty())
    {
        auto pos = fringe.front();
        fringe.pop();
        for (const auto &move : moves)
        {
            int new_i = pos.first + move.first;
            int new_j = pos.second + move.second;
            if (new_i >= 0 && new_j >= 0 && new_i < grid.size() && new_j < grid[0].size())
            {
                if (grid[new_i][new_j] == 0 && cost_matrix[new_i][new_j] == std::numeric_limits<uint16_t>::max())
                {
                    cost_matrix[new_i][new_j] = cost_matrix[pos.first][pos.second] + 1;
                    fringe.push(std::make_pair(new_i, new_j));
                }
            }
        }
    }

    // Cache and return
    full_cost2go_cache[goal] = cost_matrix;
    return cost_matrix;
}


void ObservationGenerator::generate_cost2go_obs(int agent_idx, bool only_obstacles, std::vector<std::vector<int>> &buffer)
{
    const auto &agent = agents[agent_idx];
    const auto &goal = agent.goal;
    const auto &pos = agent.pos;
    
    // Get or compute full cost2go matrix for this goal
    const auto &cost_matrix = compute_full_cost2go(goal);
    
    int middle_value = cost_matrix[pos.first][pos.second];
    
    if (only_obstacles)
    {
        for (int i = 0; i <= cfg.obs_radius * 2; i++)
            for (int j = 0; j <= cfg.obs_radius * 2; j++)
            {
                int grid_i = pos.first - cfg.obs_radius + i;
                int grid_j = pos.second - cfg.obs_radius + j;
                if (grid_i < 0 || grid_i >= grid.size() || grid_j < 0 || grid_j >= grid[0].size() || 
                    grid[grid_i][grid_j] != 0 || cost_matrix[grid_i][grid_j] == std::numeric_limits<uint16_t>::max())
                    buffer[i][j] = 1;
                else
                    buffer[i][j] = 0;
            }
        return;
    }
    
    for (int i = 0; i <= cfg.obs_radius * 2; i++)
        for (int j = 0; j <= cfg.obs_radius * 2; j++)
        {
            int grid_i = pos.first - cfg.obs_radius + i;
            int grid_j = pos.second - cfg.obs_radius + j;
            
            if (grid_i < 0 || grid_i >= grid.size() || grid_j < 0 || grid_j >= grid[0].size())
            {
                buffer[i][j] = -cfg.cost2go_value_limit * 4;
                continue;
            }
            
            int value = cost_matrix[grid_i][grid_j];
            if (value != std::numeric_limits<uint16_t>::max())
            {
                value -= middle_value;
                buffer[i][j] = value > cfg.cost2go_value_limit ? cfg.cost2go_value_limit * 2 : 
                               value < -cfg.cost2go_value_limit ? -cfg.cost2go_value_limit * 2 : value;
            }
            else
                buffer[i][j] = -cfg.cost2go_value_limit * 4;
        }
}

int ObservationGenerator::get_distance(int agent_idx, const std::pair<int, int> &pos)
{
    const auto &agent = agents[agent_idx];
    const auto &goal = agent.goal;
    
    if (pos.first < 0 || pos.first >= grid.size() || pos.second < 0 || pos.second >= grid[0].size())
        return -1;
    
    // Get or compute full cost2go matrix for this goal
    const auto &cost_matrix = compute_full_cost2go(goal);
    
    int distance = cost_matrix[pos.first][pos.second];
    if (distance == std::numeric_limits<uint16_t>::max())
        return -1;
    return distance;
}

Encoder::Encoder(const InputParameters &cfg) : cfg(cfg)
{
    for (int i = -cfg.cost2go_value_limit; i <= cfg.cost2go_value_limit; ++i)
        coord_range.push_back(i);
    coord_range.push_back(-cfg.cost2go_value_limit * 4);
    coord_range.push_back(-cfg.cost2go_value_limit * 2);
    coord_range.push_back(cfg.cost2go_value_limit * 2);

    actions_range = {'n', 'w', 'u', 'd', 'l', 'r'};
    for (int i = 0; i < 16; ++i)
    {
        std::stringstream ss;
        ss << std::bitset<4>(i);
        next_action_range.push_back(ss.str());
    }

    int idx = 0;
    for (auto &token : coord_range)
        int_vocab[token] = idx++;
    for (auto &token : actions_range)
        str_vocab[std::string(1, token)] = idx++;
    for (auto &token : next_action_range)
        str_vocab[token] = idx++;
    str_vocab["!"] = idx;

    for (auto &[token, idx] : int_vocab)
        inverse_int_vocab[idx] = token;
    for (auto &[token, idx] : str_vocab)
        inverse_str_vocab[idx] = token;
}

Decoder::Decoder(const InputParameters &cfg) : cfg(cfg)
{
    // Mirror Encoder vocabulary construction so that decoding is consistent
    for (int i = -cfg.cost2go_value_limit; i <= cfg.cost2go_value_limit; ++i)
        coord_range.push_back(i);
    coord_range.push_back(-cfg.cost2go_value_limit * 4);
    coord_range.push_back(-cfg.cost2go_value_limit * 2);
    coord_range.push_back(cfg.cost2go_value_limit * 2);

    actions_range = {'n', 'w', 'u', 'd', 'l', 'r'};
    for (int i = 0; i < 16; ++i)
    {
        std::stringstream ss;
        ss << std::bitset<4>(i);
        next_action_range.push_back(ss.str());
    }

    int idx = 0;
    for (auto &token : coord_range)
        int_vocab[token] = idx++;
    for (auto &token : actions_range)
        str_vocab[std::string(1, token)] = idx++;
    for (auto &token : next_action_range)
        str_vocab[token] = idx++;
    str_vocab["!"] = idx;

    for (auto &[token, idx_val] : int_vocab)
        inverse_int_vocab[idx_val] = token;
    for (auto &[token, idx_val] : str_vocab)
        inverse_str_vocab[idx_val] = token;
}

std::pair<std::vector<std::vector<int>>, std::vector<AgentsInfo>>
Decoder::decode(const std::vector<int> &encoded) const
{
    const int side = 2 * cfg.obs_radius + 1;
    const int num_cells = side * side;

    if (static_cast<int>(encoded.size()) < num_cells)
    {
        throw std::runtime_error("Encoded observation too short to contain cost2go matrix");
    }

    std::vector<std::vector<int>> cost2go(side, std::vector<int>(side, 0));
    for (int i = 0; i < side; ++i)
    {
        for (int j = 0; j < side; ++j)
        {
            int flat_idx = i * side + j;
            int token_idx = encoded[flat_idx];
            auto it = inverse_int_vocab.find(token_idx);
            if (it == inverse_int_vocab.end())
            {
                throw std::runtime_error("Unknown index in cost2go portion of encoded observation");
            }
            cost2go[i][j] = it->second;
        }
    }

    const int block_len = 5 + cfg.num_previous_actions;
    const int total_agents_region = cfg.num_agents * block_len;
    const int pad_idx = str_vocab.at("!");

    std::vector<AgentsInfo> agents;
    int offset = num_cells;

    // Don't read past available data, even if padded context_size was truncated
    int max_available = static_cast<int>(encoded.size()) - offset;
    int readable_agents_region = std::min(max_available, total_agents_region);

    for (int a = 0; a < cfg.num_agents; ++a)
    {
        int base = offset + a * block_len;
        if (base + block_len > offset + readable_agents_region)
            break;

        bool all_pad = true;
        for (int t = 0; t < block_len; ++t)
        {
            if (encoded[base + t] != pad_idx)
            {
                all_pad = false;
                break;
            }
        }
        if (all_pad)
        {
            // Remaining agent slots are padding
            break;
        }

        // Decode coordinates (relative position and relative goal)
        auto decode_coord = [this](int idx_val) -> int {
            auto it = inverse_int_vocab.find(idx_val);
            if (it == inverse_int_vocab.end())
                throw std::runtime_error("Unknown index in coord portion of encoded observation");
            return it->second;
        };

        int rel_pos_x = decode_coord(encoded[base + 0]);
        int rel_pos_y = decode_coord(encoded[base + 1]);
        int rel_goal_x = decode_coord(encoded[base + 2]);
        int rel_goal_y = decode_coord(encoded[base + 3]);

        // Decode previous actions
        std::deque<std::string> prev_actions;
        for (int t = 0; t < cfg.num_previous_actions; ++t)
        {
            int token_idx = encoded[base + 4 + t];
            auto it = inverse_str_vocab.find(token_idx);
            if (it == inverse_str_vocab.end())
                throw std::runtime_error("Unknown index in previous actions portion of encoded observation");
            const std::string &act = it->second;
            // Skip padding markers just in case; encoder never writes them here for real agents
            if (act != "!")
                prev_actions.push_back(act);
        }

        // Decode next action
        int next_action_idx = encoded[base + 4 + cfg.num_previous_actions];
        auto it_next = inverse_str_vocab.find(next_action_idx);
        if (it_next == inverse_str_vocab.end())
            throw std::runtime_error("Unknown index in next action portion of encoded observation");
        std::string next_action = it_next->second;

        agents.emplace_back(
            std::make_pair(rel_pos_x, rel_pos_y),
            std::make_pair(rel_goal_x, rel_goal_y),
            prev_actions,
            next_action);
    }

    return {cost2go, agents};
}

std::vector<int> Encoder::encode(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go)
{
    std::vector<int> agents_indices;
    for (const auto &agent : agents)
    {
        int goal_x = std::max(-cfg.cost2go_value_limit, std::min(cfg.cost2go_value_limit, agent.relative_goal.first));
        int goal_y = std::max(-cfg.cost2go_value_limit, std::min(cfg.cost2go_value_limit, agent.relative_goal.second));
        std::vector<int> coord_indices = {
            int_vocab.at(agent.relative_pos.first),
            int_vocab.at(agent.relative_pos.second),
            int_vocab.at(goal_x),
            int_vocab.at(goal_y)};

        std::vector<int> actions_indices;
        for (const auto &action : agent.previous_actions)
        {
            actions_indices.push_back(str_vocab.at(action));
        }
        std::vector<int> next_action_indices = {str_vocab.at(agent.next_action)};

        agents_indices.insert(agents_indices.end(), coord_indices.begin(), coord_indices.end());
        agents_indices.insert(agents_indices.end(), actions_indices.begin(), actions_indices.end());
        agents_indices.insert(agents_indices.end(), next_action_indices.begin(), next_action_indices.end());
    }

    if (agents.size() < cfg.num_agents)
        agents_indices.insert(agents_indices.end(), (cfg.num_agents - agents.size()) * (5 + cfg.num_previous_actions), str_vocab["!"]);

    std::vector<int> cost2go_indices;
    for (const auto &row : cost2go)
        for (int value : row)
            cost2go_indices.push_back(int_vocab.at(value));

    std::vector<int> result;
    result.insert(result.end(), cost2go_indices.begin(), cost2go_indices.end());
    result.insert(result.end(), agents_indices.begin(), agents_indices.end());
    while (result.size() < 256)
        result.push_back(str_vocab["!"]);
    return result;
}

void ObservationGenerator::create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals)
{
    agents.clear();
    int total_agents = positions.size();
    agents.resize(total_agents);
    agents_in_observation.resize(total_agents);
    cost2go_obs_buffer.resize(total_agents, std::vector<std::vector<int>>(2 * cfg.obs_radius + 1, std::vector<int>(2 * cfg.obs_radius + 1)));
    
    // Precompute cost2go for all unique goals
    std::set<std::pair<int, int>> unique_goals(goals.begin(), goals.end());
    for (const auto &goal : unique_goals)
    {
        compute_full_cost2go(goal);
    }
    
    for (int i = 0; i < total_agents; i++)
    {
        agents[i].pos = positions[i];
        agents[i].goal = goals[i];
        for (int j = 0; j < cfg.num_previous_actions; ++j)
        {
            agents[i].action_history.push_back("n");
        }
        update_next_action(i);
    }
}

void ObservationGenerator::update_next_action(int agent_idx)
{
    std::string next_action;
    auto &agent = agents[agent_idx];
    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int current_cost = get_distance(agent_idx, agent.pos);

    for (const auto &move : moves)
    {
        std::pair<int, int> new_pos = {agent.pos.first + move.first, agent.pos.second + move.second};
        int neighbor_cost = get_distance(agent_idx, new_pos);

        if (neighbor_cost >= 0 && current_cost > neighbor_cost)
            next_action += "1";
        else
            next_action += "0";
    }
    agent.next_action = next_action;
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions)
{
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1; // first clear old locations for ALL agents
    
    // Collect unique goals that need cost2go computation
    std::set<std::pair<int, int>> goals_to_compute;
    for (size_t i = 0; i < agents.size(); i++)
    {
        if (agents[i].goal != goals[i])
        {
            goals_to_compute.insert(goals[i]);
        }
    }
    
    // Precompute cost2go for new goals
    for (const auto &goal : goals_to_compute)
    {
        compute_full_cost2go(goal);
    }
    
    for (size_t i = 0; i < agents.size(); i++)
    {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = i;
        agent.pos = positions[i];
        switch (actions[i])
        {
        case 0:
            agent.action_history.push_back("w");
            break;
        case 1:
            agent.action_history.push_back("u");
            break;
        case 2:
            agent.action_history.push_back("d");
            break;
        case 3:
            agent.action_history.push_back("l");
            break;
        case 4:
            agent.action_history.push_back("r");
            break;
        default:
            agent.action_history.push_back("n");
            break;
        }
        agent.action_history.pop_front();
        if (agent.goal != goals[i])
        {
            agent.goal = goals[i];
        }
    }

    for (size_t i = 0; i < agents.size(); i++)
        update_next_action(i);
}

std::vector<AgentsInfo> ObservationGenerator::get_agents_info(int agent_idx)
{
    std::vector<AgentsInfo> agents_info;
    std::vector<int> considered_agents;
    const auto &cur_agent = agents[agent_idx];
    for (int i = -cfg.agents_radius; i <= cfg.agents_radius; i++)
        for (int j = -cfg.agents_radius; j <= cfg.agents_radius; j++)
            if (agents_locations[cur_agent.pos.first + i][cur_agent.pos.second + j] >= 0)
                considered_agents.push_back(agents_locations[cur_agent.pos.first + i][cur_agent.pos.second + j]);
    std::vector<int> distances(considered_agents.size(), -1);
    for (size_t i = 0; i < considered_agents.size(); i++)
        distances[i] = std::abs(agents[considered_agents[i]].pos.first - cur_agent.pos.first) +
                       std::abs(agents[considered_agents[i]].pos.second - cur_agent.pos.second);
    std::vector<std::pair<int, int>> distance_agent_pairs;
    for (size_t i = 0; i < considered_agents.size(); i++)
    {
        distance_agent_pairs.push_back({distances[i], considered_agents[i]});
    }
    std::sort(distance_agent_pairs.begin(), distance_agent_pairs.end());
    agents_in_observation[agent_idx] = std::vector<int>(13,-1);
    for (int i = 0; i < std::min(int(distance_agent_pairs.size()), cfg.num_agents); i++)
    {
        const auto &agent = agents[distance_agent_pairs[i].second];
        agents_in_observation[agent_idx][i] = distance_agent_pairs[i].second;
        agents_info.push_back(AgentsInfo(std::make_pair(agent.pos.first - cur_agent.pos.first, agent.pos.second - cur_agent.pos.second),
                                         std::make_pair(agent.goal.first - cur_agent.pos.first, agent.goal.second - cur_agent.pos.second),
                                         agent.action_history, agent.next_action));
    }
    return agents_info;
}

std::vector<std::vector<int>> ObservationGenerator::generate_observations()
{

    std::vector<std::vector<int>> observations(agents.size());
    for (size_t i = 0; i < agents.size(); i++)
    {
        generate_cost2go_obs(i, false, cost2go_obs_buffer[i]);
        std::vector<AgentsInfo> agents_info = get_agents_info(i);
        observations[i] = encoder.encode(agents_info, cost2go_obs_buffer[i]);
    }
    return observations;
}

std::vector<std::vector<int>> ObservationGenerator::get_agents_ids_in_observations()
{
    return agents_in_observation;
}

int main()
{
    std::vector<std::vector<int>> grid = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    ObservationGenerator obs_gen(grid, InputParameters());
    obs_gen.create_agents(std::vector<std::pair<int, int>>{{120, 120}}, std::vector<std::pair<int, int>>{{20, 200}});
    obs_gen.update_agents(std::vector<std::pair<int, int>>{{120, 120}}, std::vector<std::pair<int, int>>{{20, 200}}, std::vector<int>{0});
    auto obs = obs_gen.generate_observations();
    for (const auto &obs_row : obs)
    {
        for (const auto &cell : obs_row)
            std::cout << cell << " ";
        std::cout << std::endl;
    }
    return 0;
}

#ifdef PYBIND11_MODULE
namespace py = pybind11;
PYBIND11_MODULE(observation_generator, m)
{
    py::class_<InputParameters>(m, "InputParameters")
        .def(py::init<int, int, int, int, int, int>())
        .def_readwrite("cost2go_value_limit", &InputParameters::cost2go_value_limit)
        .def_readwrite("num_agents", &InputParameters::num_agents)
        .def_readwrite("num_previous_actions", &InputParameters::num_previous_actions)
        .def_readwrite("agents_radius", &InputParameters::agents_radius)
        .def_readwrite("context_size", &InputParameters::context_size)
        .def_readwrite("obs_radius", &InputParameters::obs_radius);

    py::class_<AgentsInfo>(m, "AgentsInfo")
        .def(py::init<>())
        .def(py::init<std::pair<int, int>, std::pair<int, int>, std::deque<std::string>, std::string>())
        .def_readwrite("relative_pos", &AgentsInfo::relative_pos)
        .def_readwrite("relative_goal", &AgentsInfo::relative_goal)
        .def_readwrite("previous_actions", &AgentsInfo::previous_actions)
        .def_readwrite("next_action", &AgentsInfo::next_action);

    py::class_<Decoder>(m, "Decoder")
        .def(py::init<const InputParameters &>())
        .def("decode", &Decoder::decode);

    py::class_<ObservationGenerator>(m, "ObservationGenerator", py::module_local())
        .def(py::init<const std::vector<std::vector<int>> &, const InputParameters &>())
        .def("create_agents", &ObservationGenerator::create_agents)
        .def("update_agents", &ObservationGenerator::update_agents)
        .def("generate_observations", &ObservationGenerator::generate_observations)
        .def("get_agents_ids_in_observations", &ObservationGenerator::get_agents_ids_in_observations);
}
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-fopenmp', '-m64']
cfg['extra_link_args'] = ['-m64']
setup_pybind11(cfg)
%>
*/
#endif
