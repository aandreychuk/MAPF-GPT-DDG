// cppimport
#include "observation_generator.h"

void ObservationGenerator::compute_cost2go_for_goal(const std::pair<int, int> &goal)
{
    // Check if we already have cost2go for this goal
    if (cost2go_cache.find(goal) != cost2go_cache.end())
        return;

    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::queue<std::pair<int, int>> fringe;
    fringe.push(goal);
    std::vector<std::vector<uint16_t>> cost_matrix(grid.size(), std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max()));
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
    
    // Convert distance matrix to directional flags
    std::vector<std::vector<int8_t>> direction_matrix(grid.size(), std::vector<int8_t>(grid[0].size(), 0));
    for (int i = 0; i < grid.size(); i++)
    {
        for (int j = 0; j < grid[0].size(); j++)
        {
            if (grid[i][j] != 0 || cost_matrix[i][j] == std::numeric_limits<uint16_t>::max())
            {
                direction_matrix[i][j] = 0; // Obstacle or unreachable
            }
            else
            {
                direction_matrix[i][j] = compute_direction_flags(cost_matrix, i, j) + 1; //TODO: remove redundant +1
            }
        }
    }
    
    // Store the computed directional matrix in cache
    cost2go_cache[goal] = direction_matrix;
}

int8_t ObservationGenerator::compute_direction_flags(const std::vector<std::vector<uint16_t>> &distances, int i, int j)
{
    // Moves: up, down, left, right (same order as in next_action)
    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    uint16_t current_dist = distances[i][j];
    int8_t flags = 0;
    
    if (current_dist == std::numeric_limits<uint16_t>::max())
        return 0; // Unreachable
    
    for (int move_idx = 0; move_idx < 4; move_idx++)
    {
        int ni = i + moves[move_idx].first;
        int nj = j + moves[move_idx].second;
        
        // Check bounds
        if (ni >= 0 && ni < grid.size() && nj >= 0 && nj < grid[0].size())
        {
            // Check if this neighbor has a lower cost (leads towards goal)
            if (grid[ni][nj] == 0 && distances[ni][nj] < current_dist)
            {
                flags |= (1 << move_idx); // Set bit for this direction
            }
        }
    }
    
    return flags + 1;
}

void ObservationGenerator::generate_cost2go_obs(int agent_idx, std::vector<std::vector<int>> &buffer)
{
    // UNIFIED OBSERVATION ENCODING SCHEME:
    // 0:     Obstacle/out of bounds
    // 1-16:  Direction to goal (no agent present)
    // 17-96: Agent present with combined action info
    const auto &agent = agents[agent_idx];
    const auto &goal = agent.goal;
    const auto &pos = agent.pos;
    
    // Fill the observation buffer with unified encoding
    for (int i = 0; i <= obs_radius * 2; i++)
    {
        for (int j = 0; j <= obs_radius * 2; j++)
        {
            int grid_i = pos.first - obs_radius + i;
            int grid_j = pos.second - obs_radius + j;
            
            // Check bounds
            if (grid_i < 0 || grid_i >= grid.size() || grid_j < 0 || grid_j >= grid[0].size() || grid[grid_i][grid_j] != 0)
            {
                buffer[i][j] = 0;
            }
            else
            {
                // Check if there's an agent at this position
                int agent_at_pos = agents_locations[grid_i][grid_j];
                if (agent_at_pos >= 0)
                {
                    // Agent present - encode agent information
                    buffer[i][j] = 16 + agents[agent_at_pos].last_action * 16 + agents[agent_at_pos].next_action;
                }
                else
                {
                    // No agent - use directional value
                    buffer[i][j] = cost2go_cache[goal][grid_i][grid_j];
                }
            }
        }
    }
}

void ObservationGenerator::create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals)
{
    agents.clear();
    int total_agents = positions.size();
    agents.resize(total_agents);
    obs_buffer.resize(total_agents, std::vector<std::vector<int>>(2 * obs_radius + 1, std::vector<int>(2 * obs_radius + 1)));
    
    for (int i = 0; i < total_agents; i++)
    {
        agents[i].pos = positions[i];
        agents[i].goal = goals[i];
        agents[i].last_action = 0; // Initialize with "no action" (n/w)
        compute_cost2go_for_goal(goals[i]);
        agents[i].next_action = cost2go_cache[goals[i]][positions[i].first][positions[i].second];
    }
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions)
{
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1; // first clear old locations for ALL agents
    
    for (size_t i = 0; i < agents.size(); i++)
    {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = i;
        agent.pos = positions[i];
        
        // Convert action index to action ID
        switch (actions[i])
        {
        case 0:
            agent.last_action = 0; // wait (w/n)
            break;
        case 1:
            agent.last_action = 1; // up (u)
            break;
        case 2:
            agent.last_action = 2; // down (d)
            break;
        case 3:
            agent.last_action = 3; // left (l)
            break;
        case 4:
            agent.last_action = 4; // right (r)
            break;
        default:
            agent.last_action = 0; // no action (n)
            break;
        }
        
        // If goal changed, compute cost2go for new goal
        if (agent.goal != goals[i])
        {
            agent.goal = goals[i];
            compute_cost2go_for_goal(goals[i]);
        }
    }
    
    // Update next actions for all agents
    for (size_t i = 0; i < agents.size(); i++)
        agents[i].next_action = cost2go_cache[agents[i].goal][agents[i].pos.first][agents[i].pos.second];
}

std::vector<std::vector<int>> ObservationGenerator::generate_observations()
{
    std::vector<std::vector<int>> observations(agents.size(), std::vector<int>(context_size, 0));
    for (size_t i = 0; i < agents.size(); i++)
    {
        generate_cost2go_obs(i, obs_buffer[i]);
        // Flatten the 2D observation buffer to 1D for this agent
        observations[i].clear();
        for (const auto &row : obs_buffer[i])
            for (int value : row)
                observations[i].push_back(value);
    }
    return observations;
}

pybind11::array_t<int> ObservationGenerator::generate_observations_numpy()
{
    int num_agents = agents.size();
    
    // Create numpy array directly
    auto result = pybind11::array_t<int>({num_agents, context_size});
    auto buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    
    // Fill the array directly without intermediate std::vector
    for (int agent_idx = 0; agent_idx < num_agents; agent_idx++)
    {
        generate_cost2go_obs(agent_idx, obs_buffer[agent_idx]);
        
        // Flatten 2D buffer directly into numpy array
        int idx = 0;
        for (const auto &row : obs_buffer[agent_idx])
            for (int value : row)
                ptr[agent_idx * context_size + idx++] = value;
    }
    
    return result;
}

int main()
{
    std::vector<std::vector<int>> grid = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    ObservationGenerator obs_gen(grid, 5, 128);
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
    py::class_<ObservationGenerator>(m, "ObservationGenerator")
        .def(py::init<const std::vector<std::vector<int>> &, int, int>())
        .def("create_agents", &ObservationGenerator::create_agents)
        .def("update_agents", &ObservationGenerator::update_agents)
        .def("generate_observations", &ObservationGenerator::generate_observations)
        .def("generate_observations_numpy", &ObservationGenerator::generate_observations_numpy);
}
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-m64']
cfg['extra_link_args'] = ['-m64']
setup_pybind11(cfg)
%>
*/
#endif
