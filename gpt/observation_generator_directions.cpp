// cppimport
#include "observation_generator_directions.h"

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
                direction_matrix[i][j] = compute_direction_flags(cost_matrix, i, j);// + 1; //TODO: remove redundant +1
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
    
    agents_in_observation[agent_idx].clear();
    std::vector<int> agents_in_observation_distances;
    
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
                    agents_in_observation[agent_idx].push_back(agent_at_pos);
                    agents_in_observation_distances.push_back(abs(i - obs_radius) + abs(j - obs_radius));
                }
                else
                {
                    // No agent - use directional value
                    buffer[i][j] = cost2go_cache[goal][grid_i][grid_j];
                }
            }
        }
    }
    // Create pairs of (agent_id, distance) and sort by distance
    std::vector<std::pair<int, int>> agent_distance_pairs;
    for (size_t idx = 0; idx < agents_in_observation[agent_idx].size(); idx++) {
        agent_distance_pairs.push_back({agents_in_observation[agent_idx][idx], agents_in_observation_distances[idx]});
    }
    std::sort(agent_distance_pairs.begin(), agent_distance_pairs.end(), 
        [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
            return a.second < b.second;
        });
    // Extract sorted agent IDs
    agents_in_observation[agent_idx].clear();
    for (const auto &pair : agent_distance_pairs) {
        agents_in_observation[agent_idx].push_back(pair.first);
    }
}

void ObservationGenerator::create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals)
{
    agents.clear();
    int total_agents = positions.size();
    agents.resize(total_agents);
    obs_buffer.resize(total_agents, std::vector<std::vector<int>>(2 * obs_radius + 1, std::vector<int>(2 * obs_radius + 1)));
    agents_in_observation.resize(total_agents);
    
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
        int idx = 0;
        for (const auto &row : obs_buffer[i])
            for (int value : row)
                observations[i][idx++] = value;
    }
    return observations;
}


void ObservationGenerator::display_cost2go_matrix(const std::pair<int, int> &goal, bool use_unicode)
{
    // Ensure cost2go is computed for this goal
    compute_cost2go_for_goal(goal);
    
    const auto& direction_matrix = cost2go_cache[goal];
    
    // Function to generate a 3x3 cell representation for given direction flags
    auto generate_3x3_cell = [](int8_t direction_flags, bool is_obstacle) -> std::vector<std::string> {
        if (is_obstacle) {
            return {"###", "###", "###"};
        }
        
        if (direction_flags == 0) {
            return {"...", "...", "..."};
        }
        
        // direction_flags is stored as (actual_flags + 1), so subtract 1
        int flags = (direction_flags - 1) & 0xF;
        
        // Extract individual direction bits based on compute_direction_flags order:
        // moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}} -> up, down, left, right
        bool up = flags & 1;    // bit 0: up (-1, 0)
        bool down = flags & 2;  // bit 1: down (1, 0) 
        bool left = flags & 4;  // bit 2: left (0, -1)
        bool right = flags & 8; // bit 3: right (0, 1)
        
        std::vector<std::string> cell(3, "   ");
        
        // Top row
        cell[0][1] = up ? '^' : ' ';
        
        // Middle row
        cell[1][0] = left ? '<' : ' ';
        cell[1][1] = ' '; // Center is always empty
        cell[1][2] = right ? '>' : ' ';
        
        // Bottom row
        cell[2][1] = down ? 'v' : ' ';
        
        return cell;
    };
    
    std::cout << "\nCost2Go Matrix with 3x3 Directional Cells for goal (" << goal.first << "," << goal.second << "):\n";
    std::cout << "Legend: Each cell shows optimal directions as:  ^  \n";
    std::cout << "                                               < > \n";
    std::cout << "                                                v  \n";
    std::cout << "        ### = obstacle, ... = unreachable, empty spaces = no movement in that direction\n\n";
    
    // Limit display size for readability
    int max_display_size = std::min(15, (int)grid.size());
    int max_display_cols = std::min(15, (int)grid[0].size());
    
    // Display column numbers header (each column takes 4 characters including separator)
    std::cout << "    ";
    for (int j = 0; j < max_display_cols; j++) {
        std::cout << std::setw(3) << (j % 10) << "│";
    }
    std::cout << "\n";
    
    // Display top border
    std::cout << "   ┌";
    for (int j = 0; j < max_display_cols; j++) {
        std::cout << "───" << (j == max_display_cols - 1 ? "┐" : "┬");
    }
    std::cout << "\n";
    
    // Display the matrix with 3x3 cells and grid lines
    for (int i = 0; i < max_display_size; i++) {
        // Each row is displayed as 3 lines (for the 3x3 cells)
        std::vector<std::vector<std::string>> row_cells;
        
        // Generate 3x3 representation for each cell in this row
        for (int j = 0; j < max_display_cols; j++) {
            bool is_obstacle = (grid[i][j] != 0);
            int8_t direction_flags = is_obstacle ? 0 : direction_matrix[i][j];
            row_cells.push_back(generate_3x3_cell(direction_flags, is_obstacle));
        }
        
        // Print the 3 lines of this row
        for (int line = 0; line < 3; line++) {
            if (line == 1) {
                // Middle line: show row number
                std::cout << std::setw(2) << (i % 100) << " │";
            } else {
                // Top and bottom lines: just spacing
                std::cout << "   │";
            }
            
            // Print this line for all cells in the row with vertical separators
            for (const auto& cell : row_cells) {
                std::cout << cell[line] << "│";
            }
            std::cout << "\n";
        }
        
        // Print horizontal separator after each row
        if (i == max_display_size - 1) {
            // Bottom border
            std::cout << "   └";
            for (int j = 0; j < max_display_cols; j++) {
                std::cout << "───" << (j == max_display_cols - 1 ? "┘" : "┴");
            }
        } else {
            // Middle separator
            std::cout << "   ├";
            for (int j = 0; j < max_display_cols; j++) {
                std::cout << "───" << (j == max_display_cols - 1 ? "┤" : "┼");
            }
        }
        std::cout << "\n";
    }
    
    // Display goal position
    std::cout << "\nGoal position: (" << goal.first << "," << goal.second << ")\n";
    if (max_display_size < grid.size() || max_display_cols < grid[0].size()) {
        std::cout << "Display limited to " << max_display_size << "x" << max_display_cols << " for readability.\n";
    }
}

void ObservationGenerator::display_observation(int agent_idx)
{
    if (agent_idx >= agents.size()) {
        std::cout << "Invalid agent index: " << agent_idx << std::endl;
        return;
    }
    
    // Generate observation for this agent
    generate_cost2go_obs(agent_idx, obs_buffer[agent_idx]);
    
    const auto& agent = agents[agent_idx];
    const auto& obs = obs_buffer[agent_idx];
    int obs_size = 2 * obs_radius + 1;
    
    // Function to decode observation value and generate 3x3 cell
    auto decode_and_generate_cell = [](int obs_value) -> std::vector<std::string> {
        if (obs_value == 0) {
            // Obstacle/out of bounds
            return {"###", "###", "###"};
        }
        
        bool has_agent = false;
        int direction_flags = 0;
        
        if (obs_value >= 17 && obs_value <= 96) {
            // Agent present: 16 + last_action * 16 + next_action
            has_agent = true;
            int agent_info = obs_value - 16;
            int next_action = agent_info % 16; // Extract next_action
            direction_flags = next_action; // next_action contains the direction flags (already +1)
        } else if (obs_value >= 1 && obs_value <= 16) {
            // Direction only (no agent)
            direction_flags = obs_value;
        }
        
        // Handle case where direction_flags is 0 (unreachable)
        if (direction_flags == 0) {
            return {"...", "...", "..."};
        }
        
        // direction_flags is stored as (actual_flags + 1), so subtract 1
        int flags = (direction_flags - 1) & 0xF;
        
        // Extract individual direction bits based on compute_direction_flags order:
        // moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}} -> up, down, left, right
        bool up = flags & 1;    // bit 0: up (-1, 0)
        bool down = flags & 2;  // bit 1: down (1, 0) 
        bool left = flags & 4;  // bit 2: left (0, -1)
        bool right = flags & 8; // bit 3: right (0, 1)
        
        std::vector<std::string> cell(3, "   ");
        
        // Top row
        cell[0][1] = up ? '^' : ' ';
        
        // Middle row
        cell[1][0] = left ? '<' : ' ';
        cell[1][1] = has_agent ? 'A' : ' '; // 'A' for agent, space otherwise
        cell[1][2] = right ? '>' : ' ';
        
        // Bottom row
        cell[2][1] = down ? 'v' : ' ';
        
        return cell;
    };
    
    std::cout << "\nObservation Matrix for Agent " << agent_idx << " at (" << agent.pos.first << "," << agent.pos.second << "):\n";
    std::cout << "Goal: (" << agent.goal.first << "," << agent.goal.second << "), Observation radius: " << obs_radius << "\n";
    std::cout << "Legend: Each cell shows greedy directions as:  ^  \n";
    std::cout << "                                              <A> (A = agent present)\n";
    std::cout << "                                               v  \n";
    std::cout << "        ### = obstacle, ... = unreachable\n\n";
    
    // Display column numbers header
    std::cout << "    ";
    for (int j = 0; j < obs_size; j++) {
        std::cout << std::setw(3) << j << "│";
    }
    std::cout << "\n";
    
    // Display top border
    std::cout << "   ┌";
    for (int j = 0; j < obs_size; j++) {
        std::cout << "───" << (j == obs_size - 1 ? "┐" : "┬");
    }
    std::cout << "\n";
    
    // Display the observation matrix with 3x3 cells
    for (int i = 0; i < obs_size; i++) {
        // Generate 3x3 representation for each cell in this row
        std::vector<std::vector<std::string>> row_cells;
        for (int j = 0; j < obs_size; j++) {
            row_cells.push_back(decode_and_generate_cell(obs[i][j]));
        }
        
        // Print the 3 lines of this row
        for (int line = 0; line < 3; line++) {
            if (line == 1) {
                // Middle line: show row number
                std::cout << std::setw(2) << i << " │";
            } else {
                // Top and bottom lines: just spacing
                std::cout << "   │";
            }
            
            // Print this line for all cells in the row with vertical separators
            for (const auto& cell : row_cells) {
                std::cout << cell[line] << "│";
            }
            std::cout << "\n";
        }
        
        // Print horizontal separator after each row
        if (i == obs_size - 1) {
            // Bottom border
            std::cout << "   └";
            for (int j = 0; j < obs_size; j++) {
                std::cout << "───" << (j == obs_size - 1 ? "┘" : "┴");
            }
        } else {
            // Middle separator
            std::cout << "   ├";
            for (int j = 0; j < obs_size; j++) {
                std::cout << "───" << (j == obs_size - 1 ? "┤" : "┼");
            }
        }
        std::cout << "\n";
    }
    
    std::cout << "\nAgent position in observation: (" << obs_radius << "," << obs_radius << ") [center]\n";
    
    // Debug decoding for agent position
    std::cout << "\nDEBUG - Agent decoding verification:\n";
    int agent_obs_val = obs[5][5]; // Agent at center
    std::cout << "Agent raw value: " << agent_obs_val << "\n";
    if (agent_obs_val >= 17) {
        int agent_info = agent_obs_val - 16;
        int next_action = agent_info % 16;
        int flags = (next_action - 1) & 0xF;
        std::cout << "next_action: " << next_action << ", flags: " << flags << " (binary: ";
        for (int bit = 3; bit >= 0; bit--) {
            std::cout << ((flags >> bit) & 1);
        }
        std::cout << ")\n";
        std::cout << "up=" << (flags & 1) << ", down=" << (flags & 2) << ", left=" << (flags & 4) << ", right=" << (flags & 8) << "\n";
    }
}

std::vector<std::vector<int>> ObservationGenerator::get_agents_ids_in_observations()
{
    return agents_in_observation;
}

int main()
{
    // Create a smaller grid for testing (20x20)
    std::vector<std::vector<int>> grid = std::vector<std::vector<int>>(20, std::vector<int>(20, 0));
    
    // Add some obstacles for testing
    grid[5][5] = 1;
    grid[5][6] = 1;
    grid[6][5] = 1;
    grid[10][10] = 1;
    grid[10][11] = 1;
    grid[11][10] = 1;
    grid[11][11] = 1;
    
    ObservationGenerator obs_gen(grid, 5, 128);
    
    // Position agent very close to border to see out-of-bounds areas
    // Agent at (3,3) with radius 5 will see out-of-bounds on top and left sides
    // Observation window will be from world coords (-2,-2) to (8,8)
    std::vector<std::pair<int, int>> agent_positions = {{3, 3}, {12, 8}, {6, 12}};
    std::vector<std::pair<int, int>> agent_goals = {{15, 15}, {3, 15}, {16, 4}};
    
    obs_gen.create_agents(agent_positions, agent_goals);
    obs_gen.update_agents(agent_positions, agent_goals, std::vector<int>{0, 0, 0});
    
    std::pair<int, int> goal = agent_goals[0]; // Use first agent's goal for cost2go display
    
    // Display the cost2go matrix with 3x3 directional cells
    obs_gen.display_cost2go_matrix(goal, false); // use_unicode parameter is now ignored
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "OBSERVATION MATRIX VISUALIZATION:\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Display the observation matrix for the agent
    obs_gen.display_observation(0);
    
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Raw observation values as 11x11 matrix:\n";
    auto obs = obs_gen.generate_observations();
    
    // Display as matrix matching the observation layout
    int obs_size = 2 * obs_gen.obs_radius + 1; // Should be 11
    std::cout << "     ";
    for (int j = 0; j < obs_size; j++) {
        std::cout << std::setw(3) << j;
    }
    std::cout << "\n";
    
    for (int i = 0; i < obs_size; i++) {
        std::cout << std::setw(2) << i << ": ";
        for (int j = 0; j < obs_size; j++) {
            int idx = i * obs_size + j;
            std::cout << std::setw(3) << obs[0][idx];
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "Agent IDs in observations:\n";
    for (int i = 0; i < 3; i++) {
        auto agent_ids = obs_gen.get_agents_ids_in_observations();
        std::cout << "Agent " << i << " sees agents: ";
        for (size_t j = 0; j < agent_ids.size(); j++) {
            std::cout << agent_ids[i][j] << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}

#ifdef PYBIND11_MODULE
namespace py = pybind11;
PYBIND11_MODULE(observation_generator_directions, m)
{
    py::class_<ObservationGenerator>(m, "ObservationGenerator")
        .def(py::init<const std::vector<std::vector<int>> &, int, int>())
        .def("create_agents", &ObservationGenerator::create_agents)
        .def("update_agents", &ObservationGenerator::update_agents)
        .def("generate_observations", &ObservationGenerator::generate_observations)
        .def("get_agents_ids_in_observations", &ObservationGenerator::get_agents_ids_in_observations);
}
/*
<%
cfg['extra_compile_args'] = ['-std=c++17', '-m64']
cfg['extra_link_args'] = ['-m64']
setup_pybind11(cfg)
%>
*/
#endif
