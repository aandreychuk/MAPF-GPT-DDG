// cppimport
#pragma once
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <bitset>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <fstream>
#define PYBIND11_MODULE
#ifdef PYBIND11_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#endif

struct HashPair
{
    uint64_t operator()(const std::pair<int, int>& p) const {
        return (uint64_t(p.first) << 32) | uint64_t(p.second);
    }
};

struct Agent
{
    std::pair<int, int> pos;
    std::pair<int, int> goal;
    int8_t last_action;
    int8_t next_action;
};

class ObservationGenerator
{
public:
    std::vector<Agent> agents;
    int obs_radius;
    int context_size;
    std::vector<std::vector<int>> agents_locations;
    std::vector<std::vector<std::vector<int>>> obs_buffer; // Buffer for each agent
    std::vector<std::vector<int>> grid;
    std::unordered_map<std::pair<int, int>, std::vector<std::vector<int8_t>>, HashPair> cost2go_cache;
    ObservationGenerator(const std::vector<std::vector<int>> &grid, int obs_radius, int context_size)
        : grid(grid), obs_radius(obs_radius), context_size(context_size)
    {
        agents_locations = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid[0].size(), -1));
    }
    ~ObservationGenerator() {}
    void compute_cost2go_for_goal(const std::pair<int, int> &goal);
    void generate_cost2go_obs(int agent_idx, std::vector<std::vector<int>> &buffer);
    int8_t compute_direction_flags(const std::vector<std::vector<uint16_t>> &distances, int i, int j);
    void create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals);
    void update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions);
    std::vector<std::vector<int>> generate_observations();
    pybind11::array_t<int> generate_observations_numpy();
};
