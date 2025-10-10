#include "../include/instance.hpp"

Instance::~Instance() {}

Instance::Instance(Graph &&_G, const Config &_starts, const Config &_goals)
    : G(std::move(_G)), starts(_starts), goals(_goals), N(_starts.size())
{
}

Instance::Instance(const std::string &map_filename,
                   const std::vector<int> &start_indexes,
                   const std::vector<int> &goal_indexes)
    : G(map_filename),
      starts(Config()),
      goals(Config()),
      N(start_indexes.size())
{
  for (auto k : start_indexes) starts.push_back(G.U[k]);
  for (auto k : goal_indexes) goals.push_back(G.U[k]);
}

// for load instance
static const std::regex r_instance =
    std::regex(R"(\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+)");

Instance::Instance(const std::string &scen_filename,
                   const std::string &map_filename, const int _N)
    : G(map_filename), starts(0), goals(0), N(_N)
{
  // load start-goal pairs
  std::ifstream file(scen_filename);
  if (!file) {
    info(0, 0, scen_filename, " is not found");
    return;
  }
  std::string line;
  std::smatch results;

  const auto W = G.width;
  const auto H = G.height;

  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();

    if (std::regex_match(line, results, r_instance)) {
      auto x_s = std::stoi(results[1].str());
      auto y_s = std::stoi(results[2].str());
      auto x_g = std::stoi(results[3].str());
      auto y_g = std::stoi(results[4].str());
      if (x_s < 0 || W <= x_s || x_g < 0 || W <= x_g) continue;
      if (y_s < 0 || H <= y_s || y_g < 0 || H <= y_g) continue;
      auto s = G.U[W * y_s + x_s];
      auto g = G.U[W * y_g + x_g];
      if (s == nullptr || g == nullptr) continue;
      starts.push_back(s);
      goals.push_back(g);
    }

    if (starts.size() == N) break;
  }
}

Instance::Instance(const std::string &map_filename, const int _N,
                   const int seed)
    : G(map_filename), starts(0), goals(0), N(_N)
{
  // connectivity check
  // if (!is_connected(G)) {
  //   warn("graph <", map_filename, "> is not connected");
  // }

  auto MT = std::mt19937(seed);
  // random assignment
  const auto K = G.size();

  // set starts
  auto s_indexes = std::vector<int>(K);
  std::iota(s_indexes.begin(), s_indexes.end(), 0);
  std::shuffle(s_indexes.begin(), s_indexes.end(), MT);
  int i = 0;
  while (true) {
    if (i >= K) return;
    starts.push_back(G.V[s_indexes[i]]);
    if (starts.size() == N) break;
    ++i;
  }

  // set goals
  auto g_indexes = std::vector<int>(K);
  std::iota(g_indexes.begin(), g_indexes.end(), 0);
  std::shuffle(g_indexes.begin(), g_indexes.end(), MT);
  int j = 0;
  while (true) {
    if (j >= K) return;
    goals.push_back(G.V[g_indexes[j]]);
    if (goals.size() == N) break;
    ++j;
  }
}

bool Instance::is_valid(const int verbose) const
{
  if (N != starts.size() || N != goals.size()) {
    info(1, verbose, "invalid N, check instance");
    return false;
  }
  return true;
}
