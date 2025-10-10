// cppimport
/*
<%
import sys
import pybind11
setup_pybind11(cfg)
cfg['include_dirs'] = ['include', pybind11.get_include()]
cfg['sources'] = [
    'src/dist_table.cpp',
    'src/graph.cpp',
    'src/instance.cpp',
    'src/lacam.cpp',
    'src/metrics.cpp',
    'src/pibt.cpp',
    'src/planner.cpp',
    'src/post_processing.cpp',
    'src/utils.cpp',
]

if sys.platform == 'darwin':
    # macOS/Clang: no -m64/-pthread; ensure libc++
    cfg['compiler_args'] = ['-std=c++17', '-O3', '-fPIC', '-stdlib=libc++']
    cfg['extra_link_args'] = ['-stdlib=libc++']
    cfg['linker_args'] = []
else:
    # Linux defaults
    cfg['compiler_args'] = ['-std=c++17', '-O3', '-fPIC', '-m64']
    cfg['extra_link_args'] = ['-m64', '-pthread']
    cfg['linker_args'] = []
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "planner.hpp"

namespace py = pybind11;

// Simple stateful wrapper to hold instance and parameters between calls
class LaCAMWrapper {
 public:
  LaCAMWrapper() : seed_(0), verbose_(0), time_limit_ms_(0.0), dist_multi_thread_init_(true), anytime_(false), pibt_swap_(true), pibt_hindrance_(true) {}

  void init(const std::vector<std::vector<int>> &grid,  // 0=free, 1=blocked
            const std::vector<std::pair<int, int>> &starts_rc,  // (row, col)
            const std::vector<std::pair<int, int>> &goals_rc,   // (row, col)
            int seed = 0,
            int verbose = 0,
            double time_limit_sec = 0.0,
            bool multi_thread_dist_init = true,
            bool anytime = false,
            bool pibt_swap = true,
            bool pibt_hindrance = true)
  {
    const int h = (int)grid.size();
    const int w = h > 0 ? (int)grid[0].size() : 0;
    std::vector<int> occupancy;
    occupancy.reserve(w * h);
    for (int y = 0; y < h; ++y) {
      if ((int)grid[y].size() != w) throw std::runtime_error("grid rows must have equal width");
      for (int x = 0; x < w; ++x) occupancy.push_back(grid[y][x]);
    }

    graph_ = std::make_unique<Graph>(w, h, occupancy);

    // build starts and goals using Graph::U lookup
    Config starts;
    Config goals;
    for (auto &p : starts_rc) {
      int y = p.first;   // row
      int x = p.second;  // col
      if (x < 0 || x >= w || y < 0 || y >= h) throw std::runtime_error("start out of bounds");
      auto v = graph_->U[w * y + x];
      if (v == nullptr) throw std::runtime_error("start on blocked cell");
      starts.push_back(v);
    }
    for (auto &p : goals_rc) {
      int y = p.first;   // row
      int x = p.second;  // col
      if (x < 0 || x >= w || y < 0 || y >= h) throw std::runtime_error("goal out of bounds");
      auto v = graph_->U[w * y + x];
      if (v == nullptr) throw std::runtime_error("goal on blocked cell");
      goals.push_back(v);
    }
    if (starts.size() != goals.size()) throw std::runtime_error("starts and goals must have same length");

    instance_ = std::make_unique<Instance>(std::move(*graph_), starts, goals);
    graph_.reset();

    // store params
    seed_ = seed;
    verbose_ = verbose;
    time_limit_ms_ = time_limit_sec > 0.0 ? time_limit_sec * 1000.0 : 0.0;
    dist_multi_thread_init_ = multi_thread_dist_init;
    anytime_ = anytime;
    pibt_swap_ = pibt_swap;
    pibt_hindrance_ = pibt_hindrance;
  }

  // Returns list of trajectories; each trajectory is a list of (x, y)
  std::vector<std::vector<std::pair<int, int>>> get_solution()
  {
    if (!instance_) throw std::runtime_error("call init() first");

    DistTable::MULTI_THREAD_INIT = dist_multi_thread_init_;
    LaCAM::ANYTIME = anytime_;
    PIBT::SWAP = pibt_swap_;
    PIBT::HINDRANCE = pibt_hindrance_;

    std::unique_ptr<Deadline> deadline_ptr;
    const Deadline *deadline = nullptr;
    if (time_limit_ms_ > 0.0) {
      deadline_ptr = std::make_unique<Deadline>(time_limit_ms_);
      deadline = deadline_ptr.get();
    }

    auto solution = solve(*instance_, verbose_, deadline, seed_);

    // Convert to trajectories of (x,y)
    // solution: vector<Config>; take per-agent positions over time
    const size_t T = solution.size();
    const size_t N = instance_->N;
    std::vector<std::vector<std::pair<int, int>>> trajectories(N);
    for (size_t i = 0; i < N; ++i) trajectories[i].reserve(T);
    for (size_t t = 0; t < T; ++t) {
      const auto &config = solution[t];
      for (size_t i = 0; i < N; ++i) {
        auto v = config[i];
        trajectories[i].push_back({v->x, v->y});
      }
    }
    return trajectories;
  }

 private:
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<Instance> instance_;
  int seed_;
  int verbose_;
  double time_limit_ms_;
  bool dist_multi_thread_init_;
  bool anytime_;
  bool pibt_swap_;
  bool pibt_hindrance_;
};

PYBIND11_MODULE(lacam_py, m)
{
  m.doc() = "pybind11 bindings for LaCAM";

  py::class_<LaCAMWrapper>(m, "LaCAM")
      .def(py::init<>())
      .def("init", &LaCAMWrapper::init, py::arg("grid"), py::arg("starts_xy"), py::arg("goals_xy"),
           py::arg("seed") = 0, py::arg("verbose") = 0, py::arg("time_limit_sec") = 0.0,
           py::arg("multi_thread_dist_init") = true, py::arg("anytime") = false,
           py::arg("pibt_swap") = true, py::arg("pibt_hindrance") = true)
      .def("get_solution", &LaCAMWrapper::get_solution);
}


