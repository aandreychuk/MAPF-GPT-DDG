/*
 * utility functions
 */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

using Time = std::chrono::steady_clock;

// time manager
struct Deadline {
  const Time::time_point t_s;
  const double time_limit_ms;

  Deadline(double _time_limit_ms = 0);
  double elapsed_ms() const;
  double elapsed_ns() const;
};

double elapsed_ms(const Deadline *deadline);
double elapsed_ns(const Deadline *deadline);
bool is_expired(const Deadline *deadline);
bool is_expired(const Deadline &deadline);

float get_random_float(std::mt19937 &MT, float from = 0, float to = 1);
float get_random_float(std::mt19937 *MT, float from = 0, float to = 1);
int get_random_int(std::mt19937 &MT, int from = 0, int to = 1);
int get_random_int(std::mt19937 *MT, int from = 0, int to = 1);

template <typename Head, typename... Tail>
void info(const int level, const int verbose, Head &&head, Tail &&...tail);

void info(const int level, const int verbose);

template <typename Head, typename... Tail>
void info(const int level, const int verbose, Head &&head, Tail &&...tail)
{
  if (verbose < level) return;
  std::cout << head;
  info(level, verbose, std::forward<Tail>(tail)...);
}

template <typename... Body>
void info(const int level, const int verbose, const Deadline *deadline,
          Body &&...body)
{
  if (verbose < level) return;
  std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms  ";
  info(level, verbose, (body)...);
}

template <typename... Body>
void warn(Body &&...body)
{
  info(0, 0, "[warning] ", (body)...);
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &arr)
{
  for (auto ele : arr) os << ele << ",";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::list<int> &arr)
{
  for (auto ele : arr) os << ele << ",";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::set<int> &arr)
{
  for (auto ele : arr) os << ele << ",";
  return os;
}
