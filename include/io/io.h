#pragma once

#include <vector>
#include <string>

#include "core/body.h"

struct SolverParams {
  int output_frequency;
  double runtime;
  double timestep;
  std::string integrator;
  double gravitational_constant;
  std::string pair_order;
};

void readInitialConditions(const std::string& filename, std::vector<Body>& bodies);
SolverParams readParams(const std::string& filename);
