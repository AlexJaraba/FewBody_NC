#pragma once

#include <vector>
#include <string>

#include "core/body.h"

struct SolverParams {
  int output_frequency;
  double runtime;
  double timestep;
  std::string integrator;
  std::string coordinate_mode;
  double gravitational_constant;

  bool adaptive_timesteps;
  int timestep_levels;
  double timestep_eta;
};

void readInitialConditions(const std::string& filename, std::vector<Body>& bodies);
void writeOutput(const std::string& filename, const std::vector<Body>& bodies, double time);
SolverParams readParams(const std::string& filename);
