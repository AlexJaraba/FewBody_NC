#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "core/body.h"

struct SolverParams {
    std::uint64_t output_frequency = 100;
    double runtime = 10.0;
    double timestep = 0.01;
    double gravitational_constant = 0.000296014912;
    std::string integrator = "hernandez";
    std::string pair_order = "canonical";
};

[[nodiscard]] std::vector<Body> load_initial_conditions(const std::string& filename);
[[nodiscard]] SolverParams load_solver_params(const std::string& filename);
