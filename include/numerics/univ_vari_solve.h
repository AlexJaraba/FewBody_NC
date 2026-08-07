#pragma once

#include "math/vec3.h"

struct UniversalAnomalyResult {
    double anomaly = 0.0;
    int iterations = 0;
    bool converged = false;
};

[[nodiscard]] double stumpff_C(double argument);
[[nodiscard]] double stumpff_S(double argument);

[[nodiscard]] UniversalAnomalyResult solve_univeral_anomaly(
    double gravitational_parameter, 
    double reciprocal_semimajor_axis,
    const Vec3& initial_position,
    double initial_radial_velocity, 
    double timestep,
    double abs_tol = 1e-13,
    double rel_tol = 1e-13,
    int max_iterations = 100);