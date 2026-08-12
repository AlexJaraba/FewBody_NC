#pragma once

#include <iosfwd>
#include <vector>

#include "core/body.h"
#include "math/vec3.h"

struct Diagnostics {
    double kinetic_energy = 0.0;
    double potential_energy = 0.0;
    double total_energy = 0.0;

    Vec3 linear_momentum_vec;
    Vec3 angular_momentum_vec;
    Vec3 center_of_mass;
    Vec3 center_of_mass_velocity;

    // Backward-compatible scalar aliases used by older Python paths.
    double angular_momentum = 0.0;
    double com_drift = 0.0;
    double linear_momentum = 0.0;
    double timestep = 0.0;
};

struct PairDiagnostics {
    int i = -1;
    int j = -1;
    double energy = 0.0;
    Vec3 angular_momentum;
    Vec3 total_momentum;
    Vec3 com_position;
    Vec3 com_velocity;
    double angular_momentum_norm = 0.0;
    double total_momentum_norm = 0.0;
    double com_position_norm = 0.0;
    double com_velocity_norm = 0.0;
};

struct PairDiagnosticDeviation {
    double energy_error = 0.0;
    double angular_momentum_error = 0.0;
    double total_momentum_error = 0.0;
    double com_position_error = 0.0;
    double com_velocity_error = 0.0;
};

Diagnostics compute_diagnostics(const std::vector<Body>& bodies, double G, double dt);
PairDiagnostics compute_pair_diagnostics(const std::vector<Body>& bodies, int i, int j, double G);
PairDiagnosticDeviation compare_pair_diagnostics(const PairDiagnostics& current, const PairDiagnostics& reference);

void print_pair_diagnostics(std::ostream& os, const PairDiagnostics& diagnostics, const char* label);
void print_pair_diagnostics_deviation(std::ostream& os, const PairDiagnosticDeviation& deviation, const char* label);