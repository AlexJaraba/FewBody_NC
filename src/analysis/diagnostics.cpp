#include <cmath>
#include <stdexcept>
#include <ostream>

#include "analysis/diagnostics.h"
#include "math/vec3.h"
#include "integrators-helper/hernandez/pair_state.h"



/* ====================================================================================================

    Diagnostics

    Computes physical diagnostics from Cartesian body states:
        - Total Energy
        - Linear Momentum
        - Angular Momentum
        - Center-of-mass Drift
        - Shadow-energy placeholder/estimate
    
    Diagnostics are always computed in Cartesian coordinates, even when the integrator evolves the system internally in Jacobi coordinates.

   ==================================================================================================== */

Diagnostics compute_diagnostics(const std::vector<Body>& bodies, double G, double dt) {
    Diagnostics d{};

    // Compute kinetic energy
    for (const auto& body : bodies) {
        d.kinetic_energy += body.kineticEnergy();
    }

    // Compute potential energy
    const int N = bodies.size();
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec3 dr = bodies[j].position - bodies[i].position;
            double r = dr.norm();
            if (r < 1e-14) {
                continue;
            }
            d.potential_energy -= (G * bodies[i].mass * bodies[j].mass) / r;
        }
    }

    d.total_energy = d.kinetic_energy + d.potential_energy;

    // Compute linear momentum
    Vec3 P;
    for (const auto& body : bodies) {
        P += body.momentum;
    }

    d.linear_momentum = P.norm();

    // Compute angular momentum
    Vec3 L;
    for (const auto& body : bodies) {
        L += cross(body.position, body.momentum);
    }

    d.angular_momentum = L.norm();

    // Compute center of mass drift
    Vec3 Rcm;
    double total_mass = 0.0;
    for (const auto& body : bodies) {
        total_mass += body.mass;
        Rcm += body.mass * body.position;
    }

    Rcm /= total_mass;
    d.com_drift = Rcm.norm();

    // Second-order shadow Hamiltonian estimate
    // double p2sum = 0.0;
    // for (const auto& body : bodies) {
    //     p2sum += body.momentumMagnitudeSquared();
    // }

    d.shadow_energy = d.total_energy;

    d.timestep = dt;

    return d;
}

PairDiagnostics compute_pair_diagnostics(const std::vector<Body>& bodies, int i, int j, double G) {
    if (i < 0 || j < 0) {
        throw std::runtime_error("compute_pair_diagnostics received a negative index.");
    }
    if (i == j) {
        throw std::runtime_error("compute_pair_diagnostics received identical pair indices.");
    }
    if (i >= static_cast<int>(bodies.size()) || j >= static_cast<int>(bodies.size())) {
        throw std::runtime_error("compute_pair_diagnostics pair index is out of range.");
    }

    const HernandezPairState pair = HernandezPairState::from_bodies(bodies, i, j);
    PairDiagnostics diagnostics;

    diagnostics.i = i;
    diagnostics.j = j;

    diagnostics.energy = pair.two_body_energy(G);
    diagnostics.angular_momentum = pair.two_body_angular_momentum();
    diagnostics.total_momentum = pair.total_momentum();
    diagnostics.com_position = pair.com_position;
    diagnostics.com_velocity = pair.com_velocity;
    diagnostics.angular_momentum_norm = diagnostics.angular_momentum.norm();
    diagnostics.total_momentum_norm = diagnostics.total_momentum.norm();
    diagnostics.com_position_norm = diagnostics.com_position.norm();
    diagnostics.com_velocity_norm = diagnostics.com_velocity.norm();

    return diagnostics;
};

PairDiagnosticDeviation compare_pair_diagnostics(const PairDiagnostics& current, const PairDiagnostics& reference) {
    PairDiagnosticDeviation deviation;

    deviation.energy_error = std::abs(current.energy - reference.energy);
    deviation.angular_momentum_error = (current.angular_momentum - reference.angular_momentum).norm();
    deviation.total_momentum_error = (current.total_momentum - reference.total_momentum).norm();
    deviation.com_position_error = (current.com_position - reference.com_position).norm();
    deviation.com_velocity_error = (current.com_velocity - reference.com_velocity).norm();

    return deviation;
};

void print_pair_diagnostics(std::ostream& os, const PairDiagnostics& diagnostics, const char* label) {
    os << label << "\n";
    os << "Pair: (" << diagnostics.i << ", " << diagnostics.j << ")\n";
    os << "Pair energy: " << diagnostics.energy << "\n";
    os << "Pair angular momentum norm: " << diagnostics.angular_momentum_norm << "\n";
    os << "Pair total momentum norm: " << diagnostics.total_momentum_norm << "\n";
    os << "Pair COM position norm: " << diagnostics.com_position_norm << "\n";
    os << "Pair COM velocity norm: " << diagnostics.com_velocity_norm << "\n";
};

void print_pair_diagnostics_deviation(std::ostream& os, const PairDiagnosticDeviation& deviation, const char* label) {
    os << label << "\n";
    os << "Pair energy error: " << deviation.energy_error << "\n";
    os << "Pair angular momentum error: " << deviation.angular_momentum_error << "\n";
    os << "Pair total momentum error: " << deviation.total_momentum_error << "\n";
    os << "Pair COM position error: " << deviation.com_position_error << "\n";
    os << "Pair COM velocity error: " << deviation.com_velocity_error << "\n";    
};