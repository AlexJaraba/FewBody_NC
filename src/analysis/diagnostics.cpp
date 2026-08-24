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

Diagnostics computeDiagnostics(const std::vector<Body>& bodies, double G, double dt) {
    Diagnostics d{};
    d.timestep = dt;
    double total_mass = 0.0;
    Vec3 mass_weighted_position;

    // Compute kinetic energy
    for (const auto& body : bodies) {
        if (body.mass <= 0.0) {
            throw std::runtime_error("computeDiagnostics requires positive body masses.");
        }
        d.kinetic_energy += body.kineticEnergy();
        d.linear_momentum_vec += body.momentum;
        d.angular_momentum_vec += cross(body.position, body.momentum);
        total_mass += body.mass;
        mass_weighted_position += body.mass * body.position;
    }

    // Compute potential energy
    const int N = static_cast<int>(bodies.size());
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const Vec3 dr = bodies[j].position - bodies[i].position;
            const double r = dr.norm();
<<<<<<< Updated upstream
            if (r < 1e-14) {
                continue;
=======
            if (!std::isfinite(r) || r <= 0.0) {
                throw std::runtime_error("computeDiagnostics encountered a non-finite or zero pair separation.");
>>>>>>> Stashed changes
            }
            d.potential_energy -= (G * bodies[i].mass * bodies[j].mass) / r;
        }
    }

    d.total_energy = d.kinetic_energy + d.potential_energy;

    if (total_mass > 0.0) {
        d.center_of_mass = mass_weighted_position / total_mass;
        d.center_of_mass_velocity = d.linear_momentum_vec / total_mass;
    }
    d.linear_momentum = d.linear_momentum_vec.norm();
    d.angular_momentum = d.angular_momentum_vec.norm();
    d.com_drift = d.center_of_mass.norm();
    // Second-order shadow Hamiltonian estimate
    // double p2sum = 0.0;
    // for (const auto& body : bodies) {
    //     p2sum += body.momentumMagnitudeSquared();
    // }

    d.shadow_energy = d.total_energy;

    return d;
}

PairDiagnostics computePairDiagnostics(const std::vector<Body>& bodies, int i, int j, double G) {
    if (i < 0 || j < 0) {
        throw std::runtime_error("computePairDiagnostics received a negative index.");
    }
    if (i == j) {
        throw std::runtime_error("computePairDiagnostics received identical pair indices.");
    }
    if (i >= static_cast<int>(bodies.size()) || j >= static_cast<int>(bodies.size())) {
        throw std::runtime_error("computePairDiagnostics pair index is out of range.");
    }

    const HernandezPairState pair = HernandezPairState::pairState(bodies, i, j);
    PairDiagnostics diagnostics;

    diagnostics.i = i;
    diagnostics.j = j;

    diagnostics.energy = pair.twoBodyEnergy(G);
    diagnostics.angular_momentum = pair.twoBodyAngularMomentum();
    diagnostics.total_momentum = pair.totalMomentum();
    diagnostics.com_position = pair.com_position;
    diagnostics.com_velocity = pair.com_velocity;
    diagnostics.angular_momentum_norm = diagnostics.angular_momentum.norm();
    diagnostics.total_momentum_norm = diagnostics.total_momentum.norm();
    diagnostics.com_position_norm = diagnostics.com_position.norm();
    diagnostics.com_velocity_norm = diagnostics.com_velocity.norm();

    return diagnostics;
}

PairDiagnosticDeviation comparePairDiagnostics(const PairDiagnostics& current, const PairDiagnostics& reference) {
    PairDiagnosticDeviation deviation;

    deviation.energy_error = std::abs(current.energy - reference.energy);
    deviation.angular_momentum_error = (current.angular_momentum - reference.angular_momentum).norm();
    deviation.total_momentum_error = (current.total_momentum - reference.total_momentum).norm();
    deviation.com_position_error = (current.com_position - reference.com_position).norm();
    deviation.com_velocity_error = (current.com_velocity - reference.com_velocity).norm();

    return deviation;
}

void printPairDiagnostics(std::ostream& os, const PairDiagnostics& diagnostics, const char* label) {
    os << label << "\n";
    os << "Pair: (" << diagnostics.i << ", " << diagnostics.j << ")\n";
    os << "Pair energy: " << diagnostics.energy << "\n";
    os << "Pair angular momentum norm: " << diagnostics.angular_momentum_norm << "\n";
    os << "Pair total momentum norm: " << diagnostics.total_momentum_norm << "\n";
    os << "Pair COM position norm: " << diagnostics.com_position_norm << "\n";
    os << "Pair COM velocity norm: " << diagnostics.com_velocity_norm << "\n";
}

void printPairDiagnosticsDeviation(std::ostream& os, const PairDiagnosticDeviation& deviation, const char* label) {
    os << label << "\n";
    os << "Pair energy error: " << deviation.energy_error << "\n";
    os << "Pair angular momentum error: " << deviation.angular_momentum_error << "\n";
    os << "Pair total momentum error: " << deviation.total_momentum_error << "\n";
    os << "Pair COM position error: " << deviation.com_position_error << "\n";
    os << "Pair COM velocity error: " << deviation.com_velocity_error << "\n";    
}