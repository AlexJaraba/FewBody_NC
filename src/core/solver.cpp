#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>

#include "core/solver.h"
#include "io/diagnostics_writer.h"
#include "analysis/diagnostics.h"
#include "integrators/leapfrog.h"
#include "integrators/hernandez.h"
#include "math/vec3.h"

/* ======================================================================
Solver

Central simulation driver for FewBodyNC.

Responsibilities:
    - Read simulation parameters
    - Select the coordinate mode: Jacobi or Cartesian
    - Select the canonical integrator for Jacobi mode
    - Advance the system
    - Write output.csv and diagnostics.csv

The production solver advances one fixed global timestep at a time.

   ====================================================================== */

namespace {
    constexpr double AUTO_HIERARCHY_RATIO_THRESHOLD = 100.0;
    constexpr bool AUTO_MAY_SELECT_STRENGTH = false;
    std::vector<Pair> make_all_physical_pairs(const std::vector<Body>& bodies) {
        std::vector<Pair> pairs;
        const int body_count = static_cast<int>(bodies.size());
        for (int i = 0; i < body_count; ++i) {
            for (int j = i + 1; j < body_count; ++j) {
                pairs.push_back({i, j});
            }
        }
        return pairs;
    }
    void validate_finite_bodies(const std::vector<Body>& bodies, const char* context, int step, double time) {
        for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
            const Body& body = bodies[i];
            if (!std::isfinite(body.mass) || !body.position.is_finite() || !body.velocity.is_finite() || !body.momentum.is_finite() || !body.acceleration.is_finite()) {
                std::ostringstream msg;
                msg << std::setprecision(17);
                msg << "Non-finite body state detected. "
                    << "failed_step = " << step << ", "
                    << "failed_time = " << time << ", "
                    << "body = " << i << ", "
                    << "context = " << context << ", "
                    << "mass = " << body.mass << ", "
                    << "position = (" << body.position.x << ", " << body.position.y << ", " << body.position.z << "), "
                    << "velocity = (" << body.velocity.x << ", " << body.velocity.y << ", " << body.velocity.z << "), "
                    << "acceleration = (" << body.acceleration.x << ", " << body.acceleration.y << ", " << body.acceleration.z << "), "
                    << "momentum = (" << body.momentum.x << ", " << body.momentum.y << ", " << body.momentum.z << ")";
                throw std::runtime_error(msg.str());
            }
        }
    }
}

Solver::Solver(std::vector<Body>& bodies_, CSVOutputWriter& writer_) : bodies(bodies_), integrator(nullptr), writer(writer_) {
    const SolverParams params = readParams("data/param.txt");
    fixed_pairs.clear();
    fixed_pairs = canonicalize_pairs(make_all_physical_pairs(bodies));

    effective_pair_order = "canonical";
    hierarchy_ratio = strongest_pair_strength_ratio(fixed_pairs, bodies);

    if (params.integrator == "hernandez") {
        if (params.pair_order == "canonical") {
            fixed_pairs = canonicalize_pairs(fixed_pairs);
            effective_pair_order = "canonical";
        } else if (params.pair_order == "strength") {
            fixed_pairs = order_pairs_by_strength(fixed_pairs, bodies);
            effective_pair_order = "strength";
        } else if (params.pair_order == "auto") {
            if (AUTO_MAY_SELECT_STRENGTH && hierarchy_ratio >= AUTO_HIERARCHY_RATIO_THRESHOLD) {
                fixed_pairs = order_pairs_by_strength(fixed_pairs, bodies);
                effective_pair_order = "strength";
            } else {
                fixed_pairs = canonicalize_pairs(fixed_pairs);
                effective_pair_order = "canonical";
            }
        } else {
            throw std::runtime_error("Invalid pair_order. Use 'canonical', 'strength', or 'auto'.");
        }
    }

    std::cout << "Physical Pairs Used For Full Perturbation Split: " << fixed_pairs.size() << std::endl;

    // Integrators
    if (params.integrator == "leapfrog") {
        integrator = std::make_unique<Leapfrog>(fixed_pairs);
    }
    else if (params.integrator == "hernandez") {
        integrator = std::make_unique<Hernandez>(fixed_pairs);
    }
    else {
        throw std::runtime_error("Invalid integrator specified");
    }
        // Add more integrator options here as needed
}

void recenter_system(std::vector<Body>& bodies) {
    if (bodies.empty()) {
        throw std::runtime_error("Cannot recenter an empty system.");
    }
    double total_mass = 0.0;
    Vec3 com;
    Vec3 com_velocity;

    for (const auto& body : bodies) {
        total_mass += body.mass;
        com += body.mass * body.position;
        com_velocity += body.mass * body.velocity;
    }

    if (total_mass <= 0.0) {
        throw std::runtime_error("Cannont recenter a non-positive-mass system.");
    }
    com /= total_mass;
    com_velocity /= total_mass;

    for (auto& body : bodies) {
        body.position -= com;
        body.velocity -= com_velocity;
        body.updateMomentumFromVelocity();
    }
}

// ==============================================================
// Main Simulation Loop and Output
// ==============================================================

/*
    Dispatch the simulation to the selected coordinate mode.

    Jacobi Mode:
        - Build a CanonicalState
        - Advances that state using the selected integrator
        - Reconstructs physical Body objects for output and diagnostics

    Cartesian Mode:
        - Evolves physical Body objects directly
        - Leapfrog uses Solver::cartesian_step()
        - Hernandez uses the same Hernandez integrator class through its body-state path
*/

void Solver::run() {
    const SolverParams params = readParams("data/param.txt");
    std::cout << "Reading param file..." << std::endl;
    std::cout << "dt = " << params.timestep << std::endl;
    std::cout << "Loaded timestep: " << params.timestep << std::endl;

    run_fixed_step(params);
}

void Solver::write_current_bodies(double time) {
    std::vector<BodyState> states;
    states.reserve(bodies.size());
    for (const Body& body : bodies) {
        states.push_back(body.toState(time));
    }

    writer.write(states);
}

// ==============================================================
// Specific Steps for Specified Coordinate Modes
// ==============================================================

/*
    One Cartesian integration step using a leapfrog method.
        v(t + dt/2) = v(t) + (dt/2) * a(t)
        r(t + dt) = r(t) + dt * v(t + dt/2)
        v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)
*/

void Solver::leapfrog_step(double dt, double G) {
    for (auto& body : bodies) {
        body.updateAcceleration(bodies, G);
    }
    for (auto& body : bodies) {
        body.velocity += 0.5 * dt * body.acceleration;
        body.position += dt * body.velocity;
    }
    for (auto& body : bodies) {
        body.updateAcceleration(bodies, G);
    }
    for (auto& body : bodies) {
        body.velocity += 0.5 * dt * body.acceleration;
        body.updateMomentumFromVelocity();
    }
}

/*
    Run the direct Cartesian integration path.
        - Uses a leapfrog method to evolve the Cartesian states directly.
        - Computes diagnostics at specified intervals.
*/

void Solver::run_fixed_step(const SolverParams& params) {
    const int output_frequency = params.output_frequency;
    const double runtime = params.runtime;
    const double dt = params.timestep;
    const double G = params.gravitational_constant;

    if (output_frequency <= 0) {
        throw std::runtime_error("output_frequency must be positive.");
    }
    if (!(runtime >= 0.0)) {
        throw std::runtime_error("runtime must be non-negative.");
    }
    if (!(dt > 0.0)) {
        throw std::runtime_error("timestep must be positive.");
    }
    if (!(G > 0.0)) {
        throw std::runtime_error("gravitational_constant must be positive.");
    }
    if (!integrator) {
        throw std::runtime_error("Integrator not initialized.");
    }

    const bool use_leapfrog = params.integrator == "leapfrog";
    const bool use_hernandez = params.integrator == "hernandez";
    if (!use_leapfrog && !use_hernandez) {
        throw std::runtime_error("Invalid integrator specified. Use 'leapfrog' or 'hernandez'.");
    }
    const int steps = static_cast<int>(std::ceil(runtime / dt));
    Hernandez hernandez_integrator(fixed_pairs);

    std::cout << "Integrator: " << params.integrator << '\n';
    std::cout << "Fixed timestep: " << dt << '\n';

    validate_finite_bodies(bodies, "initial_state", 0, 0.0);

    DiagnosticsWriter diagnostics_writer("diagnostics.csv");
    {
        const Diagnostics diagnostics = compute_diagnostics(bodies, G, dt);
        diagnostics_writer.write(0.0, diagnostics);
        write_current_bodies(0.0);
    }

    for (int step = 1; step <= steps; ++step) {
        try {
            if (use_hernandez) {
                hernandez_integrator.step(bodies, dt, G);
            } else {
                leapfrog_step(dt, G);
            }
        } catch (const std::exception& exc) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Integration failure. "
                << "failed_step = " << step << ", "
                << "failed_time = " << (step - 1) * dt << ", "
                << "failed_dt = " << dt << ", "
                << "integrator = " << params.integrator << ", "
                << "pair_order = " << params.pair_order << ", "
                << "effective_pair_order = " << effective_pair_order << ", "
                << "reason = " << exc.what();
            throw std::runtime_error(msg.str());
        }

        validate_finite_bodies(bodies, "cartesian_after_step", step, step * dt);

        if (step % output_frequency == 0 || step == steps) {
            const Diagnostics diag = compute_diagnostics(bodies, G, dt);
            diagnostics_writer.write(step * dt, diag);
            write_current_bodies(step * dt);
        }
    }
    diagnostics_writer.close();
}
