#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>
#include <cstddef>

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
    - Select requested integrator
    - Advance the physical body state
    - Write output.csv and diagnostics.csv

The production solver advances one fixed global timestep at a time.

   ====================================================================== */

namespace {
    std::vector<Pair> makePhysicalPairs(const std::vector<Body>& bodies) {
        std::vector<Pair> pairs;
        const int body_count = static_cast<int>(bodies.size());
        for (int i = 0; i < body_count; ++i) {
            for (int j = i + 1; j < body_count; ++j) {
                pairs.push_back({i, j});
            }
        }
        return pairs;
    }
    void validateBodies(const std::vector<Body>& bodies, const char* context, std::uint64_t step, double time) {
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
    fixed_pairs = canonicalizePairs(makePhysicalPairs(bodies));
    
    effective_pair_order = "canonical";
    if (params.integrator == "hernandez") {
        if (params.pair_order == "canonical") {
            fixed_pairs = canonicalizePairs(fixed_pairs);
            effective_pair_order = "canonical";
        } else if (params.pair_order == "strength") {
            fixed_pairs = orderPairsStrength(fixed_pairs, bodies);
            effective_pair_order = "strength";
        } else if (params.pair_order == "auto") { // Auto mode: default to canonical for now, can be improved later
            fixed_pairs = canonicalizePairs(fixed_pairs);
            effective_pair_order = "canonical";
        } else {
            throw std::runtime_error("Invalid pair_order. Use 'canonical', 'strength', or 'auto'.");
        }
    }

    std::cout << "Physical Pairs: " << fixed_pairs.size() << std::endl;

    // Integrators
    if (params.integrator == "leapfrog") {
        integrator = std::make_unique<Leapfrog>();
    }
    else if (params.integrator == "hernandez") {
        integrator = std::make_unique<Hernandez>(fixed_pairs);
    }
    else {
        throw std::runtime_error("Invalid integrator specified");
    }
        // Add more integrator options here as needed
}

void recenterSystem(std::vector<Body>& bodies) {
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
    Run the simulation using the integrator selected in Solver constructor. 
    All active integroators evolve the physical Body state directly.
*/

void Solver::run() {
    const SolverParams params = readParams("data/param.txt");
    std::cout << "Reading param file..." << std::endl;
    std::cout << "dt = " << params.timestep << std::endl;
    std::cout << "Loaded timestep: " << params.timestep << std::endl;

    runFixedStep(params);
}

void Solver::writeCurrentBodies(double time) {
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

void Solver::runFixedStep(const SolverParams& params) {
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

    const double step_ratio = runtime / dt;
    if (!std::isfinite(step_ratio)) {
        throw std::runtime_error("runtime / timestep is not finite.");
    }
    const double nearest_integer_steps = std::round(step_ratio);
    const double rounding_tolerance = 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(step_ratio));
    const double requested_steps = (std::abs(step_ratio - nearest_integer_steps) <= rounding_tolerance) ? nearest_integer_steps : std::ceil(step_ratio);
    if (requested_steps > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::runtime_error("Requested number of steps exceeds maximum representable value.");
    }
    const std::uint64_t steps = static_cast<std::uint64_t>(requested_steps);
    const std::uint64_t output_stride = static_cast<std::uint64_t>(output_frequency);

    std::cout << "Integrator: " << params.integrator << '\n';
    std::cout << "Fixed timestep: " << dt << '\n';

    validateBodies(bodies, "initial_state", 0, 0.0);

    DiagnosticsWriter diagnostics_writer("diagnostics.csv");
    {
        const Diagnostics diagnostics = computeDiagnostics(bodies, G, dt);
        diagnostics_writer.write(0.0, diagnostics);
        writeCurrentBodies(0.0);
    }

    for (std::uint64_t step = 1; step <= steps; ++step) {
        try {
            integrator->step(bodies, dt, G);
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
        const double current_time = static_cast<double>(step) * dt;
        validateBodies(bodies, "cartesian_after_step", step, current_time);

        if (step % output_stride == 0 || step == steps) {
            const Diagnostics diag = computeDiagnostics(bodies, G, dt);
            diagnostics_writer.write(current_time, diag);
            writeCurrentBodies(current_time);
        }
    }
    diagnostics_writer.close();
}
