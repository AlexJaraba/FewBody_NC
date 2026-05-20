#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "solver.h"
#include "io.h"
#include "globals.h"
#include "body.h"
#include "csv_output_writer.h"
#include "diagnostics.h"
#include "hernandez.h"
#include "yoshida4.h"
#include "pairing.h"
#include "pair_graph.h"
#include "jacobi.h"
#include "canonical_state.h"

Solver::Solver(std::vector<Body>& bodies, CSVOutputWriter& writer) : bodies(bodies), integrator(nullptr), writer(writer) {

    SolverParams params = readParams("data/param.txt");
    
    PairGraph graph = build_hierarchical_pair_graph(bodies);
    std::vector<Pair> fixed_pairs = graph.kepler_pairs;

    std::cout << "Kepler Pairs:" << graph.kepler_pairs.size() << std::endl;
    std::cout << "Pertubation pairs:" << graph.pertubation_pairs.size() << std::endl;

    if (params.integrator == "leapfrog") {
        integrator = std::make_unique<Leapfrog>();
    }

    if (params.integrator == "hernandez") {
        integrator = std::make_unique<Hernandez>(fixed_pairs);
    }

    if (params.integrator == "Yoshida4") {
        integrator = std::make_unique<Yoshida4>(fixed_pairs);
    }

    if (!integrator) {
        throw std::runtime_error("Invalid integrator specified");
    }
    // Add more integrator options here as needed
}

void recenter_system(std::vector<Body>& bodies) {
    double total_mass = 0.0;
    std::vector<double> com(3, 0.0);
    std::vector<double> com_velocity(3, 0.0);

    for (const auto& body : bodies) {
        total_mass += body.mass;
        for (int k = 0; k < 3; ++k) {
            com[k] += body.position[k] * body.mass;
            com_velocity[k] += body.velocity[k] * body.mass;
        }
    }

    for (int k = 0; k < 3; ++k) {
        com[k] /= total_mass;
        com_velocity[k] /= total_mass;
    }

    for (auto& body : bodies) {
        for (int k = 0; k < 3; ++k) {
            body.position[k] -= com[k];
            body.velocity[k] -= com_velocity[k];
        }
        body.updateMomentumFromVelocity();
    }
}

void Solver::run() {
    SolverParams params = readParams("data/param.txt");

    int output_frequency = params.output_frequency;
    double runtime = params.runtime;
    double dt = params.timestep;
    int steps = static_cast<int>(runtime / dt);

    G = params.gravitational_constant;  // Set the global gravitational constant

    CanonicalState state = compute_jacobi_state(bodies);

    for (int step = 0; step < steps; ++step) {
        integrator->step(state, dt);
        reconstruct_bodies(state, bodies);
        Diagnostics diag = compute_diagnostics(bodies, G, dt);
        std::cout << "Step: " << step << ", Time: " << step * dt 
                  << ", | Total Energy: " << diag.total_energy 
                  << ", | Linear Momentum: " << diag.linear_momentum 
                  << ", | Angular Momentum: " << diag.angular_momentum 
                  << ", | Shadow Energy: " << diag.shadow_energy 
                  << ", | COM Drift: " << diag.com_drift
                  << std::endl;

        // Write output at the specified frequency
        if (step % output_frequency == 0) {
            std::vector<BodyState> states;
            for (const auto& body : bodies) {
                states.push_back(body.toState(step * dt));
            }
	        writer.write(states);
        }
    }
    reconstruct_bodies(state, bodies);
    std::vector<BodyState> states;
    for (const auto& b : bodies) {
        states.push_back(b.toState(steps * dt));
    }
    // Write final output
    writer.write(states);
}

void Solver::ReversibilityTest() {
    std::cout << "\n=== REVERSIBILITY TEST ===\n";

    // Save initial state
    std::vector<Body> initial = bodies;

    const double dt = 0.01;
    const int steps = 10000;

    CanonicalState state = compute_jacobi_state(bodies);

    // Forward integration
    for (int i = 0; i < steps; ++i) {
        integrator->step(state, dt);
    }

    // Reverse velocities
    for (size_t i = 1; i < state.P.size(); ++i) {
        for (int k = 0; k < 3; ++k) {
            state.P[i][k] *= -1.0;
        }
    }

    // Backward integration
    for (int i = 0; i < steps; ++i) {
        integrator->step(state, dt);
    }

    reconstruct_bodies(state, bodies);

    // Reverse again
    for (auto& body : bodies) {
        for (int k = 0; k < 3; ++k) {
            body.velocity[k] *= -1.0;
        }
        body.updateMomentumFromVelocity();
    }

    double max_pos_error = 0.0;
    double max_vel_error = 0.0;

    for (size_t i = 0; i < bodies.size(); ++i) {
        double pos_err = 0.0;
        double vel_err = 0.0;
        for (int k = 0; k < 3; ++k) {
            double dp = bodies[i].position[k] - initial[i].position[k];
            double dv = bodies[i].velocity[k] - initial[i].velocity[k];
            pos_err += dp * dp;
            vel_err += dv * dv;
        }

        pos_err = std::sqrt(pos_err);
        vel_err = std::sqrt(vel_err);
        max_pos_error = std::max(max_pos_error, pos_err);
        max_vel_error = std::max(max_vel_error, vel_err);
    }

    std::cout << "Max Position Error: " << max_pos_error << "\n";
    std::cout << "Max Velocity Error: " << max_vel_error << "\n";
}
