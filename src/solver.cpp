#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "solver.h"
#include "io.h"
#include "globals.h"
#include "body.h"
#include "csv_output_writer.h"

Solver::Solver(std::vector<Body>& bodies, CSVOutputWriter& writer) : bodies(bodies), integrator(nullptr), writer(writer) {

    SolverParams params = readParams("data/param.txt");

    if (params.integrator == "leapfrog") {
        integrator = std::make_unique<Leapfrog>();
    }

    if (params.integrator == "hernandez") {
        std::vector<Pair> fixed_pairs = {{0,1},{0,2}};
        integrator = std::make_unique<Hernandez>(fixed_pairs);
    }

    if (!integrator) {
        throw std::runtime_error("Invalid integrator specified");
    }
    // Add more integrator options here as needed
}

void Solver::run() {
    SolverParams params = readParams("data/param.txt");

    int output_frequency = params.output_frequency;
    double runtime = params.runtime;
    double dt = params.timestep;
    int steps = static_cast<int>(runtime / dt);

    G = params.gravitational_constant;  // Set the global gravitational constant

    for (int step = 0; step < steps; ++step) {
        integrator->step(bodies, dt);

        // Write output at the specified frequency
        if (step % output_frequency == 0) {
            std::vector<BodyState> states;
            for (const auto& body : bodies) {
                states.push_back(body.toState(step * dt));
            }
	        writer.write(states);
        }
    }

    std::vector<BodyState> states;
    for (const auto& b : bodies) {
        states.push_back(b.toState(steps * dt));
    }
    // Write final output
    writer.write(states);
}

void Solver::ReversibilityTest() {
    SolverParams params = readParams("data/param.txt");

    double runtime = params.runtime;
    double dt = params.timestep;

    int steps = static_cast<int>(runtime / dt);

    std::vector<Body> initial_bodies = bodies; // Save initial state

    for (int step = 0; step < steps; ++step) {
        integrator->step(bodies, dt);
    }

    for (auto& body: bodies) {
        for (int k = 0; k < 3; ++k) {
            body.velocity[k] *= -1.0; // Reverse velocities
        }
    }
    
    for (int step = 0; step < steps; ++step) {
        integrator->step(bodies, dt);
    }

    double max_position_error = 0.0;
    double max_velocity_error = 0.0;

    for (size_t i = 0; i < bodies.size(); ++i){

        double pos_err = 0.0;
        double vel_err = 0.0;

        for (int k = 0; k < 3; ++k){

            double dp = bodies[i].position[k] - initial_bodies[i].position[k];
            double dv = bodies[i].velocity[k] + initial_bodies[i].velocity[k];
            pos_err += std::abs(dp);
            vel_err += std::abs(dv);
        }

        pos_err = std::sqrt(pos_err);
        vel_err = std::sqrt(vel_err);

        max_position_error = std::max(max_position_error, pos_err);
        max_velocity_error = std::max(max_velocity_error, vel_err);
    }

    std::cout << "Reversibility Test Results:" << std::endl;
    std::cout << "Max Position Error: " << max_position_error << std::endl;
    std::cout << "Max Velocity Error: " << max_velocity_error << std::endl;
}
