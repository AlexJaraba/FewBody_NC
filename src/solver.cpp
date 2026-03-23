#include <memory>
#include <iostream>
#include <stdexcept>

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
        integrator = std::make_unique<Hernandez>();
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

