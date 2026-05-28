#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "core/solver.h"
#include "core/body.h"
#include "core/canonical_state.h"
#include "io/io.h"
#include "io/diagnostics_writer.h"
#include "io/csv_output_writer.h"
#include "analysis/diagnostics.h"
#include "integrators/hernandez.h"
#include "integrators/yoshida4.h"
#include "integrators/integrator.h"
#include "dynamics/pairing.h"
#include "dynamics/pair_graph.h"
#include "dynamics/jacobi.h"
#include "math/vec3.h"

Solver::Solver(std::vector<Body>& bodies, CSVOutputWriter& writer) : bodies(bodies), integrator(nullptr), writer(writer) {

    SolverParams params = readParams("data/param.txt");
    
    PairGraph graph = build_hierarchical_pair_graph(bodies);
    std::vector<Pair> fixed_pairs = graph.perturbation_pairs;

    std::cout << "Kepler Pairs: " << graph.kepler_pairs.size() << std::endl;
    std::cout << "Perturbation Pairs: " << graph.perturbation_pairs.size() << std::endl;

    if (params.integrator == "leapfrog") {
        integrator = std::make_unique<Leapfrog>(fixed_pairs);
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
    Vec3 com;
    Vec3 com_velocity;

    for (const auto& body : bodies) {
        total_mass += body.mass;
        com += body.mass * body.position;
        com_velocity += body.mass * body.velocity;
    }

    com /= total_mass;
    com_velocity /= total_mass;

    for (auto& body : bodies) {
        body.position -= com;
        body.velocity -= com_velocity;
        body.updateMomentumFromVelocity();
    }
}

void Solver::run() {
    std::cout << "Reading param file..." << std::endl;
    SolverParams params = readParams("data/param.txt");
    std::cout << "dt = " << params.timestep << std::endl;

    int output_frequency = params.output_frequency;
    double runtime = params.runtime;
    double dt = params.timestep;
    std::cout << "Loaded timestep: " << dt << std::endl;

    int steps = static_cast<int>(runtime / dt);

    double G = params.gravitational_constant;  // Set the global gravitational constant

    CanonicalState state = compute_jacobi_state(bodies);

    DiagnosticsWriter dianostics_writer("diagnostics.csv");

    for (int step = 0; step < steps; ++step) {
        integrator->step(state, dt, G);
        reconstruct_bodies(state, bodies);

        if (step % output_frequency == 0) {
            Diagnostics diag = compute_diagnostics(bodies, G, dt);
            std::cout << "Step: " << step << ", Time: " << step * dt 
                    << ", | Total Energy: " << diag.total_energy 
                    << ", | Linear Momentum: " << diag.linear_momentum 
                    << ", | Angular Momentum: " << diag.angular_momentum 
                    << ", | Shadow Energy: " << diag.shadow_energy 
                    << ", | COM Drift: " << diag.com_drift
                    << std::endl;
            dianostics_writer.write(step * dt, diag);
        }

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
    dianostics_writer.close();
}

void Solver::TestHernandezAdjoint(double dt) {
    std::cout << "\n=== HERNANDEZ ADJOINT TEST ===\n";
    
    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    CanonicalState state = compute_jacobi_state(bodies);
    CanonicalState initial = state;

    // Forward step
    integrator->step(state, dt, G);

    // Backward step
    integrator->step(state, -dt, G);

    double max_q_error = 0.0;
    double max_p_error = 0.0;

    for (size_t i = 1; i < state.Q.size(); ++i) {
        Vec3 dQ = state.Q[i] - initial.Q[i];
        Vec3 dP = state.P[i] - initial.P[i];
        max_q_error = std::max(max_q_error, dQ.norm());
        max_p_error = std::max(max_p_error, dP.norm());

    std::cout << "Max Q Error: " << max_q_error << std::endl;
    std::cout << "Max P Error: " << max_p_error << std::endl;
    }
}

void Solver::TestLocalOrder() {
    std::cout << "\n=== LOCAL ORDER TEST ===\n";
    const double G = 0.000296014912;
    CanonicalState inital = compute_jacobi_state(bodies);
    CanonicalState reference = inital;
    const double dt_ref = 1e-6;
    const double T = 0.1;
    int ref_steps = static_cast<int>(T / dt_ref);
    for (int i = 0; i < ref_steps; ++i) {
        integrator->step(reference, dt_ref, G);
    }
    std::vector<double> dts = {0.1, 0.05, 0.025};
    std::vector<double> errors;
    for (double dt : dts) {
        CanonicalState test = inital;
        int steps = static_cast<int>(T / dt);
        for (int i = 0; i < steps; ++i) {
            integrator->step(test, dt, G);
        }

        double err = 0.0;

        for (size_t i = 1; i < test.Q.size(); ++i) {
           Vec3 dQ = test.Q[i] - reference.Q[i];
           Vec3 dP = test.P[i] - reference.P[i];
           err += dQ.norm2();
           err += dP.norm2();
        }

        err = std::sqrt(err);
        errors.push_back(err);
        std::cout << "dt: " << dt << ", Error: " << err << std::endl;
    }
    for (size_t i = 0; i < errors.size() - 1; ++i) {
        double ratio = errors[i] / errors[i + 1];
        std::cout << dts[i] << " -> " << dts[i + 1] << ", Error Ratio: " << ratio << std::endl;
    }
}

void Solver::ReversibilityTest() {
    std::cout << "\n=== REVERSIBILITY TEST ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    // Save initial state
    std::vector<Body> initial = bodies;

    const double dt = 0.01;
    const int steps = 10000;

    CanonicalState state = compute_jacobi_state(bodies);

    // Forward integration
    for (int i = 0; i < steps; ++i) {
        integrator->step(state, dt, G);
    }

    // Reverse velocities
    for (size_t i = 1; i < state.P.size(); ++i) {
        state.P[i] *= -1.0;
    }

    // Backward integration
    for (int i = 0; i < steps; ++i) {
        integrator->step(state, dt, G);
    }

    for (size_t i = 1; i < state.P.size(); ++i) {
        state.P[i] *= -1.0;
    }

    reconstruct_bodies(state, bodies);

    double max_pos_error = 0.0;
    double max_vel_error = 0.0;

    for (size_t i = 0; i < bodies.size(); ++i) {
        Vec3 dp = bodies[i].position - initial[i].position;
        Vec3 dv = bodies[i].velocity - initial[i].velocity;

        double pos_err = dp.norm();
        double vel_err = dv.norm();
        max_pos_error = std::max(max_pos_error, pos_err);
        max_vel_error = std::max(max_vel_error, vel_err);
    }
    std::cout << "Max Position Error: " << max_pos_error << "\n";
    std::cout << "Max Velocity Error: " << max_vel_error << "\n";
}
