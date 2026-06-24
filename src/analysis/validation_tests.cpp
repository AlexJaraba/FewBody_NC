#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "analysis/diagnostics.h"
#include "analysis/validation_tests.h"
#include "core/canonical_state.h"
#include "core/reconstruction.h"
#include "core/solver.h"
#include "dynamics/jacobi.h"
#include "dynamics/pairing.h"
#include "dynamics/hierarchy_node.h"
#include "dynamics/hierarchy_tree.h"
#include "dynamics/timestep_planner.h"
#include "integrators/hernandez.h"
#include "integrators-helper/hernandez/pair_map.h"
#include "integrators-helper/hernandez/pair_state.h"
#include "integrators-helper/hernandez/state.h"
#include "io/io.h"
#include "math/vec3.h"

namespace {
    std::vector<Pair> make_all_physical_pairs(const std::vector<Body>& test_bodies) {
        std::vector<Pair> pairs;
        const int N = static_cast<int>(test_bodies.size());
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                pairs.push_back({i, j});
            }
        }
        return pairs;
    }

    std::vector<Body> make_hernandez_two_body_circular_test(double G) {
        const double m0 = 1.0;
        const double m1 = 1.0e-6;
        const double total_mass = m0 + m1;
        const double separation = 1.0;
        const double circular_speed = std::sqrt(G * total_mass / separation);

        std::vector<Body> test_bodies;
        test_bodies.emplace_back(m0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
        test_bodies.emplace_back(m1, Vec3(separation, 0.0, 0.0), Vec3(0.0, circular_speed, 0.0));

        return test_bodies;
    }

    std::vector<Body> make_hernandez_three_body_planet_test() {
        std::vector<Body> test_bodies;

        test_bodies.emplace_back(1.0, Vec3(-0.00239640539191213, 0.0, 0.0), Vec3(0.0, -2.23224877762317e-05, 0.0));
        test_bodies.emplace_back(0.001, Vec3(0.997603594608088, 0.0, 0.0), Vec3(0.0, 0.0171913618044373, 0.0));
        test_bodies.emplace_back(0.0005, Vec3(2.79760359460809, 0.0, 0.0), Vec3(0.0, 0.0102622519435887, 0.0));

        return test_bodies;       
    }

    void update_all_momenta(std::vector<Body>& test_bodies) {
        for(Body& body : test_bodies) {
            body.updateMomentumFromVelocity();
        }
    }
}

Tests::Tests(Solver& solver) : solver_(solver) {}

void Tests::TestHernandezAdjoint(double dt) {
    std::cout << "\n=== HERNANDEZ ADJOINT TEST ===\n";
    
    SolverParams params = readParams("data/param.txt");
    
    if (params.coordinate_mode != "jacobi") {
        std::cout << "Skipped: Hernandez adjoint test only applies to Jacobi mode.\n";
        return;
    }

    const double G = params.gravitational_constant;  // Set the global gravitational constant

    CanonicalState state = compute_jacobi_state(solver_.bodies);
    CanonicalState initial = state;

    // Forward step
    solver_.integrator->step(state, dt, G);
    // Backward step
    solver_.integrator->step(state, -dt, G);

    double max_q_error = 0.0;
    double max_p_error = 0.0;

    for (size_t i = 1; i < state.Q.size(); ++i) {
        Vec3 dQ = state.Q[i] - initial.Q[i];
        Vec3 dP = state.P[i] - initial.P[i];

        max_q_error = std::max(max_q_error, dQ.norm());
        max_p_error = std::max(max_p_error, dP.norm());
    }
    std::cout << "Max Q Error: " << max_q_error << std::endl;
    std::cout << "Max P Error: " << max_p_error << std::endl;
}

void Tests::TestLocalOrder() {
    std::cout << "\n=== LOCAL ORDER TEST ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");

    if (params.coordinate_mode != "jacobi") {
        std::cout << "Skipped: Local-order test currently applies to Jacobi mode.\n";
        return;
    }

    const double G = params.gravitational_constant;

    CanonicalState initial = compute_jacobi_state(solver_.bodies);
    const double dt_ref = 0.0003125;
    const double T = 10.0;
    CanonicalState reference = initial;

    int ref_steps = static_cast<int>(T / dt_ref);

    for (int n = 0; n < ref_steps; ++n) {
        solver_.integrator->step(reference, dt_ref, G);
    }

    std::vector<double> dts = {0.08, 0.04, 0.02, 0.01};
    std::vector<double> errors;

    for (double dt : dts) {
        CanonicalState test = initial;
        const int steps = static_cast<int>(T / dt);

        for (int i = 0; i < steps; ++i) {
            solver_.integrator->step(test, dt, G);
        }

        double err2 = 0.0;

        for (size_t i = 1; i < test.Q.size(); ++i) {
           Vec3 dQ = test.Q[i] - reference.Q[i];
           Vec3 dP = test.P[i] - reference.P[i];
           err2 += dQ.norm2();
           err2 += dP.norm2();
        }

        const double err = std::sqrt(err2);
        errors.push_back(err);
        std::cout << "dt: " << dt << ", Error: " << err << std::endl;
    }

    std::cout << "\nConvergence Ratios:\n";

    for (size_t i = 0; i + 1 < errors.size(); ++i) {
        std::cout << dts[i] << " -> " << dts[i + 1] << " : ratio = " << errors[i] / errors[i + 1] << std::endl;
    }
}

void Tests::ReversibilityTest() {
    std::cout << "\n=== REVERSIBILITY TEST ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    // Save initial state
    std::vector<Body> initial = solver_.bodies;

    const double dt = 0.01;
    const int steps = 10000;

    if (params.coordinate_mode == "cartesian") {
        for (int i = 0; i < steps; ++i) {
            solver_.cartesian_step(dt, G);
        }
        for (auto& body : solver_.bodies) {
            body.velocity *= -1.0;
            body.updateMomentumFromVelocity();
        }
        for (int i = 0; i < steps; ++i) {
            solver_.cartesian_step(dt, G);
        }
        for (auto& body : solver_.bodies) {
            body.velocity *= -1.0;
            body.updateMomentumFromVelocity();
        }
    }

    else if (params.coordinate_mode == "jacobi") {
        CanonicalState state = compute_jacobi_state(solver_.bodies);

        // Forward integration
        for (int i = 0; i < steps; ++i) {
            solver_.integrator->step(state, dt, G);
        }

        // Reverse velocities
        for (size_t i = 1; i < state.P.size(); ++i) {
            state.P[i] *= -1.0;
        }

        // Backward integration
        for (int i = 0; i < steps; ++i) {
            solver_.integrator->step(state, dt, G);
        }

        for (size_t i = 1; i < state.P.size(); ++i) {
            state.P[i] *= -1.0;
        }

        reconstruct_bodies(state, solver_.bodies);
    }

    else {
        throw std::runtime_error("Invalid coordinate_mode. Use 'jacobi' or 'cartesian'.");
    }

    double max_pos_error = 0.0;
    double max_vel_error = 0.0;

    for (size_t i = 0; i < solver_.bodies.size(); ++i) {
        Vec3 dp = solver_.bodies[i].position - initial[i].position;
        Vec3 dv = solver_.bodies[i].velocity - initial[i].velocity;

        double pos_err = dp.norm();
        double vel_err = dv.norm();
        max_pos_error = std::max(max_pos_error, pos_err);
        max_vel_error = std::max(max_vel_error, vel_err);
    }
    std::cout << "Max Position Error: " << max_pos_error << "\n";
    std::cout << "Max Velocity Error: " << max_vel_error << "\n";
}

void Tests::TestHernandezPairStateRoundTrip() {
    std::cout << "\n=== Hernandez Pair State Round-Trip Test ===\n";

    if (solver_.bodies.size() < 2) {
        std::cout << "Skipped: At least TWO bodies required.";
        return;
    }

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    const int i = 0;
    const int j = 1;

    std::vector<Body> initial = solver_.bodies;

    HernandezPairState before = HernandezPairState::from_bodies(solver_.bodies, i, j); // Store Before Change

    const double pair_energy_before = before.two_body_energy(G);
    const Vec3 pair_angular_momentum_before = before.two_body_angular_momentum();
    const Vec3 pair_total_momentum_before = before.total_momentum();

    before.write_to_bodies(solver_.bodies);

    HernandezPairState after = HernandezPairState::from_bodies(solver_.bodies, i, j); // Store After Change

    const double pair_energy_after = after.two_body_energy(G);
    const Vec3 pair_angular_momentum_after = after.two_body_angular_momentum();
    const Vec3 pair_total_momentum_after = after.total_momentum();

    // Error Calculations
    const double pos_i_error = (solver_.bodies[i].position - initial[i].position).norm();
    const double pos_j_error = (solver_.bodies[j].position - initial[j].position).norm();
    const double vel_i_error = (solver_.bodies[i].velocity - initial[i].velocity).norm();
    const double vel_j_error = (solver_.bodies[j].velocity - initial[j].velocity).norm();
    const double energy_error = std::abs(pair_energy_after - pair_energy_before);
    const double angular_momentum_error = (pair_angular_momentum_after - pair_angular_momentum_before).norm();
    const double total_momentum_error = (pair_total_momentum_after - pair_total_momentum_before).norm();

    // Terminal Statements
    std::cout << "Pair tested: (" << i << ", " << j << ")\n";
    std::cout << "Position error body i: " << pos_i_error << "\n";
    std::cout << "Position error body j: " << pos_j_error << "\n";
    std::cout << "Velocity error body i: " << vel_i_error << "\n";
    std::cout << "Velocity error body j: " << vel_j_error << "\n";
    std::cout << "Pair energy round-trip error: " << energy_error << "\n";
    std::cout << "Pair angular momentum round-trip error: " << angular_momentum_error << "\n";
    std::cout << "Pair total momentum round-trip error: " << total_momentum_error << "\n";

    solver_.bodies = initial;
}

void Tests::TestHernandezPairKeplerSuite() {
    std::cout << "\n=== Hernandez Pair Kepler Map Suite ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    const double pi = std::acos(-1.0);

    auto make_pair_bodies = [](double m0, double m1, const Vec3& q0, const Vec3& u0) {
        const double total_mass = m0 + m1;
        
        const Vec3 R0;
        const Vec3 V0;

        std::vector<Body> test_bodies;
        test_bodies.emplace_back(m0, R0 + (m1 / total_mass) * q0, V0 + (m1 / total_mass) * u0);
        test_bodies.emplace_back(m1, R0 - (m0 / total_mass) * q0, V0 - (m0 / total_mass) * u0);

        return test_bodies;
    };
    auto run_case = [&](const std::string& name, double m0, double m1, const Vec3& q0, const Vec3& u0, double dt, int steps) {
        std::cout << "\n---" << name << " ---\n";
        
        std::vector<Body> test_bodies = make_pair_bodies(m0, m1, q0, u0);
        std::vector<Body> initial_bodies = test_bodies;

        HernandezPairState initial_pair = HernandezPairState::from_bodies(test_bodies, 0, 1);

        const double energy_initial = initial_pair.two_body_energy(G);
        const Vec3 angular_momentum_initial = initial_pair.two_body_angular_momentum();
        const Vec3 total_momentum_initial = initial_pair.total_momentum();

        double max_energy_error = 0.0;
        double max_angular_momentum_error = 0.0;
        double max_total_momentum_error = 0.0;
        
        bool all_converged = true;

        int max_iterations = 0;

        for (int step = 0; step < steps; ++step) {
            HernandezPairMapResult result = apply_hernandez_pair_kepler_map(test_bodies, 0, 1, dt, G);
            if (!result.converged) {
                all_converged = false;
            }
            max_iterations = std::max(max_iterations, result.iterations);

            HernandezPairState current_pair = HernandezPairState::from_bodies(test_bodies, 0, 1);

            const double energy_current = current_pair.two_body_energy(G);
            const Vec3 angular_momentum_current = current_pair.two_body_angular_momentum();
            const Vec3 total_momentum_current = current_pair.total_momentum();

            max_energy_error = std::max(max_energy_error, std::abs(energy_current - energy_initial));
            max_angular_momentum_error = std::max(max_angular_momentum_error, (angular_momentum_current - angular_momentum_initial).norm());
            max_total_momentum_error = std::max(max_total_momentum_error, (total_momentum_current - total_momentum_initial).norm());
        }

        HernandezPairState final_pair = HernandezPairState::from_bodies(test_bodies, 0, 1);

        const double one_period_relative_position_error = (final_pair.relative_position - initial_pair.relative_position).norm();
        const double one_period_relative_velocity_error = (final_pair.relative_velocity - initial_pair.relative_velocity).norm();

        for (int step = 0; step < steps; ++step) {
            HernandezPairMapResult result = apply_hernandez_pair_kepler_map(test_bodies, 0, 1, -dt, G);
            if (!result.converged) {
                all_converged = false;
            }
            max_iterations = std::max(max_iterations, result.iterations);
        }

        const double forward_backward_position_error_body_0 = (test_bodies[0].position - initial_bodies[0].position).norm();
        const double forward_backward_position_error_body_1 = (test_bodies[1].position - initial_bodies[1].position).norm();
        const double forward_backward_velocity_error_body_0 = (test_bodies[0].velocity - initial_bodies[0].velocity).norm();
        const double forward_backward_velocity_error_body_1 = (test_bodies[1].velocity - initial_bodies[1].velocity).norm();

        std::cout << "All Kepler solves converged: " << (all_converged ? "true" : "false") << "\n";
        std::cout << "Max Newton iterations: " << max_iterations << "\n";
        std::cout << "Max pair energy error: " << max_energy_error << "\n";
        std::cout << "Max pair angular momentum error: " << max_angular_momentum_error << "\n";
        std::cout << "Max pair total momentum error: " << max_total_momentum_error << "\n";
        std::cout << "One-period relative position error: " << one_period_relative_position_error << "\n";
        std::cout << "One-period relative velocity error: " << one_period_relative_velocity_error << "\n";
        std::cout << "Forward-backward position error body 0: " << forward_backward_position_error_body_0 << "\n";
        std::cout << "Forward-backward position error body 1: " << forward_backward_position_error_body_1 << "\n";
        std::cout << "Forward-backward velocity error body 0: " << forward_backward_velocity_error_body_0 << "\n";
        std::cout << "Forward-backward velocity error body 1: " << forward_backward_velocity_error_body_1 << "\n";
    };

    const double m0 = 1.0;
    const double m1 = 1e-6;
    const double total_mass = m0 + m1;

    // Circular Binary Test
    {
        const double a = 1.0;
        const double r = a;
        const double circular_speed = std::sqrt(G * total_mass / r);
        const double period = 2.0 * pi * std::sqrt((a * a * a) / (G * total_mass));
        const int steps = 512;
        const double dt = period / static_cast<double>(steps);

        run_case("Circular binary, one orbit", m0, m1, Vec3(r, 0.0, 0.0), Vec3(0.0, circular_speed, 0.0), dt, steps);
    }
    // Eccentric Binary Test
    {
        const double a = 1.0;
        const double e = 0.6;
        const double r_pericenter = a * (1.0 - e);
        const double pericenter_speed = std::sqrt(G * total_mass * (1.0 + e) / (a * (1.0 - e)));
        const double period = 2.0 * pi * std::sqrt((a * a * a) / (G * total_mass));
        const int steps = 1024;
        const double dt = period / static_cast<double>(steps);

        run_case("Eccentric binary, e = 0.6, one orbit", m0, m1, Vec3(r_pericenter, 0.0, 0.0), Vec3(0.0, pericenter_speed, 0.0), dt, steps);
    }
}

void Tests::TestHernandezPairDiagnostics() {
    std::cout << "\n=== Hernandez Pair Diagnostics Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    std::vector<Body> test_bodies;
    test_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    test_bodies.emplace_back(1.0e-6, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.01720209895, 0.0));

    for (Body& body : test_bodies) {
        body.updateMomentumFromVelocity();
    }

    const PairDiagnostics initial = compute_pair_diagnostics(test_bodies, 0, 1, G);

    print_pair_diagnostics(std::cout, initial, "Initial pair diagnostics:");

    const double dt = 0.25;
    const HernandezPairMapResult forward_result = apply_hernandez_pair_kepler_map(test_bodies, 0, 1, dt, G);
    const PairDiagnostics after_forward = compute_pair_diagnostics(test_bodies, 0, 1, G);
    const PairDiagnosticDeviation forward_deviation = compare_pair_diagnostics(after_forward, initial);

    std::cout << "Forward converged: " << std::boolalpha << forward_result.converged << ", iterations: " << forward_result.iterations << "\n";

    print_pair_diagnostics_deviation(std::cout, forward_deviation, "Forward pair diagnostic deviation:");

    const HernandezPairMapResult backward_result = apply_hernandez_pair_kepler_map(test_bodies, 0, 1, -dt, G);
    const PairDiagnostics after_backward = compute_pair_diagnostics(test_bodies, 0, 1, G);
    const PairDiagnosticDeviation backward_deviation = compare_pair_diagnostics(after_backward, initial);

    std::cout << "Backward converged: " << std::boolalpha << backward_result.converged << ", iterations: " << backward_result.iterations << "\n";

    print_pair_diagnostics_deviation(std::cout, backward_deviation, "Forward-backward pair diagnostic deviation:");
}

void Tests::TestHernandezSymmetricOrdering() {
    std::cout << "\n=== Hernandez Symmetric Ordering Test ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    std::vector<Body> test_bodies;

    test_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    test_bodies.emplace_back(1.0e-3, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0172020985, 0.0));
    test_bodies.emplace_back(5.0e-4, Vec3(2.3, 0.0, 0.0), Vec3(0.0, 0.0113, 0.0));

    std::vector<Body> initial = test_bodies;
    std::vector<Pair> test_pairs = {{0, 1}, {0, 2}, {1, 2}};

    Hernandez hernandez(test_pairs);

    const double dt = 0.05;
    const int steps = 100;

    for(int step = 0; step < steps; ++step) {
        hernandez.step(test_bodies, dt, G);
    }
    for (int step = 0; step < steps; ++step) {
        hernandez.step(test_bodies, -dt, G);
    }

    double max_position_error = 0.0;
    double max_velocity_error = 0.0;

    for (size_t i = 0; i < test_bodies.size(); ++i) {
        const double position_error = (test_bodies[i].position - initial[i].position).norm();
        const double velocity_error = (test_bodies[i].velocity - initial[i].velocity).norm();

        max_position_error = std::max(max_position_error, position_error);
        max_velocity_error = std::max(max_velocity_error, velocity_error);
    }

    std::cout << "Forward steps: " << steps << "\n";
    std::cout << "Backward steps: " << steps << "\n";
    std::cout << "dt: " << dt << "\n";
    std::cout << "Max position error: " << max_position_error << "\n";
    std::cout << "Max velocity error: " << max_velocity_error << "\n";
}

void Tests::TestHernandezFixedStepValidation() {
    std::cout << "\n=== Hernandez Fixed-Step Validation Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto run_case = [&](const std::string& name, std::vector<Body> test_bodies, double dt, int steps) {
        std::cout << "\n--- " << name << " ---\n";

        recenter_system(test_bodies);
        update_all_momenta(test_bodies);

        const std::vector<Pair> pairs = make_all_physical_pairs(test_bodies);
        Hernandez hernandez(pairs);

        const Diagnostics initial = compute_diagnostics(test_bodies, G, dt);

        double max_relative_energy_error = 0.0;
        double max_relative_angular_momentum_error = 0.0;
        double max_linear_momentum = initial.linear_momentum;
        double max_com_drift = initial.com_drift;

        for (int step = 0; step < steps; ++step) {
            hernandez.step(test_bodies, dt, G);

            const Diagnostics current = compute_diagnostics(test_bodies, G, dt);

            const double energy_scale = std::max(1.0e-30, std::abs(initial.total_energy));
            const double angular_momentum_scale = std::max(1.0e-30, std::abs(initial.angular_momentum));
            const double relative_energy_error = std::abs(current.total_energy - initial.total_energy) / energy_scale;
            const double relative_angular_momentum_error = std::abs(current.angular_momentum - initial.angular_momentum) / angular_momentum_scale;

            max_relative_energy_error = std::max(max_relative_energy_error, relative_energy_error);
            max_relative_angular_momentum_error = std::max(max_relative_angular_momentum_error, relative_angular_momentum_error);
            max_linear_momentum = std::max(max_linear_momentum, current.linear_momentum);
            max_com_drift = std::max(max_com_drift, current.com_drift);
        }
        const Diagnostics final = compute_diagnostics(test_bodies, G, dt);

        std::cout << "Bodies: " << test_bodies.size() << "\n";
        std::cout << "Pairs: " << pairs.size() << "\n";
        std::cout << "dt: " << dt << "\n";
        std::cout << "Steps: " << steps << "\n";
        std::cout << "Runtime: " << dt * static_cast<double>(steps) << "\n";
        std::cout << "Initial energy: " << initial.total_energy << "\n";
        std::cout << "Final energy: " << final.total_energy << "\n";
        std::cout << "Max relative energy error: " << max_relative_energy_error << "\n";
        std::cout << "Initial angular momentum: " << initial.angular_momentum << "\n";
        std::cout << "Final angular momentum: " << final.angular_momentum << "\n";
        std::cout << "Max relative angular momentum error: " << max_relative_angular_momentum_error << "\n";
        std::cout << "Max linear momentum: " << max_linear_momentum << "\n";
        std::cout << "Max COM drift: " << max_com_drift << "\n";
    };

    {
        const double total_mass = 1.0 + 1.0e-6;
        const double separation = 1.0;
        std::vector<Body> two_body = make_hernandez_two_body_circular_test(G);

        const double period = 2.0 * std::acos(-1.0) * std::sqrt((separation * separation * separation) / (G * total_mass));
        const int steps = 512;
        const double dt = period / static_cast<double>(steps);

        run_case("Two-body circular exactness", two_body, dt, steps);
    }

    {
        std::vector<Body> three_body = make_hernandez_three_body_planet_test();

        const double dt = 0.25;
        const int steps = static_cast<int>(365.0 / dt);

        run_case("Three-body fixed-step stability", three_body, dt, steps);
    }
}

void Tests::TestHernandezReversibility() {
    std::cout << "\n=== Hernandez Direct Reversibility Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto run_case = [&](const std::string& name, std::vector<Body> test_bodies, double dt, int steps) {
        std::cout << "\n--- " << name << " ---\n";

        recenter_system(test_bodies);
        update_all_momenta(test_bodies);

        const std::vector<Body> initial = test_bodies;
        const std::vector<Pair> pairs = make_all_physical_pairs(test_bodies);
        Hernandez hernandez(pairs);

        for (int step = 0; step < steps; ++step) {
            hernandez.step(test_bodies, dt, G);
        }
        for (int step = 0; step < steps; ++step) {
            hernandez.step(test_bodies, -dt, G);
        }

        double max_position_error = 0.0;
        double max_velocity_error = 0.0;
        double max_momentum_error = 0.0;

        for (size_t i = 0; i < test_bodies.size(); ++i) {
            const double position_error= (test_bodies[i].position - initial[i].position).norm();
            const double velocity_error = (test_bodies[i].velocity - initial[i].velocity).norm();
            const double momentum_error = (test_bodies[i].momentum - initial[i].momentum).norm();

            max_position_error = std::max(max_position_error, position_error);
            max_velocity_error = std::max(max_velocity_error, velocity_error);
            max_momentum_error = std::max(max_momentum_error, momentum_error);
        }

        const Diagnostics final = compute_diagnostics(test_bodies, G, dt);

        std::cout << "Bodies: " << test_bodies.size() << "\n";
        std::cout << "Pairs: " << pairs.size() << "\n";
        std::cout << "dt: " << dt << "\n";
        std::cout << "Forward Steps: " << steps << "\n";
        std::cout << "Backward Steps: " << steps << "\n";
        std::cout << "Max position error: " << max_position_error << "\n";
        std::cout << "Max velocity error: " << max_velocity_error << "\n";
        std::cout << "Max momentum error: " << max_momentum_error << "\n";
        std::cout << "Final linear momentum: " << final.linear_momentum << "\n";
        std::cout << "Final COM drift: " << final.com_drift << "\n";
    };

    {
        std::vector<Body> two_body = make_hernandez_two_body_circular_test(G);
        run_case("Two-body Hernandez reversibility", two_body, 0.25, 1000);
    }

    {
        std::vector<Body> three_body = make_hernandez_three_body_planet_test();
        run_case("Three-body fixed-step stability", three_body, 0.25, 1460);
    }
}

void Tests::TestHernandezStateRoundTrip() {
    std::cout << "\n=== Hernandez Cartesian State Round-Trip Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    std::vector<Body> test_bodies = make_hernandez_three_body_planet_test();
    update_all_momenta(test_bodies);

    const std::vector<Body> original_bodies = test_bodies;
    const HernandezState state = HernandezState::from_bodies(test_bodies);
    const std::vector<Body> reconstructed_bodies = state.to_bodies();

    double max_mass_error = 0.0;
    double max_position_error = 0.0;
    double max_velocity_error = 0.0;
    double max_momentum_error = 0.0;

    for (std::size_t i = 0; i <original_bodies.size(); ++i) {
        const double mass_error = std::abs(reconstructed_bodies[i].mass - original_bodies[i].mass);
        const double position_error = (reconstructed_bodies[i].position - original_bodies[i].position).norm();
        const double velocity_error = (reconstructed_bodies[i].velocity - original_bodies[i].velocity).norm();
        const double momentum_error = (reconstructed_bodies[i].momentum - original_bodies[i].momentum).norm();

        max_mass_error = std::max(max_mass_error, mass_error);
        max_position_error = std::max(max_position_error, position_error);
        max_velocity_error = std::max(max_velocity_error, velocity_error);
        max_momentum_error = std::max(max_momentum_error, momentum_error);
    }

    std::vector<Body> overwritten_bodies = test_bodies;
    state.write_to_bodies(overwritten_bodies);

    double max_write_position_error = 0.0;
    double max_write_velocity_error = 0.0;
    double max_write_momentum_error = 0.0;

    for (std::size_t i = 0; i < original_bodies.size(); ++i) {
        const double position_error = (overwritten_bodies[i].position - original_bodies[i].position).norm();
        const double velocity_error = (overwritten_bodies[i].velocity - original_bodies[i].velocity).norm();
        const double momentum_error = (overwritten_bodies[i].momentum - original_bodies[i].momentum).norm();

        max_write_position_error = std::max(max_write_position_error, position_error);
        max_write_velocity_error = std::max(max_write_velocity_error, velocity_error);
        max_write_momentum_error = std::max(max_write_momentum_error, momentum_error);
    }

    std::cout << "Bodies: " << state.size() << "\n";
    std::cout << "Total mass: " << state.total_mass() << "\n";
    std::cout << "Total momentum norm: " << state.total_momentum().norm() << "\n";
    std::cout << "COM position norm: " << state.com_positions().norm() << "\n";
    std::cout << "COM velocity norm: " << state.com_velocity().norm() << "\n";
    std::cout << "to_bodies max mass error: " << max_mass_error << "\n";
    std::cout << "to_bodies max position error: " << max_position_error << "\n";
    std::cout << "to_bodies max velocity error: " << max_velocity_error << "\n";
    std::cout << "to_bodies max momentum error: " << max_momentum_error << "\n";
    std::cout << "write_to_bodies max position error: " << max_write_position_error << "\n";
    std::cout << "write_to_bodies max velocity error: " << max_write_velocity_error << "\n";
    std::cout << "write_to_bodies max momentum error: " << max_write_momentum_error << "\n";
}

void Tests::TestHernandezRemainderOperator() {
    std::cout << "\n=== Hernandez Remainder Operator Test ===\n";
    std::cout << std::scientific << std::setprecision(17);
    
    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto max_state_difference = [](const std::vector<Body>& a, const std::vector<Body>& b) {
        double max_position_error = 0.0;
        double max_velocity_error = 0.0;
        double max_momentum_error = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            max_position_error = std::max(max_position_error, (a[i].position - b[i].position).norm());
            max_velocity_error = std::max(max_velocity_error, (a[i].velocity - b[i].velocity).norm());
            max_momentum_error = std::max(max_momentum_error, (a[i].momentum - b[i].momentum).norm());
        }

        std::cout << "Max Hernandez vs direct pair-map position error: " << max_position_error << "\n";
        std::cout << "Max Hernandez vs direct pair-map velocity error: " << max_velocity_error << "\n";
        std::cout << "Max Hernandez vs direct pair-map momentum error: " << max_momentum_error << "\n";
    };

    {
        std::cout << "\n-- N = 2 remainder-zero check --\n";
        const double dt = 0.25;
        std::vector<Body> hernandez_bodies = make_hernandez_two_body_circular_test(G);
        update_all_momenta(hernandez_bodies);
        
        std::vector<Body> direct_pair_bodies = hernandez_bodies;

        Hernandez hernandez(make_all_physical_pairs(hernandez_bodies));
        hernandez.step(hernandez_bodies, dt, G);

        const HernandezPairMapResult direct_result = apply_hernandez_pair_kepler_map(direct_pair_bodies, 0, 1, dt, G);

        std::cout << "Direct pair map converged: " << std::boolalpha << direct_result.converged << ", iterations: " << direct_result.iterations << "\n";
        max_state_difference(hernandez_bodies, direct_pair_bodies);
    }

    {
        std::cout << "\n--- N = 3 bounded remainder-composition check ---\n";

        std::vector<Body> three_body = make_hernandez_three_body_planet_test();

        recenter_system(three_body);
        update_all_momenta(three_body);

        const double dt = 0.25;
        const int steps = 1460;

        Hernandez hernandez(make_all_physical_pairs(three_body));

        const Diagnostics initial = compute_diagnostics(three_body, G, dt);
        
        double max_relative_energy_error = 0.0;
        double max_relative_angular_momentum_error = 0.0;

        for (int step = 0; step < steps; ++step) {
            hernandez.step(three_body, dt, G);

            const Diagnostics current = compute_diagnostics(three_body, G, dt);
            const double energy_scale = std::max(1.0e-30, std::abs(initial.total_energy));
            const double angular_momentum_scale = std::max(1.0e-30, std::abs(initial.angular_momentum));
            const double relative_energy_error = std::abs(current.total_energy - initial.total_energy) / energy_scale;
            const double relative_angular_momentum_error = std::abs(current.angular_momentum - initial.angular_momentum) / angular_momentum_scale;

            max_relative_energy_error = std::max(max_relative_energy_error, relative_energy_error);
            max_relative_angular_momentum_error = std::max(max_relative_angular_momentum_error, relative_angular_momentum_error);
        }

        const Diagnostics final = compute_diagnostics(three_body, G, dt);

        std::cout << "Steps: " << steps << "\n";
        std::cout << "Runtime: " << dt * static_cast<double>(steps) << "\n";
        std::cout << "Initial energy: " << initial.total_energy << "\n";
        std::cout << "Final energy: " << final.total_energy << "\n";
        std::cout << "Max relative energy error: " << max_relative_energy_error << "\n";
        std::cout << "Max relative angular momentum error: " << max_relative_angular_momentum_error << "\n";
        std::cout << "Final linear momentum: " << final.linear_momentum << "\n";
        std::cout << "Final COM drift: " << final.com_drift << "\n";
    }
}

void Tests::TestHernandezPairOrdering() {
    std::cout << "\n=== Hernandez Pair Ordering Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    std::vector<Body> test_bodies;
    test_bodies.emplace_back(1.0, Vec3(10.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    test_bodies.emplace_back(0.5, Vec3(-0.5, 0.0, 0.0), Vec3(0.0, -0.008602049475, 0.0));
    test_bodies.emplace_back(0.5, Vec3(0.5, 0.0, 0.0),  Vec3(0.0, 0.008602049475, 0.0));

    update_all_momenta(test_bodies);

    const std::vector<Pair> raw_pairs = make_all_physical_pairs(test_bodies);
    const std::vector<Pair> canonical_pairs = canonicalize_pairs(raw_pairs);
    const std::vector<Pair> strength_pairs = order_pairs_by_strength(raw_pairs, test_bodies);

    auto print_order = [&](const char* label, const std::vector<Pair>& pairs) {
        std::cout << label << "\n";

        for (const Pair& pair : pairs) {
            std::cout << "Pair (" << pair.i << ", " << pair.j << ")" << ", strength = " << pair_strength(test_bodies, pair) << "\n";
        }
    };

    print_order("Canonical pair order:", canonical_pairs);
    print_order("Strength pair order:", strength_pairs);

    if (canonical_pairs.size() != strength_pairs.size()) {
        throw std::runtime_error("HernandezPairOrdering failed: no strength-ordered pairs were produced.");
    }
    if (strength_pairs.empty()) {
        throw std::runtime_error("HernandezPairOrdering failed: strongest pair should be (1, 2).");
    }
    if (canonical_pairs.front().i == strength_pairs.front().i && canonical_pairs.front().j == strength_pairs.front().j) {
        throw std::runtime_error("HernandezPairOrdering failed: canonical and strength order did not differ.");
    }

    std::cout << "Hernandez pair ordering validation passed.\n";
}

void Tests::TestHernandezHierarchyDiagnostics() {
    std::cout << "\n=== Hernandez Hierarchy Diagnostics Test ==\n";
    std::cout << std::scientific << std::setprecision(17);

    std::vector<Body> test_bodies;
    test_bodies.emplace_back(1.0, Vec3(10.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    test_bodies.emplace_back(0.5, Vec3(-0.5, 0.0, 0.0), Vec3(0.0, -0.008602049475, 0.0));
    test_bodies.emplace_back(0.5, Vec3(0.5, 0.0, 0.0),  Vec3(0.0, 0.008602049475, 0.0));

    update_all_momenta(test_bodies);

    const std::vector<Pair> pairs = make_all_physical_pairs(test_bodies);
    const std::vector<Pair> strength_ordered_pairs = order_pairs_by_strength(pairs, test_bodies);
    const double ratio = strongest_pair_strength_ratio(pairs, test_bodies);

    std::cout << "Pair strength ranking:\n";

    for (const Pair& pair : strength_ordered_pairs) {
        std::cout << "Pair (" << pair.i << ", " << pair.j << ")" << ", strength = " << pair_strength(test_bodies, pair) << "\n";
    }

    std::cout <<"Strongest / second strongest ratio: " << ratio << "\n";

    if (strength_ordered_pairs.empty()) {
        throw std::runtime_error("TestHernandezHierarchyDiagnostics failed: no pairs were produced.");
    }
    if (strength_ordered_pairs.front().i != 1 || strength_ordered_pairs.front().j != 2) {
        throw std::runtime_error("TestHernandezHierarchyDiagnostics failed: strongest pair should be (1, 2).");
    }
    if (ratio < 10.0) {
        throw std::runtime_error("TestHernandezHierarchyDiagnostics failed: hierarchy ratio is too weak.");
    }

    std::cout << "Hernandez hierarchy diagnostics validation passed.\n";
}

void Tests::TestHierarchyTreeModel() {
    std::cout << "\n=== Hierarchy Tree Model Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    std::vector<Body> test_bodies;
    test_bodies.emplace_back(1.0, Vec3(-0.05, 0.0, 0.0), Vec3(0.0, -0.01, 0.0));
    test_bodies.emplace_back(1.0, Vec3(0.05, 0.0, 0.0),  Vec3(0.0, 0.01, 0.0));
    test_bodies.emplace_back(0.5, Vec3(9.95, 0.0, 0.0),  Vec3(0.0, -0.005, 0.0));
    test_bodies.emplace_back(0.5, Vec3(10.05, 0.0, 0.0), Vec3(0.0, 0.005, 0.0));

    update_all_momenta(test_bodies);

    HierarchyTree tree(test_bodies);
    tree.validate(static_cast<int>(test_bodies.size()));

    if (!tree.root) {
        throw std::runtime_error("TestHierarchyTreeModel failed: root is null.");
    }
    if (tree.leaf_count() != 4) {
        throw std::runtime_error("TestHierarchyTreeModel failed: tree should contain 4 leaves.");
    }
    if (tree.node_count() != 7) {
        throw std::runtime_error("TestHierarchyTreeModel failed: a full 4-leaf binary tree should have 7 nodes.");
    }

    const std::vector<int> leaves = tree.leaf_body_indices();

    if (leaves != std::vector<int>{0, 1, 2, 3}) {
        throw std::runtime_error("TestHierarchyTreeModel failed: leaf indices are incorrect.");
    }
    if (!tree.root->left || !tree.root->right) {
        throw std::runtime_error("TestHierarchyTreeModel failed: root should be binary.");
    }

    const std::vector<int> left_cluster = tree.root->left->body_indices;
    const std::vector<int> right_cluster = tree.root->right->body_indices;

    std::cout << "Node count: " << tree.node_count() << "\n";
    std::cout << "Leaf count: " << tree.leaf_count() << "\n";
    std::cout << "Root mass: " << tree.root->total_mass << "\n";
    std::cout << "Root internal separation: " << tree.root->internal_separation << "\n";
    std::cout << "Root internal strength: " << tree.root->internal_strength << "\n";
    std::cout << "Root left cluster:";

    for (int index : left_cluster) {
        std::cout << " " << index;
    }
    std::cout << "\n";
    std::cout << "Root right cluster:";
    for (int index : right_cluster) {
        std::cout << " " << index;
    }
    std::cout << "\n";

    if (left_cluster != std::vector<int>{0, 1}) {
        throw std::runtime_error("TestHierarchyTreeModel failed: left root cluster should be {0, 1}.");
    }
    if (right_cluster != std::vector<int>{2, 3}) {
        throw std::runtime_error("TestHierarchyTreeModel failed: right root cluster should be {2, 3}.");
    }
    if (!tree.root->left->is_binary()) {
        throw std::runtime_error("TestHierarchyTreeModel failed: left child should be binary.");
    }
    if (!tree.root->right->is_binary()) {
        throw std::runtime_error("TestHierarchyTreeModel failed: right child should be binary.");
    }
    std::cout << "Hierarchy tree model validation passed.\n";
}

void Tests::TestHierarchySelectionCriteria() {
    std::cout << "\n=== Hierarchy Selection Criteria Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    HierarchySelectionCriteria criteria;
    criteria.min_separation_ratio = 5.0;
    criteria.min_strength_ratio = 10.0;

    std::vector<Body> hierarchical_bodies;
    hierarchical_bodies.emplace_back(1.0, Vec3(-0.05, 0.0, 0.0), Vec3(0.0, -0.01, 0.0));
    hierarchical_bodies.emplace_back(1.0, Vec3(0.05, 0.0, 0.0),  Vec3(0.0, 0.01, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3(9.95, 0.0, 0.0),  Vec3(0.0, -0.005, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3(10.05, 0.0, 0.0), Vec3(0.0, 0.005, 0.0));

    update_all_momenta(hierarchical_bodies);

    HierarchyTree hierarchical_tree(hierarchical_bodies);

    const std::vector<HierarchyBinaryCandidate> hierarchical_candidates = hierarchical_tree.leaf_binary_candidates(hierarchical_bodies, criteria);
    const std::vector<Pair> selected_pairs = hierarchical_tree.selected_leaf_pairs(hierarchical_bodies, criteria);

    std::cout << "Hierarchical candidates:\n";

    for (const HierarchyBinaryCandidate& candidate : hierarchical_candidates) {
        std::cout << "Pair (" << candidate.pair.i << ", " << candidate.pair.j << ")"
                  << ", separation ratio = " << candidate.separation_ratio
                  << ", strength ratio = " << candidate.strength_ratio
                  << ", accepted = " << candidate.accepted << "\n";
    }

    if (selected_pairs.size() != 2) {
        throw std::runtime_error("TestHierarchySelectionCriteria failed: expected two selected pairs.");
    }
    if (selected_pairs[0].i != 0 || selected_pairs[0].j != 1) {
        throw std::runtime_error("TestHierarchySelectionCriteria failed: first selected pair should be (0, 1).");
    }
    if (selected_pairs[1].i != 2 || selected_pairs[1].j != 3) {
        throw std::runtime_error("TestHierarchySelectionCriteria failed: second selected pair should be (2, 3).");
    }

    std::vector<Body> nonhierarchical_bodies;
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0),                Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(1.0, 0.0, 0.0),                Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.5, 0.8660254037844386, 0.0), Vec3(0.0, 0.0, 0.0));

    update_all_momenta(nonhierarchical_bodies);

    HierarchyTree nonhierarchical_tree(nonhierarchical_bodies);

    const std::vector<Pair> nonhierarchical_selected_pairs = nonhierarchical_tree.selected_leaf_pairs(nonhierarchical_bodies, criteria);
    
    if (!nonhierarchical_selected_pairs.empty()) {
        throw std::runtime_error("TestHierarchySelection failed: non-hierarchical triangle should select no pairs.");
    }
    std::cout << "Hierarchy selection criteria validation passed.\n";
}

void Tests::TestHernandezRecursiveOrderingPrototype() {
    std::cout << "\n=== Hernandez Recursive Ordering Prototype Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    auto require_pair = [](const Pair& pair, int expected_i, int expected_j, const std::string& message) {
        if (pair.i != expected_i || pair.j != expected_j) {
            throw std::runtime_error(message);
        }
    };
    auto require_unique_pairs = [](const std::vector<Pair>& pairs, const std::string& message) {
        for (std::size_t a = 0; a < pairs.size(); ++a) {
            for (std::size_t b = a + 1; b < pairs.size(); ++b) {
                if (pairs[a].i == pairs[b].i && pairs[a].j == pairs[b].j) {
                    throw std::runtime_error(message);
                }
            }
        }
    };
    auto print_pairs = [](const std::string& label, const std::vector<Pair>& pairs) {
        std::cout << label << "\n";
        for (const Pair& pair : pairs) {
            std::cout << "Pair (" << pair.i << ", " << pair.j << ")\n";
        }
    };

    HierarchySelectionCriteria criteria;
    criteria.min_separation_ratio = 5.0;
    criteria.min_strength_ratio = 10.0;

    std::vector<Body> hierarchical_bodies;
    hierarchical_bodies.emplace_back(1.0, Vec3(-0.05, 0.0, 0.0), Vec3(0.0, -0.01, 0.0));
    hierarchical_bodies.emplace_back(1.0, Vec3(0.05, 0.0, 0.0),  Vec3(0.0, 0.01, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3(9.95, 0.0, 0.0),  Vec3(0.0, -0.005, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3(10.05, 0.0, 0.0), Vec3(0.0, 0.005, 0.0));

    update_all_momenta(hierarchical_bodies);
    HierarchyTree hierarchical_tree(hierarchical_bodies);

    const std::vector<Pair> recursive_selected_pairs = hierarchical_tree.recursive_selected_leaf_pairs(hierarchical_bodies, criteria);
    const std::vector<Pair> recursive_full_order = hierarchical_tree.recursive_hernandez_pair_order(hierarchical_bodies, criteria);

    print_pairs("Recursive selected hierarchy pairs:", recursive_selected_pairs);
    print_pairs("Recursive full Hernandez pair order:", recursive_full_order);

    if (recursive_selected_pairs.size() != 2) {
        throw std::runtime_error("TestHernandezRecursiveOrderingPrototype failed: expected two recursive selected pairs.");
    }

    require_pair(recursive_selected_pairs[0], 0, 1, "TestHernandezRecursiveOrderingPrototype failed: first recursive selected pair should be (0, 1).");
    require_pair(recursive_selected_pairs[1], 2, 3, "TestHernandezRecursiveOrderingPrototype failed: second recursive selected pair should be (2, 3).");

    if (recursive_full_order.size() != 6) {
        throw std::runtime_error("TestHernandezRecursiveOrderingPrototype failed: full four-body order should contain six pairs.");
    }

    require_unique_pairs(recursive_full_order, "TestHernandezRecursiveOrderingPrototype failed: recursive full order contains duplicate pairs.");
    require_pair(recursive_full_order[0], 0, 1, "TestHernandezRecursiveOrderingPrototype failed: full order should start with pair (0, 1).");
    require_pair(recursive_full_order[1], 2, 3, "TestHernandezRecursiveOrderingPrototype failed: full order should place (2, 3) second.");

    std::vector<Body> nonhierarchical_bodies;
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.5, 0.8660254037844386, 0.0), Vec3(0.0, 0.0, 0.0));

    update_all_momenta(nonhierarchical_bodies);
    HierarchyTree nonhierarchical_tree(nonhierarchical_bodies);

    const std::vector<Pair> nonhierarchical_selected_pair = nonhierarchical_tree.recursive_selected_leaf_pairs(nonhierarchical_bodies, criteria);
    const std::vector<Pair> nonhierarchical_full_order = nonhierarchical_tree.recursive_hernandez_pair_order(nonhierarchical_bodies, criteria);

    if (!nonhierarchical_selected_pair.empty()) {
        throw std::runtime_error("TestHernandezRecursiveOrderingPrototype failed: non-hierarchical triangle should have no recursive selected pairs.");
    }
    if (nonhierarchical_full_order.size() != 3) {
        throw std::runtime_error("TestHernandezRecursiveOrderingPrototype failed: non-hierarchical triangle full order should contain three pairs.");
    }

    require_unique_pairs(nonhierarchical_full_order, "TestHernandezRecursiveOrderingPrototype failed: non-hierarchical full order contains duplicate pairs.");
    require_pair(nonhierarchical_full_order[0], 0, 1, "TestHernandezRecursiveOrderingPrototype failed: non-hierarchical order should start canonical.");
    require_pair(nonhierarchical_full_order[1], 0, 2, "TestHernandezRecursiveOrderingPrototype failed: non-hierarchical order should remain canonical.");
    require_pair(nonhierarchical_full_order[2], 1, 2, "TestHernandezRecursiveOrderingPrototype failed: non-hierarchical order should remain canonical.");

    std::cout << "Hernandez recursive ordering prototype validation passed.\n";
}

void Tests::TestHernandezRecursiveOrderingValidation() {
    std::cout << "\n=== Hernandez Recursive Ordering Validation Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto contains_pairs = [](const std::vector<Pair>& pairs, const Pair& target) {
        for (const Pair& pair : pairs) {
            if (pair.i == target.i && pair.j == target.j) {
                return true;
            }
        }
        return false;
    };
    auto require_unique_pairs = [](const std::vector<Pair>& pairs, const std::string& message) {
        for (std::size_t a = 0; a < pairs.size(); ++a) {
            for (std::size_t b = a + 1; b < pairs.size(); ++b) {
                if (pairs[a].i == pairs[b].i && pairs[a].j == pairs[b].j) {
                    throw std::runtime_error(message);
                }
            }
        }
    };
    auto require_same_pair_set = [&](const std::vector<Pair>& expected, const std::vector<Pair>& actual, const std::string& message) {
        if (expected.size() != actual.size()) {
            throw std::runtime_error(message);
        }
        for (const Pair& pair : expected) {
            if (!contains_pairs(actual, pair)) {
                throw std::runtime_error(message);
            }
        }
    };
    auto same_pair_order = [](const std::vector<Pair>& a, const std::vector<Pair>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (std::size_t index = 0; index < a.size(); ++index) {
            if (a[index].i != b[index].i || a[index].j != b[index].j) {
                return false;
            }
        }
        return true;
    };
    auto max_forward_backward_error = [&](std::vector<Body> test_bodies, const std::vector<Pair>& pair_order, double dt, int steps) {
        recenter_system(test_bodies);
        update_all_momenta(test_bodies);

        const std::vector<Body> initial = test_bodies;

        Hernandez hernandez(pair_order);

        for (int step = 0; step < steps; ++step) {
            hernandez.step(test_bodies, dt, G);
        }
        for (int step = 0; step < steps; ++step) {
            hernandez.step(test_bodies, -dt, G);
        }

        double max_position_error = 0.0;
        double max_velocity_error = 0.0;
        double max_momentum_error = 0.0;

        for (std::size_t i = 0; i < test_bodies.size(); ++i) {
            max_position_error = std::max(max_position_error, (test_bodies[i].position - initial[i].position).norm());
            max_velocity_error = std::max(max_velocity_error, (test_bodies[i].velocity - initial[i].velocity).norm());
            max_momentum_error = std::max(max_momentum_error, (test_bodies[i].momentum - initial[i].momentum).norm());
        }

        std::cout << "Max forward-backward position error: " << max_position_error << "\n";
        std::cout << "Max forward-backward velocity error: " << max_velocity_error << "\n";
        std::cout << "Max forward-backward momentum error: " << max_momentum_error << "\n";

        return std::max(max_position_error, std::max(max_velocity_error, max_momentum_error));
    };

    HierarchySelectionCriteria criteria;
    criteria.min_separation_ratio = 5.0;
    criteria.min_strength_ratio = 10.0;

    std::vector<Body> hierarchical_bodies;

    const double binary_a_separation = 0.1;
    const double binary_b_separation = 0.1;
    const double binary_a_relative_speed = std::sqrt((G * 2.0) / binary_a_separation);
    const double binary_b_relative_speed = std::sqrt((G * 1.0) / binary_b_separation);
    hierarchical_bodies.emplace_back(1.0, Vec3(-0.05, 0.0, 0.0), Vec3(0.0,  -0.5 * binary_a_relative_speed, 0.0));
    hierarchical_bodies.emplace_back(1.0, Vec3( 0.05, 0.0, 0.0),  Vec3(0.0,  0.5 * binary_a_relative_speed, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3( 9.95, 0.0, 0.0),  Vec3(0.0, -0.5 * binary_b_relative_speed, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3(10.05, 0.0, 0.0), Vec3(0.0,   0.5 * binary_b_relative_speed, 0.0));

    update_all_momenta(hierarchical_bodies);
    HierarchyTree hierarchical_tree(hierarchical_bodies);

    const std::vector<Pair> canonical_pairs = canonicalize_pairs(make_all_physical_pairs(hierarchical_bodies));
    const std::vector<Pair> recursive_order = hierarchical_tree.recursive_hernandez_pair_order(hierarchical_bodies, criteria);

    std::cout << "Canonical pair count: " << canonical_pairs.size() << "\n";
    std::cout << "Recursive pair count: " << recursive_order.size() << "\n";
    std::cout << "Recursive pair order:\n";
        for (const Pair& pair : recursive_order) {
            std::cout << "Pair (" << pair.i << ", " << pair.j << ")\n";
        } 
    
    require_unique_pairs(recursive_order, "TestHernandezRecursiveOrderingValidation failed: recursive order contains duplicate pairs.");
    require_same_pair_set(canonical_pairs, recursive_order, "TestHernandezRecursiveOrderingValidation failed: recursive order does not contain the same pair set as canonical order.");

    if (recursive_order.size() != 6) {
        throw std::runtime_error("TestHernandezRecursiveOrderingValidation failed: four-body recursive order should contain six pairs.");
    }
    if (recursive_order[0].i != 0 || recursive_order[0].j != 1) {
        throw std::runtime_error("TestHernandezRecursiveOrderingValidation failed: recursive order should begin with pair (0, 1).");
    }
    if (recursive_order[1].i != 2 || recursive_order[1].j != 3) {
        throw std::runtime_error("TestHernandezRecursiveOrderingValidation failed: recursive order should place pair (2, 3) second.");
    }

    const double hierarchical_error = max_forward_backward_error(hierarchical_bodies, recursive_order, 0.001, 1000);

    if (hierarchical_error > 1.0e-8) {
        throw std::runtime_error("TestHernandezRecursiveOrderingValidation failed: recursive order reversiblity error is too large.");
    }

    std::vector<Body> nonhierarchical_bodies;
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.5, 0.8660254037844386, 0.0), Vec3(0.0, 0.0, 0.0));

    update_all_momenta(nonhierarchical_bodies);
    HierarchyTree nonhierarchical_tree(nonhierarchical_bodies);

    const std::vector<Pair> nonhierarchical_canonical_pairs = canonicalize_pairs(make_all_physical_pairs(nonhierarchical_bodies));
    const std::vector<Pair> nonhierarchical_recursive_order = nonhierarchical_tree.recursive_hernandez_pair_order(nonhierarchical_bodies, criteria);

    if (!same_pair_order(nonhierarchical_recursive_order, nonhierarchical_canonical_pairs)) {
        throw std::runtime_error("TestHernandezRecursiveOrderingValidation failed: non-hierarchical recursive order should equal canonical order.");
    }

    const double nonhierarchical_error = max_forward_backward_error(nonhierarchical_bodies, nonhierarchical_recursive_order, 0.001, 1000);

    if (nonhierarchical_error > 1.0e-8) {
        throw std::runtime_error("TestHernandezRecursiveOrderingValidation failed: non-hierarchical fallback reversibility error is too large.");
    }

    std::cout << "Hernandez recursive ordering validation passed.\n";
}

void Tests::TestHernandezPairLevelScheduler() {
    std::cout << "\n=== Hernandez Pair-Level Scheduler Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto same_pair = [](const Pair& a, const Pair& b) {
        return a.i == b.i && a.j == b.j;
    };
    auto contains_pair = [&](const std::vector<Pair>& pairs, const Pair& target) {
        for (const Pair& pair : pairs) {
            if (same_pair(pair, target)) {
                return true;
            }
        }
        return false;
    };
    auto count_pair_occurrences = [&](const HernandezPairLevelSchedule& schedule, const Pair& target) {
        int count = 0;
        for (const HernandezPairLevelGroup& group : schedule.levels) {
            for (const Pair& pair : group.pairs) {
                if (same_pair(pair, target)) {
                    ++count;
                }
            }
        }
        return count;
    };
    auto find_pair_level = [&](const HernandezPairLevelSchedule& schedule, const Pair& target) {
        for (const HernandezPairLevelGroup& group : schedule.levels) {
            for (const Pair& pair : group.pairs) {
                if (same_pair(pair, target)) {
                    return group.level;
                }
            }
        }
        return -1;
    };

    HierarchySelectionCriteria criteria;
    criteria.min_separation_ratio = 5.0;
    criteria.min_strength_ratio = 10.0;

    std::vector<Body> hierarchical_bodies;

    const double binary_a_separation = 0.1;
    const double binary_b_separation = 0.1;
    const double binary_a_relative_speed = std::sqrt((G * 2.0) / binary_a_separation);
    const double binary_b_relative_speed = std::sqrt((G * 1.0) / binary_b_separation);
    hierarchical_bodies.emplace_back(1.0, Vec3(-0.05, 0.0, 0.0), Vec3(0.0,  -0.5 * binary_a_relative_speed, 0.0));
    hierarchical_bodies.emplace_back(1.0, Vec3( 0.05, 0.0, 0.0),  Vec3(0.0,  0.5 * binary_a_relative_speed, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3( 9.95, 0.0, 0.0),  Vec3(0.0, -0.5 * binary_b_relative_speed, 0.0));
    hierarchical_bodies.emplace_back(0.5, Vec3(10.05, 0.0, 0.0), Vec3(0.0,   0.5 * binary_b_relative_speed, 0.0));

    update_all_momenta(hierarchical_bodies);
    HierarchyTree hierarchical_tree(hierarchical_bodies);

    const std::vector<Pair> recursive_order = hierarchical_tree.recursive_hernandez_pair_order(hierarchical_bodies, criteria);
    const double base_dt = 0.25;
    const double eta = 0.05;
    const int max_level = 4;
    const TimestepPlan plan = build_timestep_plan(hierarchical_bodies, recursive_order, base_dt, G, max_level, eta);
    const HernandezPairLevelSchedule schedule = build_hernandez_pair_level_schedule(plan);

    print_hernandez_pair_level_schedule(schedule);

    if (plan.pair_info.size() != recursive_order.size()) {
        throw std::runtime_error(
            "TestHernandezPairLevelScheduler failed: planner did not preserve the full recursive pair count."
        );
    }
    if (schedule.levels.size() != static_cast<std::size_t>(max_level + 1)) {
        throw std::runtime_error("TestHernandezPairLevelScheduler failed: schedule has the wrong number of levels.");
    }
    for (std::size_t index = 0; index < recursive_order.size(); ++index) {
        if (!same_pair(plan.pair_info[index].pair, recursive_order[index])) {
            throw std::runtime_error("TestHernandezPairLevelScheduler failed: planner did not preserve recursive pair order.");
        }
    }

    int scheduled_pair_count = 0;

    for (const HernandezPairLevelGroup& group : schedule.levels) {
        scheduled_pair_count += static_cast<int>(group.pairs.size());
    }
    if (scheduled_pair_count != static_cast<int>(recursive_order.size())) {
        throw std::runtime_error("TestHernandezPairLevelScheduler failed: schedule lost or added pairs.");
    }
    for (const Pair& pair : recursive_order) {
        if (count_pair_occurrences(schedule, pair) != 1) {
            throw std::runtime_error("TestHernandezPairLevelScheduler failed: pair appears wrong number of times in schedule.");
        }
    }

    const Pair binary_a{0, 1};
    const Pair binary_b{2, 3};
    const int binary_a_level = find_pair_level(schedule, binary_a);
    const int binary_b_level = find_pair_level(schedule, binary_b);

    std::cout << "Pair (0, 1) level: " << binary_a_level << "\n";
    std::cout << "Pair (2, 3) level: " << binary_b_level << "\n";

    if (binary_a_level <= 0) {
        throw std::runtime_error("TestHernandezPairLevelScheduler failed: tight binary (0, 1) should be assigned to a refined level.");
    }

    if (binary_b_level <= 0) {
        throw std::runtime_error("TestHernandezPairLevelScheduler failed: tight binary (2, 3) should be assigned to a refined level.");
    }

    std::vector<Body> nonhierarchical_bodies;

    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    nonhierarchical_bodies.emplace_back(1.0, Vec3(0.5, 0.8660254037844386, 0.0), Vec3(0.0, 0.0, 0.0));

    update_all_momenta(nonhierarchical_bodies);
    HierarchyTree nonhierarchical_tree(nonhierarchical_bodies);

    const std::vector<Pair> nonhierarchical_order = nonhierarchical_tree.recursive_hernandez_pair_order(nonhierarchical_bodies, criteria);
    const TimestepPlan nonhierarchical_plan = build_timestep_plan(nonhierarchical_bodies, nonhierarchical_order, base_dt, G, max_level, eta);
    const HernandezPairLevelSchedule nonhierarchical_schedule = build_hernandez_pair_level_schedule(nonhierarchical_plan);

    for (const Pair& pair : nonhierarchical_order) {
        if (!contains_pair(nonhierarchical_schedule.levels[0].pairs, pair)) {
            throw std::runtime_error("TestHernandezPairLevelScheduler failed: non-hierarchical fallback should keep all pairs at level 0.");
        }
    }

    std::cout << "Hernandez pair-level scheduler validation passed.\n";    
}

void Tests::TestHernandezBlockTimestepSequenceDesign() {
    std::cout << "\n=== Hernandez Block-Timestep Sequence Design Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    struct TraceStep {
        char kind;   // 'C' = correction/remainder wrapper, 'K' = pair-map group
        int level;   // -1 for correction, otherwise timestep level
        double dt;
    };

    const double base_dt = 1.0;
    const int max_level = 2;
    const double tolerance = 1.0e-14;

    std::vector<TraceStep> trace;

    auto append_block_step =
        [&](auto&& self, int level, double dt) -> void {
            if (level == max_level) {
                trace.push_back({'K', level, dt});
                return;
            }

            trace.push_back({'K', level, 0.5 * dt});

            self(self, level + 1, 0.5 * dt);
            self(self, level + 1, 0.5 * dt);

            trace.push_back({'K', level, 0.5 * dt});
        };

    trace.push_back({'C', -1, 0.5 * base_dt});
    append_block_step(append_block_step, 0, base_dt);
    trace.push_back({'C', -1, 0.5 * base_dt});

    std::cout << "Designed sequence:\n";

    for (const TraceStep& step : trace) {
        if (step.kind == 'C') {
            std::cout << "C(" << step.dt << ")\n";
        }
        else {
            std::cout << "K_" << step.level << "(" << step.dt << ")\n";
        }
    }

    if (trace.size() != 12) {
        throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: expected 12 trace operations for levels 0, 1, and 2.");
    }
    for (std::size_t i = 0; i < trace.size(); ++i) {
        const std::size_t j = trace.size() - 1 - i;
        if (trace[i].kind != trace[j].kind || trace[i].level != trace[j].level || std::fabs(trace[i].dt - trace[j].dt) > tolerance) {
            throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: sequence is not palindromic.");
        }
    }

    int correction_count = 0;
    int level_counts[3] = {0, 0, 0};
    double correction_total_dt = 0.0;
    double level_total_dt[3] = {0.0, 0.0, 0.0};

    for (const TraceStep& step : trace) {
        if (step.kind == 'C') {
            ++correction_count;
            correction_total_dt += step.dt;
        }
        else if (step.level >= 0 && step.level <= max_level) {
            ++level_counts[step.level];
            level_total_dt[step.level] += step.dt;
        }
        else {
            throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: invalid trace step level.");
        }
    }
    if (correction_count != 2) {
        throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: correction wrapper should appear twice.");
    }
    if (std::fabs(correction_total_dt - base_dt) > tolerance) {
        throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: correction wrapper does not sum to one base step." );
    }
    if (level_counts[0] != 2) {
        throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: level 0 should appear twice as two half steps.");
    }
    if (level_counts[1] != 4) {
        throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: level 1 should appear four times.");
    }
    if (level_counts[2] != 4) {
        throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: level 2 should appear four times.");
    }
    for (int level = 0; level <= max_level; ++level) {
        std::cout << "Level " << level << " total dt = " << level_total_dt[level] << ", count = " << level_counts[level] << "\n";
        if (std::fabs(level_total_dt[level] - base_dt) > tolerance) {
            throw std::runtime_error("TestHernandezBlockTimestepSequenceDesign failed: one level does not sum to one base step.");
        }
    }
    std::cout << "Correction total dt = " << correction_total_dt << "\n";
    std::cout << "Hernandez block-timestep sequence design validation passed.\n";
}

void Tests::TestHernandezAdaptiveBlockValidation() {
    std::cout << "\n=== Hernandez Adaptive Block Validation Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto max_state_error = [](const std::vector<Body>& a, const std::vector<Body>& b) {
        double max_error = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i) {
            max_error = std::max(max_error, (a[i].position - b[i].position).norm());
            max_error = std::max(max_error, (a[i].velocity - b[i].velocity).norm());
            max_error = std::max(max_error, (a[i].momentum - b[i].momentum).norm());
        }
        return max_error;
    };

    auto single_body_error = [](const Body& a, const Body& b) {
        double max_error = 0.0;

        max_error = std::max(max_error, (a.position - b.position).norm());
        max_error = std::max(max_error, (a.velocity - b.velocity).norm());
        max_error = std::max(max_error, (a.momentum - b.momentum).norm());

        return max_error;
    };

    auto make_hierarchical_block_test = [&]() {
        std::vector<Body> hierarchical_bodies;

        const double binary_a_separation = 0.1;
        const double binary_b_separation = 0.1;
        const double binary_a_relative_speed = std::sqrt((G * 2.0) / binary_a_separation);
        const double binary_b_relative_speed = std::sqrt((G * 1.0) / binary_b_separation);

        hierarchical_bodies.emplace_back(1.0, Vec3(-0.05, 0.0, 0.0), Vec3(0.0, -0.5 * binary_a_relative_speed, 0.0));
        hierarchical_bodies.emplace_back(1.0, Vec3(0.05, 0.0, 0.0), Vec3(0.0, 0.5 * binary_a_relative_speed, 0.0));
        hierarchical_bodies.emplace_back(0.5, Vec3(9.95, 0.0, 0.0), Vec3(0.0, -0.5 * binary_b_relative_speed, 0.0));
        hierarchical_bodies.emplace_back(0.5, Vec3(10.05, 0.0, 0.0), Vec3(0.0, 0.5 * binary_b_relative_speed, 0.0));

        recenter_system(hierarchical_bodies);
        update_all_momenta(hierarchical_bodies);

        return hierarchical_bodies;
    };

    HierarchySelectionCriteria criteria;
    criteria.min_separation_ratio = 5.0;
    criteria.min_strength_ratio = 10.0;

    std::vector<Body> initial_bodies = make_hierarchical_block_test();
    HierarchyTree tree(initial_bodies);

    const std::vector<Pair> recursive_order = tree.recursive_hernandez_pair_order(initial_bodies, criteria);
    const double base_dt = 0.25;
    const int max_level = 4;
    const double eta = 0.05;
    const TimestepPlan plan = build_timestep_plan(initial_bodies, recursive_order, base_dt, G, max_level, eta);
    const HernandezPairLevelSchedule schedule = build_hernandez_pair_level_schedule(plan);

    Hernandez hernandez(recursive_order);

    {
        std::vector<Body> filtered_bodies = initial_bodies;
        const std::vector<Body> before = filtered_bodies;
        const std::vector<Pair> active_pairs = {{0, 1}};

        hernandez.apply_pair_group(filtered_bodies, active_pairs, 0.01, G);

        const double untouched_body_2_error = single_body_error(filtered_bodies[2], before[2]);

        const double untouched_body_3_error = single_body_error(filtered_bodies[3], before[3]);

        std::cout << "Filtered group untouched body 2 error: " << untouched_body_2_error << "\n";
        std::cout << "Filtered group untouched body 3 error: " << untouched_body_3_error << "\n";

        if (untouched_body_2_error > 1.0e-14 || untouched_body_3_error > 1.0e-14) {
            throw std::runtime_error( "TestHernandezAdaptiveBlockValidation failed: filtered pair group changed inactive bodies.");
        }

        hernandez.apply_pair_group(filtered_bodies, active_pairs, -0.01, G);

        const double filtered_reversibility_error = max_state_error(filtered_bodies, before);

        std::cout << "Filtered pair-group forward-backward error: " << filtered_reversibility_error << "\n";

        if (filtered_reversibility_error > 1.0e-10) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: filtered pair group is not reversible enough.");
        }
    }

    {
        HernandezPairLevelSchedule level_zero_schedule;
        level_zero_schedule.base_dt = base_dt;
        level_zero_schedule.max_level = 0;
        level_zero_schedule.levels.push_back({0, base_dt, recursive_order});

        std::vector<Body> fixed_bodies = initial_bodies;
        std::vector<Body> block_bodies = initial_bodies;

        hernandez.step(fixed_bodies, base_dt, G);
        hernandez.step_block(block_bodies, level_zero_schedule, base_dt, G);

        const double fixed_vs_block_error = max_state_error(fixed_bodies, block_bodies);

        std::cout << "Fixed Hernandez vs level-0 block Hernandez error: " << fixed_vs_block_error << "\n";

        if (fixed_vs_block_error > 1.0e-13) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: level-0 block mode does not recover fixed Hernandez.");
        }
    }

    {
        std::vector<Body> block_bodies = initial_bodies;
        const std::vector<Body> before = block_bodies;
        const int steps = 20;

        for (int step = 0; step < steps; ++step) {
            hernandez.step_block(block_bodies, schedule, base_dt, G);
        }
        for (int step = 0; step < steps; ++step) {
            hernandez.step_block(block_bodies, schedule, -base_dt, G);
        }

        const double block_reversibility_error = max_state_error(block_bodies, before);

        std::cout << "Scheduled block-mode forward-backward error: " << block_reversibility_error << "\n";

        if (block_reversibility_error > 1.0e-8) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: scheduled block mode reversibility error is too large.");
        }
    }

    {
        AdaptiveLevelState state;
        const int decrease_delay = 3;

        update_adaptive_level_state(state, 2, decrease_delay, true);
        if (state.active_level != 2) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: first adaptive refresh did not initialize active level.");
        }

        update_adaptive_level_state(state, 4, decrease_delay, false);
        if (state.active_level != 4 || state.pending_lower_level != -1) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: deeper level was not accepted immediately.");
        }

        update_adaptive_level_state(state, 1, decrease_delay, false);
        if (state.active_level != 4 || state.pending_lower_level != 1 || state.pending_lower_level_count != 1) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: first lower-level request was not delayed.");
        }

        update_adaptive_level_state(state, 1, decrease_delay, false);
        if (state.active_level != 4 || state.pending_lower_level_count != 2) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: second lower-level request was not delayed.");
        }

        update_adaptive_level_state(state, 1, decrease_delay, false);
        if (state.active_level != 3 || state.pending_lower_level != -1 || state.pending_lower_level_count != 0) {
            throw std::runtime_error( "TestHernandezAdaptiveBlockValidation failed: delayed lower-level transition did not decrease by one level.");
        }

        std::cout << "Adaptive level safety-rule validation passed.\n";
    }

    {
        std::vector<Body> block_bodies = initial_bodies;
        const Diagnostics initial = compute_diagnostics(block_bodies, G, base_dt);
        const int steps = 20;

        double max_relative_energy_error = 0.0;
        double max_linear_momentum = 0.0;
        double max_com_drift = 0.0;

        for (int step = 0; step < steps; ++step) {
            hernandez.step_block(block_bodies, schedule, base_dt, G);
            const Diagnostics current = compute_diagnostics(block_bodies, G, base_dt);

            if (!std::isfinite(current.total_energy) || !std::isfinite(current.linear_momentum) || !std::isfinite(current.com_drift)) {
                throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: diagnostics became non-finite.");
            }

            const double energy_scale = std::max(1.0e-30, std::abs(initial.total_energy));
            const double relative_energy_error = std::abs(current.total_energy - initial.total_energy) / energy_scale;

            max_relative_energy_error = std::max(max_relative_energy_error, relative_energy_error);
            max_linear_momentum = std::max(max_linear_momentum, current.linear_momentum);
            max_com_drift = std::max(max_com_drift, current.com_drift);
        }

        std::cout << "Block-mode max relative energy error: " << max_relative_energy_error << "\n";
        std::cout << "Block-mode max linear momentum: " << max_linear_momentum << "\n";
        std::cout << "Block-mode max COM drift: " << max_com_drift << "\n";

        if (max_linear_momentum > 1.0e-8) {
            throw std::runtime_error("TestHernandezAdaptiveBlockValidation failed: linear momentum drift is too large.");
        }
    }
    std::cout << "Hernandez adaptive block validation passed.\n";
}