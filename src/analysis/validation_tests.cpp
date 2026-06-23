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
#include "integrators/hb15.h"
#include "integrators-helper/hb15/pair_map.h"
#include "integrators-helper/hb15/pair_state.h"
#include "integrators-helper/hb15/state.h"
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

    std::vector<Body> make_hb15_two_body_circular_test(double G) {
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

    std::vector<Body> make_hb15_three_body_planet_test() {
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

void Tests::TestHB15PairStateRoundTrip() {
    std::cout << "\n=== HB15 Pair State Round-Trip Test ===\n";

    if (solver_.bodies.size() < 2) {
        std::cout << "Skipped: At least TWO bodies required.";
        return;
    }

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    const int i = 0;
    const int j = 1;

    std::vector<Body> initial = solver_.bodies;

    HB15PairState before = HB15PairState::from_bodies(solver_.bodies, i, j); // Store Before Change

    const double pair_energy_before = before.two_body_energy(G);
    const Vec3 pair_angular_momentum_before = before.two_body_angular_momentum();
    const Vec3 pair_total_momentum_before = before.total_momentum();

    before.write_to_bodies(solver_.bodies);

    HB15PairState after = HB15PairState::from_bodies(solver_.bodies, i, j); // Store After Change

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

void Tests::TestHB15PairKeplerSuite() {
    std::cout << "\n=== HB15 Pair Kepler Map Suite ===\n";

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

        HB15PairState initial_pair = HB15PairState::from_bodies(test_bodies, 0, 1);

        const double energy_initial = initial_pair.two_body_energy(G);
        const Vec3 angular_momentum_initial = initial_pair.two_body_angular_momentum();
        const Vec3 total_momentum_initial = initial_pair.total_momentum();

        double max_energy_error = 0.0;
        double max_angular_momentum_error = 0.0;
        double max_total_momentum_error = 0.0;
        
        bool all_converged = true;

        int max_iterations = 0;

        for (int step = 0; step < steps; ++step) {
            HB15PairMapResult result = apply_hb15_pair_kepler_map(test_bodies, 0, 1, dt, G);
            if (!result.converged) {
                all_converged = false;
            }
            max_iterations = std::max(max_iterations, result.iterations);

            HB15PairState current_pair = HB15PairState::from_bodies(test_bodies, 0, 1);

            const double energy_current = current_pair.two_body_energy(G);
            const Vec3 angular_momentum_current = current_pair.two_body_angular_momentum();
            const Vec3 total_momentum_current = current_pair.total_momentum();

            max_energy_error = std::max(max_energy_error, std::abs(energy_current - energy_initial));
            max_angular_momentum_error = std::max(max_angular_momentum_error, (angular_momentum_current - angular_momentum_initial).norm());
            max_total_momentum_error = std::max(max_total_momentum_error, (total_momentum_current - total_momentum_initial).norm());
        }

        HB15PairState final_pair = HB15PairState::from_bodies(test_bodies, 0, 1);

        const double one_period_relative_position_error = (final_pair.relative_position - initial_pair.relative_position).norm();
        const double one_period_relative_velocity_error = (final_pair.relative_velocity - initial_pair.relative_velocity).norm();

        for (int step = 0; step < steps; ++step) {
            HB15PairMapResult result = apply_hb15_pair_kepler_map(test_bodies, 0, 1, -dt, G);
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

void Tests::TestHB15PairDiagnostics() {
    std::cout << "\n=== HB15 Pair Diagnostics Test ===\n";
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
    const HB15PairMapResult forward_result = apply_hb15_pair_kepler_map(test_bodies, 0, 1, dt, G);
    const PairDiagnostics after_forward = compute_pair_diagnostics(test_bodies, 0, 1, G);
    const PairDiagnosticDeviation forward_deviation = compare_pair_diagnostics(after_forward, initial);

    std::cout << "Forward converged: " << std::boolalpha << forward_result.converged << ", iterations: " << forward_result.iterations << "\n";

    print_pair_diagnostics_deviation(std::cout, forward_deviation, "Forward pair diagnostic deviation:");

    const HB15PairMapResult backward_result = apply_hb15_pair_kepler_map(test_bodies, 0, 1, -dt, G);
    const PairDiagnostics after_backward = compute_pair_diagnostics(test_bodies, 0, 1, G);
    const PairDiagnosticDeviation backward_deviation = compare_pair_diagnostics(after_backward, initial);

    std::cout << "Backward converged: " << std::boolalpha << backward_result.converged << ", iterations: " << backward_result.iterations << "\n";

    print_pair_diagnostics_deviation(std::cout, backward_deviation, "Forward-backward pair diagnostic deviation:");
}

void Tests::TestHB15SymmetricOrdering() {
    std::cout << "\n=== HB15 Symmetric Ordering Test ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    std::vector<Body> test_bodies;

    test_bodies.emplace_back(1.0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
    test_bodies.emplace_back(1.0e-3, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0172020985, 0.0));
    test_bodies.emplace_back(5.0e-4, Vec3(2.3, 0.0, 0.0), Vec3(0.0, 0.0113, 0.0));

    std::vector<Body> initial = test_bodies;
    std::vector<Pair> test_pairs = {{0, 1}, {0, 2}, {1, 2}};

    HB15 hb15(test_pairs);

    const double dt = 0.05;
    const int steps = 100;

    for(int step = 0; step < steps; ++step) {
        hb15.step(test_bodies, dt, G);
    }
    for (int step = 0; step < steps; ++step) {
        hb15.step(test_bodies, -dt, G);
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

void Tests::TestHB15FixedStepValidation() {
    std::cout << "\n=== HB15 Fixed-Step Validation Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto run_case = [&](const std::string& name, std::vector<Body> test_bodies, double dt, int steps) {
        std::cout << "\n--- " << name << " ---\n";

        recenter_system(test_bodies);
        update_all_momenta(test_bodies);

        const std::vector<Pair> pairs = make_all_physical_pairs(test_bodies);
        HB15 hb15(pairs);

        const Diagnostics initial = compute_diagnostics(test_bodies, G, dt);

        double max_relative_energy_error = 0.0;
        double max_relative_angular_momentum_error = 0.0;
        double max_linear_momentum = initial.linear_momentum;
        double max_com_drift = initial.com_drift;

        for (int step = 0; step < steps; ++step) {
            hb15.step(test_bodies, dt, G);

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
        std::vector<Body> two_body = make_hb15_two_body_circular_test(G);

        const double period = 2.0 * std::acos(-1.0) * std::sqrt((separation * separation * separation) / (G * total_mass));
        const int steps = 512;
        const double dt = period / static_cast<double>(steps);

        run_case("Two-body circular exactness", two_body, dt, steps);
    }

    {
        std::vector<Body> three_body = make_hb15_three_body_planet_test();

        const double dt = 0.25;
        const int steps = static_cast<int>(365.0 / dt);

        run_case("Three-body fixed-step stability", three_body, dt, steps);
    }
}

void Tests::TestHB15Reversibility() {
    std::cout << "\n=== HB15 Direct Reversibility Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto run_case = [&](const std::string& name, std::vector<Body> test_bodies, double dt, int steps) {
        std::cout << "\n--- " << name << " ---\n";

        recenter_system(test_bodies);
        update_all_momenta(test_bodies);

        const std::vector<Body> initial = test_bodies;
        const std::vector<Pair> pairs = make_all_physical_pairs(test_bodies);
        HB15 hb15(pairs);

        for (int step = 0; step < steps; ++step) {
            hb15.step(test_bodies, dt, G);
        }
        for (int step = 0; step < steps; ++step) {
            hb15.step(test_bodies, -dt, G);
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
        std::vector<Body> two_body = make_hb15_two_body_circular_test(G);
        run_case("Two-body HB15 reversibility", two_body, 0.25, 1000);
    }

    {
        std::vector<Body> three_body = make_hb15_three_body_planet_test();
        run_case("Three-body fixed-step stability", three_body, 0.25, 1460);
    }
}

void Tests::TestHB15StateRoundTrip() {
    std::cout << "\n=== HB15 Cartesian State Round-Trip Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    std::vector<Body> test_bodies = make_hb15_three_body_planet_test();
    update_all_momenta(test_bodies);

    const std::vector<Body> original_bodies = test_bodies;
    const HB15State state = HB15State::from_bodies(test_bodies);
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

void Tests::TestHB15RemainderOperator() {
    std::cout << "\n=== HB15 Remainder Operator Test ===\n";
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

        std::cout << "Max HB15 vs direct pair-map position error: " << max_position_error << "\n";
        std::cout << "Max HB15 vs direct pair-map velocity error: " << max_velocity_error << "\n";
        std::cout << "Max HB15 vs direct pair-map momentum error: " << max_momentum_error << "\n";
    };

    {
        std::cout << "\n-- N = 2 remainder-zero check --\n";
        const double dt = 0.25;
        std::vector<Body> hb15_bodies = make_hb15_two_body_circular_test(G);
        update_all_momenta(hb15_bodies);
        
        std::vector<Body> direct_pair_bodies = hb15_bodies;

        HB15 hb15(make_all_physical_pairs(hb15_bodies));
        hb15.step(hb15_bodies, dt, G);

        const HB15PairMapResult direct_result = apply_hb15_pair_kepler_map(direct_pair_bodies, 0, 1, dt, G);

        std::cout << "Direct pair map converged: " << std::boolalpha << direct_result.converged << ", iterations: " << direct_result.iterations << "\n";
        max_state_difference(hb15_bodies, direct_pair_bodies);
    }

    {
        std::cout << "\n--- N = 3 bounded remainder-composition check ---\n";

        std::vector<Body> three_body = make_hb15_three_body_planet_test();

        recenter_system(three_body);
        update_all_momenta(three_body);

        const double dt = 0.25;
        const int steps = 1460;

        HB15 hb15(make_all_physical_pairs(three_body));

        const Diagnostics initial = compute_diagnostics(three_body, G, dt);
        
        double max_relative_energy_error = 0.0;
        double max_relative_angular_momentum_error = 0.0;

        for (int step = 0; step < steps; ++step) {
            hb15.step(three_body, dt, G);

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