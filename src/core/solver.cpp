#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <string>

#include "core/solver.h"
#include "core/body.h"
#include "core/variational_state.h"
#include "core/canonical_state.h"

#include "io/io.h"
#include "io/diagnostics_writer.h"
#include "io/csv_output_writer.h"

#include "analysis/diagnostics.h"

#include "integrators/hernandez.h"
#include "integrators/yoshida4.h"
#include "integrators/integrator.h"
#include "integrators/hb15_pair_state.h"
#include "integrators/hb15_pair_map.h"
#include "integrators/hb15.h"

#include "dynamics/pairing.h"
#include "dynamics/pair_graph.h"
#include "dynamics/jacobi.h"
#include "dynamics/jacobi_transform.h"
#include "dynamics/variational_operators.h"

#include "math/vec3.h"

#include "numerics/perturbation_forces.h"
#include "dynamics/timestep_planner.h"

/* ======================================================================
Solver

Central simulation driver for FewBodyNC.

Responsibilities:
    - Read simulation parameters
    - Select the coordinate mode: Jacobi or Cartesian
    - Select the canonical integrator for Jacobi mode
    - Advance the system
    - Write output.csv and diagnostics.csv

Coordinate Modes:
    - Jacobi mode evolves a CanonicalState using the selected canonical integrator, then reconstructs Cartesian bodies for output and diagnostics
    - Cartesian mode evolves the Body objects directly using Solver::cartesian_step()

Important:
    - In Cartesian mode, the integrator string is currently ignored. The Cartesian path is directly velocity-Verlet/leapfrog.

   ====================================================================== */

double tangent_norm(const VariationalState& v) {
    double sum = 0.0;
    for (size_t i = 1; i < v.delta_q.size(); ++i) {
        sum += v.delta_q[i].norm2();
        sum += v.delta_p[i].norm2();
    }
    return std::sqrt(sum);
}

Solver::Solver(std::vector<Body>& bodies_, CSVOutputWriter& writer_) : bodies(bodies_), integrator(nullptr), writer(writer_) {

    SolverParams params = readParams("data/param.txt");
    
    PairGraph graph = build_hierarchical_pair_graph(bodies);
    fixed_pairs.clear();

    const int N = static_cast<int>(bodies.size());

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            fixed_pairs.push_back({i, j});
        }
    }

    std::cout << "Graph Kepler Pairs: " << graph.kepler_pairs.size() << std::endl;
    std::cout << "Graph Perturbation Pairs: " << graph.perturbation_pairs.size() << std::endl;
    std::cout << "Physical Pairs Used For Full Perturbation Split: " << fixed_pairs.size() << std::endl;
    std::cout << "Coordinate Mode: " << params.coordinate_mode << std::endl;

    if (params.coordinate_mode == "cartesian") {
        return;
    }
    if (params.coordinate_mode != "jacobi") {
        throw std::runtime_error("Invalid coordinate_mode. Use 'jacobi' or 'cartesian'.");
    }

    if (params.integrator == "leapfrog") {
        integrator = std::make_unique<Leapfrog>(fixed_pairs);
    }
    else if (params.integrator == "hernandez") {
        integrator = std::make_unique<Hernandez>(fixed_pairs);
    }
    else if (params.integrator == "Yoshida4") {
        integrator = std::make_unique<Yoshida4>(fixed_pairs);
    }
    else {
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

// ==============================================================
// Main Simulation Loop and Output
// ==============================================================

/*
    Dispatch the simulation to the selected coordinate mode.

    Jacobi Mode:
    - Uses a symplectic integrator on the canonical state.
    - Applies variational operators to track the growth of perturbations.
    - Reconstructs Cartesian states for output and diagnostics.
    Cartesian Mode:
    - Uses a standard leapfrog method to evolve the Cartesian states directly.
*/

void Solver::run() {
    SolverParams params = readParams("data/param.txt");

    std::cout << "Reading param file..." << std::endl;
    std::cout << "dt = " << params.timestep << std::endl;
    std::cout << "Loaded timestep: " << params.timestep << std::endl;
    std::cout << "Coordinate Mode: " << params.coordinate_mode << std::endl;
    std::cout << "Adaptive Timesteps: " << (params.adaptive_timesteps ? "true" : "false") << std::endl;

    if (params.adaptive_timesteps) {
        std::cout << "Timestep Levels: " << params.timestep_levels << std::endl;
        std::cout << "Timestep Eta: " << params.timestep_eta << std::endl;
        std::cout << "Timestep Refresh Interval: " << params.timestep_refresh_interval << std::endl;
        std::cout << "Timestep Level Decrease Delay: " << params.timestep_level_decrease_delay << std::endl;
    }
    if (params.coordinate_mode == "cartesian") {
        run_cartesian(params);
    } else if (params.coordinate_mode == "jacobi") {
        run_jacobi(params);
    } else {
        throw std::runtime_error("Invalid coordinate_mode. Use 'jacobi' or 'cartesian'.");
    }
}

/*
    Write the current Cartesian body states to the output CSV file.
    Even in Jacobi mode, we reconstruct the Cartesian states at each output step for diagnostics and output purposes.
*/

void Solver::write_current_bodies(double time) {
    std::vector<BodyState> states;
    
    states.reserve(bodies.size());
    for (const auto& body : bodies) {
        states.push_back(body.toState(time));
    }

    writer.write(states);
}

// ==============================================================
// Specific Steps for Specified Coordinate Modes
// ==============================================================

/*
    Run the canonical Jacobi-coordinate integration path.

    If adaptive_timesteps is false:
        - one canonical integrator step is taken per base timestep
    
    If adaptive_timesteps is true:
        - Step 10.6 uses safe refreshed finest-level subcycling
        - The timestep planner is refreshed only at global base-step synchronization points.
        - If a deeper level is needed, the code accepts it immediately
        - If a shallower level is requested, the decrease is delayed to avoid rapid timestep-level oscillations
        - The full Jacobi integrator is advanced using inner_dt = dt / 2^level
        - Individual pair-local subcycling is not implemented yet
    
    Output and diagnostics are still written only at the base timestep cadence
*/

void Solver::run_jacobi(const SolverParams& params) {
    int output_frequency = params.output_frequency;
    double runtime = params.runtime;
    double dt = params.timestep;
    double G = params.gravitational_constant;
    const int steps = static_cast<int>(runtime / dt);

    if (!integrator) {
        throw std::runtime_error("Jacobi mode requires a CanonicalState integrator.");
    }

    CanonicalState state = compute_jacobi_state(bodies);

    int adaptive_substeps = 1;
    double inner_dt = dt;
    
    int planner_deepest_level = 0;
    int active_adaptive_level = 0;
    int pending_lower_level = -1;
    int pending_lower_level_count = 0;

    const int timestep_refresh_interval = std::max(1, params.timestep_refresh_interval);
    const int timestep_level_decrease_delay = std::max(1, params.timestep_level_decrease_delay);

    auto apply_adaptive_level = [&]() {
        adaptive_substeps = 1;
        for (int k = 0; k < active_adaptive_level; ++k) {
            adaptive_substeps *= 2;
        }
        inner_dt = dt / static_cast<double>(adaptive_substeps);
    };

    auto refresh_adaptive_schedule = [&](int base_step_index, bool print_full_summary) {
        if (!params.adaptive_timesteps) {
            adaptive_substeps = 1;
            inner_dt = dt;
            planner_deepest_level = 0;
            active_adaptive_level = 0;
            pending_lower_level = -1;
            pending_lower_level_count = 0;
            return;
        }

        TimestepPlan plan = build_timestep_plan(bodies, fixed_pairs, dt, G, params.timestep_levels, params.timestep_eta);
        TimestepSchedule schedule = build_timestep_schedule(plan);

        planner_deepest_level = 0;

        for (const TimestepLevelSchedule& level_schedule : schedule.levels) {
            if (!level_schedule.pairs.empty()) {
                planner_deepest_level = std::max(planner_deepest_level, level_schedule.level);
            }
        }

        if (base_step_index == 0) {
            active_adaptive_level = planner_deepest_level;
            pending_lower_level = -1;
            pending_lower_level_count = 0;
        }
        else if (planner_deepest_level > active_adaptive_level) {
            // Safety rule:
            // If the planner asks for a deeper level, accept immediately
            active_adaptive_level = planner_deepest_level;
            pending_lower_level = -1;
            pending_lower_level_count = 0;
        }
        else if (planner_deepest_level < active_adaptive_level) {
            // Stability rule:
            // If the planner asks for a shallower level, delay the decrease
            if (pending_lower_level == planner_deepest_level) {
                ++pending_lower_level_count;
            }
            else {
                pending_lower_level = planner_deepest_level;
                pending_lower_level_count = 1;
            }
            if (pending_lower_level_count >= timestep_level_decrease_delay) {
                active_adaptive_level = std::max(planner_deepest_level, active_adaptive_level - 1);
            }
        }
        else {
            pending_lower_level = -1;
            pending_lower_level_count = 0;
        }

        apply_adaptive_level();

        if (print_full_summary) {
            print_timestep_plan_summary(plan);
            print_timestep_schedule_summary(schedule);

            std::cout << "Step 10.6: safe refreshed adaptive subcycling is active.\n";
            std::cout << "Base dt: " << dt << "\n";
            std::cout << "Planner deepest level: " << planner_deepest_level << "\n";
            std::cout << "Active adaptive level: " << active_adaptive_level << "\n";
            std::cout << "Substeps per base step: " << adaptive_substeps << "\n";
            std::cout << "Inner dt: " << inner_dt << "\n";
            std::cout << "Refresh interval: every " << timestep_refresh_interval << " base step(s)\n";
            std::cout << "Level decrease delay: " << timestep_level_decrease_delay << " refresh(es)\n";
            std::cout << "Note: deeper levels are accepted immediately; shallower levels are delayed.\n";
            std::cout << "Note: this still subcycles the full Jacobi integrator, not individual pairs yet.\n\n";
        } 
        else {
            std::cout << "[adaptive refresh] base step " << base_step_index 
                      << ": planner_level = " << planner_deepest_level 
                      << ", active_level = " << active_adaptive_level 
                      << ", substeps = " << adaptive_substeps
                      << ", inner_dt = " << inner_dt;
            if (pending_lower_level >= 0) {
                std::cout << ", pending_lower_level = " << pending_lower_level
                          << ", pending_count = " << pending_lower_level_count
                          << "/" << timestep_level_decrease_delay;
            }
            std::cout << "\n";
        }
    };

    refresh_adaptive_schedule(0, true);

    VariationalState var_state;
    const int N = static_cast<int>(bodies.size());

    var_state.delta_q.resize(N);
    var_state.delta_p.resize(N);

    for (int i = 1; i < N; ++i) {
        var_state.delta_q[i] = Vec3(1e-10, 0.0, 0.0);  // Small perturbation in position
        var_state.delta_p[i] = Vec3();  // Small perturbation in momentum
    }

    DiagnosticsWriter diagnostics_writer("diagnostics.csv");

    {
        Diagnostics diag = compute_diagnostics(bodies, G, dt);
        std::cout << "Step: 0, Time: 0" 
                << ", | Total Energy: " << diag.total_energy 
                << ", | Linear Momentum: " << diag.linear_momentum 
                << ", | Angular Momentum: " << diag.angular_momentum 
                << ", | Shadow Energy: " << diag.shadow_energy 
                << ", | COM Drift: " << diag.com_drift
                << std::endl;
        diagnostics_writer.write(0.0, diag);
        write_current_bodies(0.0);
    }

    for (int step = 1; step <= steps; ++step) {
        const int current_base_step = step - 1;
        if (params.adaptive_timesteps && current_base_step > 0 && current_base_step % timestep_refresh_interval == 0) {
            refresh_adaptive_schedule(current_base_step, false);
        }
        for (int substep = 0; substep < adaptive_substeps; ++substep) {
            integrator->step(state, inner_dt, G);
            variational_drift_operator(state, var_state, inner_dt);
            variational_kick_operator(state, var_state, fixed_pairs, inner_dt, G);
        }

        reconstruct_bodies(state, bodies);

        const double time = step * dt;

        if (step % output_frequency == 0 || step == steps) {
            Diagnostics diag = compute_diagnostics(bodies, G, dt);

            double tangent = std::max(tangent_norm(var_state), 1e-300);
            double lambda = std::log(tangent / 1e-10) / time;

            // std::cout << "Step: " << step << ", Time: " << time 
            //         << ", | Total Energy: " << diag.total_energy 
            //         << ", | Linear Momentum: " << diag.linear_momentum 
            //         << ", | Angular Momentum: " << diag.angular_momentum 
            //         << ", | Shadow Energy: " << diag.shadow_energy 
            //         << ", | COM Drift: " << diag.com_drift
            //         << ", | Lyapunov Exponent: " << lambda
            //         << ", | Tangent Norm: " << tangent
            //         << std::endl;
            diagnostics_writer.write(step * dt, diag);
            write_current_bodies(step * dt);
        }
    }
    diagnostics_writer.close();
}

/*
    One Cartesian integration step using a leapfrog method.
        v(t + dt/2) = v(t) + (dt/2) * a(t)
        r(t + dt) = r(t) + dt * v(t + dt/2)
        v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)
*/

void Solver::cartesian_step(double dt, double G) {
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

void Solver::run_cartesian(const SolverParams& params) {
    const int output_frequency = params.output_frequency;
    const double runtime = params.runtime;
    const double dt = params.timestep;
    const double G = params.gravitational_constant;
    const int steps = static_cast<int>(runtime / dt);

    if (params.adaptive_timesteps) {
        std::cout << "Adaptive timestep planning currently applies only to Jacobi mode.\n";
        std::cout << "Cartesian mode will continue using fixed global dt.\n";
    }

    const bool use_leapfrog = (params.integrator == "leapfrog");
    const bool use_hb15 = (params.integrator == "hb15");

    if (!use_leapfrog && !use_hb15) {
        throw std::runtime_error("Cartesian mode currently supports integrator 'leapfrog' or 'hb15'.");
    }

    std::cout << "Cartesian integrator: " << params.integrator << "\n";

    HB15 hb15_integrator(fixed_pairs);

    DiagnosticsWriter diagnostics_writer("diagnostics.csv");

    {
        Diagnostics diag = compute_diagnostics(bodies, G, dt);
        std::cout << "Step: 0, Time: 0" 
                << ", | Total Energy: " << diag.total_energy 
                << ", | Linear Momentum: " << diag.linear_momentum 
                << ", | Angular Momentum: " << diag.angular_momentum 
                << ", | Shadow Energy: " << diag.shadow_energy 
                << ", | COM Drift: " << diag.com_drift
                << std::endl;
        diagnostics_writer.write(0.0, diag);
        write_current_bodies(0.0);
    }

    for (int step = 1; step <= steps; ++step) {
        if (use_hb15) {
            hb15_integrator.step(bodies, dt, G);
        }
        else {
            cartesian_step(dt, G);
        }

        const double time = step * dt;

        if (step % output_frequency == 0 || step == steps) {
            Diagnostics diag = compute_diagnostics(bodies, G, dt);

            std::cout << "Step: " << step << ", Time: " << time 
                    << ", | Total Energy: " << diag.total_energy 
                    << ", | Linear Momentum: " << diag.linear_momentum 
                    << ", | Angular Momentum: " << diag.angular_momentum 
                    << ", | Shadow Energy: " << diag.shadow_energy 
                    << ", | COM Drift: " << diag.com_drift
                    << std::endl;
            diagnostics_writer.write(step * dt, diag);
            write_current_bodies(step * dt);
        }
    }
    diagnostics_writer.close();
}

// ==============================================================
// Tests for integrator correctness and local order of convergence
// ==============================================================

void Solver::TestHernandezAdjoint(double dt) {
    std::cout << "\n=== HERNANDEZ ADJOINT TEST ===\n";
    
    SolverParams params = readParams("data/param.txt");
    
    if (params.coordinate_mode != "jacobi") {
        std::cout << "Skipped: Hernandez adjoint test only applies to Jacobi mode.\n";
        return;
    }

    const double G = params.gravitational_constant;  // Set the global gravitational constant

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
    }
    std::cout << "Max Q Error: " << max_q_error << std::endl;
    std::cout << "Max P Error: " << max_p_error << std::endl;
}

void Solver::TestLocalOrder() {
    std::cout << "\n=== LOCAL ORDER TEST ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");

    if (params.coordinate_mode != "jacobi") {
        std::cout << "Skipped: Local-order test currently applies to Jacobi mode.\n";
        return;
    }

    const double G = params.gravitational_constant;

    CanonicalState initial = compute_jacobi_state(bodies);
    const double dt_ref = 0.0003125;
    const double T = 10.0;
    CanonicalState reference = initial;

    int ref_steps = static_cast<int>(T / dt_ref);

    for (int n = 0; n < ref_steps; ++n) {
        integrator->step(reference, dt_ref, G);
    }

    std::vector<double> dts = {0.08, 0.04, 0.02, 0.01};
    std::vector<double> errors;

    for (double dt : dts) {
        CanonicalState test = initial;
        const int steps = static_cast<int>(T / dt);

        for (int i = 0; i < steps; ++i) {
            integrator->step(test, dt, G);
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

void Solver::ReversibilityTest() {
    std::cout << "\n=== REVERSIBILITY TEST ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    // Save initial state
    std::vector<Body> initial = bodies;

    const double dt = 0.01;
    const int steps = 10000;

    if (params.coordinate_mode == "cartesian") {
        for (int i = 0; i < steps; ++i) {
            cartesian_step(dt, G);
        }
        for (auto& body : bodies) {
            body.velocity *= -1.0;
            body.updateMomentumFromVelocity();
        }
        for (int i = 0; i < steps; ++i) {
            cartesian_step(dt, G);
        }
        for (auto& body : bodies) {
            body.velocity *= -1.0;
            body.updateMomentumFromVelocity();
        }
    }

    else if (params.coordinate_mode == "jacobi") {
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
    }

    else {
        throw std::runtime_error("Invalid coordinate_mode. Use 'jacobi' or 'cartesian'.");
    }

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

void Solver::TestHB15PairStateRoundTrip() {
    std::cout << "\n=== HB15 Pair State Round-Trip Test===\n";

    if (bodies.size() < 2) {
        std::cout << "Skipped: At least TWO bodies required.";
        return;
    }

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    const int i = 0;
    const int j = 1;

    std::vector<Body> initial = bodies;

    HB15PairState before = HB15PairState::from_bodies(bodies, i, j); // Store Before Change

    const double pair_energy_before = before.two_body_energy(G);
    const Vec3 pair_angular_momentum_before = before.two_body_angular_momentum();
    const Vec3 pair_total_momentum_before = before.total_momentum();

    before.write_to_bodies(bodies);

    HB15PairState after = HB15PairState::from_bodies(bodies, i, j); // Store After Change

    const double pair_energy_after = after.two_body_energy(G);
    const Vec3 pair_angular_momentum_after = after.two_body_angular_momentum();
    const Vec3 pair_total_momentum_after = after.total_momentum();

    // Error Calculations
    const double pos_i_error = (bodies[i].position - initial[i].position).norm();
    const double pos_j_error = (bodies[j].position - initial[j].position).norm();
    const double vel_i_error = (bodies[i].velocity - initial[i].velocity).norm();
    const double vel_j_error = (bodies[j].velocity - initial[j].velocity).norm();
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

    bodies = initial;
}

void Solver::TestHB15PairKeplerMap() {
    std::cout << "\n=== HB15 Pair Kepler Map Test ===\n";

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    const double m0 = 1.0;
    const double m1 = 1.0e-6;
    const double total_mass = m0 + m1;
    const double separation = 1.0;
    const double circular_speed = std::sqrt(G * total_mass / separation);

    const Vec3 q0(separation, 0.0, 0.0);
    const Vec3 u0(0.0, circular_speed, 0.0);
    const Vec3 R0;
    const Vec3 V0;

    std::vector<Body> test_bodies;

    test_bodies.emplace_back(m0, R0 + (m1 / total_mass) * q0, V0 + (m1 / total_mass) * u0);
    test_bodies.emplace_back(m1, R0 + (m0 / total_mass) * q0, V0 + (m0 / total_mass) * u0);

    std::vector<Body> initial = test_bodies;

    HB15PairState before = HB15PairState::from_bodies(test_bodies, 0, 1);

    const double dt = 0.1;
    const double energy_before = before.two_body_energy(G);
    const Vec3 angular_momentum_before = before.two_body_angular_momentum();
    const Vec3 total_momentum_before = before.total_momentum();

    HB15PairMapResult forward = apply_hb15_pair_kepler_map(test_bodies, 0, 1, dt, G);

    HB15PairState after = HB15PairState::from_bodies(test_bodies, 0, 1);

    const double energy_after = after.two_body_energy(G);
    const Vec3 angular_momentum_after = after.two_body_angular_momentum();
    const Vec3 total_momentum_after = after.total_momentum();

    HB15PairMapResult backward = apply_hb15_pair_kepler_map(test_bodies, 0, 1, -dt, G);

    // Error Calculations
    const double pos0_reversibility_error = (test_bodies[0].position - initial[0].position).norm();
    const double pos1_reversibility_error = (test_bodies[1].position - initial[1].position).norm();
    const double vel0_reversibility_error = (test_bodies[0].velocity - initial[0].velocity).norm();
    const double vel1_reversibility_error = (test_bodies[1].velocity - initial[1].velocity).norm();
    const double energy_error = std::abs(energy_after - energy_before);
    const double angular_momentum_error = (angular_momentum_after - angular_momentum_before).norm();
    const double total_momentum_error = (total_momentum_after - total_momentum_before).norm();

    // Terminal Statements
    std::cout << "Forward converged: " << (forward.converged ? "true" : "false") << ", iterations: " << forward.iterations << "\n";
    std::cout << "Backward converged: " << (backward.converged ? "true" : "false") << ", iterations: " << backward.iterations << "\n";
    std::cout << "Forward pair energy error: " << energy_error << "\n";
    std::cout << "Forward pair angular momentum error: " << angular_momentum_error << "\n";
    std::cout << "Forward pair total momentum error: " << total_momentum_error << "\n";
    std::cout << "Forward-backward position error body 0: " << pos0_reversibility_error << "\n";
    std::cout << "Forward-backward position error body 1: " << pos1_reversibility_error << "\n";
    std::cout << "Forward-backward velocity error body 0: " << vel0_reversibility_error << "\n";
    std::cout << "Forward-backward velocity error body 1: " << vel1_reversibility_error << "\n";
}

void Solver::TestHB15PairKeplerSuite() {
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
            HB15PairMapResult result = apply_hb15_pair_kepler_map(test_bodies, 0, 1, -dt, G);
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

    // Circulat Binary Test
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

void Solver::TestHB15SymmetricOrdering() {
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

void Solver::TestHB15FixedStepValidation() {
    std::cout << "\n=== HB15 Fixed-Step Validation Test ===\n";
    std::cout << std::scientific << std::setprecision(17);

    SolverParams params = readParams("data/param.txt");
    const double G = params.gravitational_constant;

    auto make_all_pairs = [](const std::vector<Body>& test_bodies) {
        std::vector<Pair> pairs;
        const int N = static_cast<int>(test_bodies.size());

        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                pairs.push_back({i, j});
            }
        }

        return pairs;
    };

    auto update_all_momenta = [](std::vector<Body>& test_bodies) {
        for (auto& body : test_bodies) {
            body.updateMomentumFromVelocity();
        }
    };

    auto run_case = [&](const std::string& name, std::vector<Body> test_bodies, double dt, int steps) {
        std::cout << "\n--- " << name << " ---\n";

        recenter_system(test_bodies);
        update_all_momenta(test_bodies);

        const std::vector<Pair> pairs = make_all_pairs(test_bodies);
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
        const double m0 = 1.0;
        const double m1 = 1.0e-6;
        const double total_mass = m0 + m1;
        const double separation = 1.0;
        const double circular_speed = std::sqrt(G * total_mass / separation);

        std::vector<Body> two_body;

        two_body.emplace_back(m0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
        two_body.emplace_back(m1, Vec3(separation, 0.0, 0.0), Vec3(0.0, circular_speed, 0.0));

        const double period = 2.0 * std::acos(-1.0) * std::sqrt((separation * separation * separation) / (G * total_mass));
        const int steps = 512;
        const double dt = period / static_cast<double>(steps);

        run_case("Two-body circular exactness", two_body, dt, steps);
    }

    {
        std::vector<Body> three_body;

        three_body.emplace_back(1.0, Vec3(-0.00239640539191213, 0.0, 0.0), Vec3(0.0, -2.23224877762317e-05, 0.0));
        three_body.emplace_back(0.001, Vec3(0.997603594608088, 0.0, 0.0), Vec3(0.0, 0.0171913618044373, 0.0));
        three_body.emplace_back(0.0005, Vec3(2.79760359460809, 0.0, 0.0), Vec3(0.0, 0.0102622519435887, 0.0));

        const double dt = 0.25;
        const int steps = static_cast<int>(365.0 / dt);

        run_case("Three-body fixed-step stability", three_body, dt, steps);
    }
}