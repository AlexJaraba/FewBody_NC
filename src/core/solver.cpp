#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <algorithm>

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
    
    If addaptive_timesteps is true:
        - Step 10.5 uses refreshed finest-level subcycling
        - The timestep planner is refreshed only at global base-step synchronization points.
        - The full Jacobi integrator is advanced using inner_dt = dt / 2^level
        - Individual pair-local subcycling is not implemented yet
    
    Output and diagnostics are still written only at the base timestep cadence
*/

void Solver::run_jacobi(const SolverParams& params) {
    int output_frequency = params.output_frequency;
    double runtime = params.runtime;
    double dt = params.timestep;
    double G = params.gravitational_constant;  // Set the global gravitational constant
    const int steps = static_cast<int>(runtime / dt);

    if (!integrator) {
        throw std::runtime_error("Jacobi mode requires a CanonicalState integrator.");
    }

    CanonicalState state = compute_jacobi_state(bodies);

    int adaptive_substeps = 1;
    double inner_dt = dt;
    int deepest_active_level = 0;
    const int timestep_refresh_interval = std::max(1, params.timestep_refresh_interval);

    auto refresh_adaptive_schedule = [&](int base_step_index, bool print_full_summary) {
        if (!params.adaptive_timesteps) {
            adaptive_substeps = 1;
            inner_dt = dt;
            deepest_active_level = 0;
            return;
        }

        TimestepPlan plan = build_timestep_plan(bodies, fixed_pairs, dt, G, params.timestep_levels, params.timestep_eta);
        TimestepSchedule schedule = build_timestep_schedule(plan);

        deepest_active_level = 0;

        for (const TimestepLevelSchedule& level_schedule : schedule.levels) {
            if (!level_schedule.pairs.empty()) {
                deepest_active_level = std::max(deepest_active_level, level_schedule.level);
            }
        }

        adaptive_substeps = 1;

        for (int k =0; k < deepest_active_level; ++k) {
            adaptive_substeps *= 2;
        }

        inner_dt = dt / static_cast<double>(adaptive_substeps);

        if (print_full_summary) {
            print_timestep_plan_summary(plan);
            print_timestep_schedule_summary(schedule);

            std::cout << "Step 10.5: schedule refresh at global synchronization points is active.\n";
            std::cout << "Base dt: " << dt << "\n";
            std::cout << "Initial deepest active level: " << deepest_active_level << "\n";
            std::cout << "Initial substeps per base step: " << adaptive_substeps << "\n";
            std::cout << "Initial inner dt: " << inner_dt << "\n";
            std::cout << "Refresh interval: every " << timestep_refresh_interval << " base step(s)\n";
            std::cout << "Note: this still subcycles the full Jacobi integrator, not individual pairs yet.\n\n";
        } 
        else {
            std::cout << "[adaptive refresh] base step " << base_step_index << ": deepest_level = " << deepest_active_level << ", substeps = " << adaptive_substeps << ", inner_dt = " << inner_dt << "\n";
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
        cartesian_step(dt, G);

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
