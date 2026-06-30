#include <memory>
#include <iostream>
#include <sstream>
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

   ====================================================================== */

double tangent_norm(const VariationalState& v) {
    double sum = 0.0;
    for (size_t i = 1; i < v.delta_q.size(); ++i) {
        sum += v.delta_q[i].norm2();
        sum += v.delta_p[i].norm2();
    }
    return std::sqrt(sum);
}

namespace {
    /*
    Pair-order policy:
        canonical = production default
        strength = optional fixed-step diagnostic mode
        auto = conservative by default; it resovles to canonical unless AUTO_MAY_SELECT_STRENGTH is deliberately changed to true later.
    */
    constexpr double AUTO_HIERARCHY_RATIO_THRESHOLD = 100.0;
    constexpr bool AUTO_MAY_SELECT_STRENGTH = false;
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

Solver::Solver(std::vector<Body>& bodies_, CSVOutputWriter& writer_) : tests(*this), bodies(bodies_), integrator(nullptr), writer(writer_) {

    SolverParams params = readParams("data/param.txt");
    
    PairGraph graph = build_hierarchical_pair_graph(bodies);
    fixed_pairs.clear();

    fixed_pairs = make_all_physical_pairs(bodies);
    fixed_pairs = canonicalize_pairs(fixed_pairs);
    effective_pair_order = "canonical";
    hierarchy_ratio = strongest_pair_strength_ratio(fixed_pairs, bodies);

    const bool use_cartesian_hernandez = params.coordinate_mode == "cartesian" && params.integrator == "hernandez";

    if (use_cartesian_hernandez) {
        if (params.pair_order == "canonical") {
            fixed_pairs = canonicalize_pairs(fixed_pairs);
            effective_pair_order = "canonical";
        }
        else if (params.pair_order == "strength") {
            fixed_pairs = order_pairs_by_strength(fixed_pairs, bodies);
            effective_pair_order = "strength";
        }
        else if (params.pair_order == "auto") {
            if (AUTO_MAY_SELECT_STRENGTH && hierarchy_ratio >= AUTO_HIERARCHY_RATIO_THRESHOLD) {
                fixed_pairs = order_pairs_by_strength(fixed_pairs, bodies);
                effective_pair_order = "strength";
            }
            else {
                fixed_pairs = canonicalize_pairs(fixed_pairs);
                effective_pair_order = "canonical";
            }
        }
        else {
            throw std::runtime_error("Invalid pair_order. Use 'canonical', 'strength', or 'auto'.");
        }
    }

    std::cout << "Graph Kepler Pairs: " << graph.kepler_pairs.size() << std::endl;
    std::cout << "Graph Perturbation Pairs: " << graph.perturbation_pairs.size() << std::endl;
    std::cout << "Physical Pairs Used For Full Perturbation Split: " << fixed_pairs.size() << std::endl;
    std::cout << "Coordinate Mode: " << params.coordinate_mode << std::endl;

    // Pairing
    if (use_cartesian_hernandez) {
        std::cout << "Hernandez Pair Order Requested: " << params.pair_order << std::endl;
        std::cout << "Hernandez Hierarchy Ratio: " << hierarchy_ratio << std::endl;
        std::cout << "Hernandez Auto Hierarchy Threshold: " << AUTO_HIERARCHY_RATIO_THRESHOLD << std::endl;
        std::cout << "Hernandez Auto Strength Selection Enabled: " << (AUTO_MAY_SELECT_STRENGTH ? "true" : "false") << std::endl;
        std::cout << "Hernandez Effective Pair Order: " << effective_pair_order << std::endl;
    }
    if (params.pair_order == "auto" && !AUTO_MAY_SELECT_STRENGTH) {
        std::cout << "Hernandez Auto Policy: strength selection is disabled; auto resovles to canonical." << std::endl;
    }

    // Coordinate Mode
    if (params.coordinate_mode == "cartesian") {
        return;
    }
    if (params.coordinate_mode != "jacobi") {
        throw std::runtime_error("Invalid coordinate_mode. Use 'jacobi' or 'cartesian'.");
    }

    // Integrators
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
    - Evolves Body objects directly using the selected Cartesian integrator.
    - Supported Cartesian integrators are leapfrog and hernandez.
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
        - Uses safe refreshed finest-level subcycling
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
    validate_finite_bodies(bodies, "jacobi_initial_state", 0, 0.0);

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
        try {
            for (int substep = 0; substep < adaptive_substeps; ++substep) {
                integrator->step(state, inner_dt, G);
                variational_drift_operator(state, var_state, inner_dt);
                variational_kick_operator(state, var_state, fixed_pairs, inner_dt, G);
            }
        }
        catch (const std::exception& exc) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Jacobi integration failure. "
                << "failed_step = " << step << ", "
                << "failed_time = " << (step - 1) * dt << ", "
                << "failed_dt = " << inner_dt << ", "
                << "integrator = " << params.integrator << ", "
                << "coordinate_mode = " << params.coordinate_mode << ", "
                << "adaptive_timesteps = " << (params.adaptive_timesteps ? "true" : "false") << ", "
                << "reason = " << exc.what();
            throw std::runtime_error(msg.str());
        }

        reconstruct_bodies(state, bodies);
        validate_finite_bodies(bodies, "jacobi_after_step", step, step * dt);

        // const double time = step * dt;

        if (step % output_frequency == 0 || step == steps) {
            Diagnostics diag = compute_diagnostics(bodies, G, dt);

            // double tangent = std::max(tangent_norm(var_state), 1e-300);
            // double lambda = std::log(tangent / 1e-10) / time;

            // std::cout << "Step: " << step << ", Time: " << time 
            //         << ", | Total Energy: " << diag.total_energy 
            //         << ", | Linear Momentum: " << diag.linear_momentum 
            //         << ", | Angular Momentum: " << diag.angular_momentum 
            //         << ", | Shadow Energy: " << diag.shadow_energy 
            //         << ", | COM Drift: " << diag.com_drift
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
    const bool use_leapfrog = (params.integrator == "leapfrog");
    const bool use_hernandez_pairwise = (params.integrator == "hernandez");

    if (!use_leapfrog && !use_hernandez_pairwise) {
        throw std::runtime_error("Cartesian mode currently supports integrator 'leapfrog' or 'hernandez'.");
    }
    if (params.adaptive_timesteps && !use_hernandez_pairwise) {
        std::cout << "Adaptive timestep planning currently applies only to Cartesian Hernandez mode.\n";
        std::cout << "Cartesian leapfrog will continue using fixed global dt.\n";
    }

    std::cout << "Cartesian integrator: " << params.integrator << "\n";
    validate_finite_bodies(bodies, "cartesian_initial_state", 0, 0.0);

    if (use_hernandez_pairwise) {
        std::cout << "Hernandez Pair Order Requested: " << params.pair_order << "\n";
        std::cout << "Hernandez Effective Pair Order: " << effective_pair_order << "\n";
    }

    Hernandez hernandez_integrator(fixed_pairs);
    AdaptiveLevelState hernandez_adaptive_state;
    HernandezPairLevelSchedule hernandez_active_schedule;
    int hernandez_planner_deepest_level = 0;

    const int timestep_refresh_interval = std::max(1, params.timestep_refresh_interval);
    const int timestep_level_decrease_delay = std::max(1, params.timestep_level_decrease_delay);

    auto refresh_hernandez_block_schedule = [&](int base_step_index, bool print_full_summary) {
        const TimestepPlan plan = build_timestep_plan(bodies, fixed_pairs, dt, G, params.timestep_levels, params.timestep_eta);
        const HernandezPairLevelSchedule raw_schedule = build_hernandez_pair_level_schedule(plan);

        hernandez_planner_deepest_level = deepest_nonempty_hernandez_level(raw_schedule);
        update_adaptive_level_state(hernandez_adaptive_state, hernandez_planner_deepest_level, timestep_level_decrease_delay, base_step_index == 0);
        hernandez_active_schedule = restrict_hernandez_pair_level_schedule(raw_schedule, hernandez_adaptive_state.active_level);

        if (print_full_summary) {
            print_timestep_plan_summary(plan);
            print_hernandez_pair_level_schedule(hernandez_active_schedule);

            std::cout << "Hernandez adaptive block mode is active.\n";
            std::cout << "Base dt: " << dt << "\n";
            std::cout << "Planner deepest level: " << hernandez_planner_deepest_level << "\n";
            std::cout << "Active adaptive level: " << hernandez_adaptive_state.active_level << "\n";
            std::cout << "Refresh interval: every " << timestep_refresh_interval << " base step(s)\n";
            std::cout << "Level decrease delay: " << timestep_level_decrease_delay << " refresh(es)\n";
            std::cout << "Note: deeper levels are accepted immediately; shallower levels are delayed.\n\n";
        }
        else {
            std::cout << "[Hernandez adaptive refresh] base step " << base_step_index 
                      << ": planner_level = " << hernandez_planner_deepest_level 
                      << ", active_level = " << hernandez_adaptive_state.active_level;

            if (hernandez_adaptive_state.pending_lower_level >= 0) {
                std::cout << ", pending_lower_level = " << hernandez_adaptive_state.pending_lower_level 
                          << ", pending_count = " << hernandez_adaptive_state.pending_lower_level_count << "/" << timestep_level_decrease_delay;
            }
            std::cout << "\n";
        }};

    if (use_hernandez_pairwise && params.adaptive_timesteps) {
        refresh_hernandez_block_schedule(0, true);
    }

    DiagnosticsWriter diagnostics_writer("diagnostics.csv");
    {
        Diagnostics diag = compute_diagnostics(bodies, G, dt);
        // std::cout << "Step: 0, Time: 0" 
        //         << ", | Total Energy: " << diag.total_energy 
        //         << ", | Linear Momentum: " << diag.linear_momentum 
        //         << ", | Angular Momentum: " << diag.angular_momentum 
        //         << ", | Shadow Energy: " << diag.shadow_energy 
        //         << ", | COM Drift: " << diag.com_drift
        //         << std::endl;
        diagnostics_writer.write(0.0, diag);
        write_current_bodies(0.0);
    }

    for (int step = 1; step <= steps; ++step) {
        try {
            if (use_hernandez_pairwise) {
                if (params.adaptive_timesteps) {
                    const int current_base_step = step - 1;
                    if (current_base_step > 0 && current_base_step % timestep_refresh_interval == 0) {
                        refresh_hernandez_block_schedule(current_base_step, false);
                    }
                    hernandez_integrator.step_block(bodies, hernandez_active_schedule, dt, G);
                }
                else {
                    hernandez_integrator.step(bodies, dt, G);
                }
            }
            else {
                cartesian_step(dt, G);
            }
        }
        catch (const std::exception& exc) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Cartesian integration failure. "
                << "failed_step = " << step << ", "
                << "failed_time = " << (step - 1) * dt << ", "
                << "failed_dt = " << dt << ", "
                << "integrator = " << params.integrator << ", "
                << "coordinate_mode = " << params.coordinate_mode << ", "
                << "pair_order = " << params.pair_order << ", "
                << "effective_pair_order = " << effective_pair_order << ", "
                << "adaptive_timesteps = " << (params.adaptive_timesteps ? "true" : "false") << ", "
                << "reason = " << exc.what();
            throw std::runtime_error(msg.str());
        }

        validate_finite_bodies(bodies, "cartesian_after_step", step, step * dt);

        if (step % output_frequency == 0 || step == steps) {
            Diagnostics diag = compute_diagnostics(bodies, G, dt);
            // std::cout << "Step: " << step << ", Time: " << time 
            //         << ", | Total Energy: " << diag.total_energy 
            //         << ", | Linear Momentum: " << diag.linear_momentum 
            //         << ", | Angular Momentum: " << diag.angular_momentum 
            //         << ", | Shadow Energy: " << diag.shadow_energy 
            //         << ", | COM Drift: " << diag.com_drift
            //         << std::endl;
            diagnostics_writer.write(step * dt, diag);
            write_current_bodies(step * dt);
        }
    }
    diagnostics_writer.close();
}
