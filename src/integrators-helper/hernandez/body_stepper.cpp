#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>

#include "integrators-helper/hernandez/body_stepper.h"
#include "integrators-helper/hernandez/pair_map.h"
#include "dynamics/pairing.h"
#include "dynamics/timestep_planner.h"

namespace {
    struct HernandezHamiltonianBookKeeping {
        int body_count = 0;
        int pair_count = 0;

        /*
        
        In the all-pairs split, each body's kinetic energy appears once in every pair that contains that body.
        For N bodies, that is N - 1 appearances.
        The Newtonian Hamiltonian needs one copy, so the correction Hamiltonian is:

            H_correction = -(N - 2) T
        
        where T is the total Cartesian kinetic energy.
        The second-order symmetric Hernandez step applies pair maps in two half sweeps:

            correction half drift
            forward pair half maps
            reverse pair half maps
            correction half drift

        This keeps the operator sequence palindromic.

        */

       double kinetic_correction_coefficient = 0.0;
       double pair_half_dt = 0.0;
       double correction_half_dt = 0.0;
    };
    constexpr int PAIR_MAP_RETRY_DEPTH = 8;
    constexpr bool PAIR_MAP_RETRY_ENABLED = true;
    int deepest_nonempty_level(const HernandezPairLevelSchedule& schedule) {
        int deepest_level = 0;
        for (const HernandezPairLevelGroup& group : schedule.levels) {
            if (!group.pairs.empty()) {
                deepest_level = std::max(deepest_level, group.level);
            }
        }
        return deepest_level;
    }
    void apply_block_level(const HernandezBodyStepper& hernandez, std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, int level, int max_level, double dt, double G) {
        if (level > max_level || level >= static_cast<int>(schedule.levels.size())) {
            return;
        }

        const std::vector<Pair>& active_pairs = schedule.levels[static_cast<std::size_t>(level)].pairs;

        if (level == max_level) {
            hernandez.apply_pair_group(bodies, active_pairs, dt, G);
            return;
        }

        hernandez.apply_pair_group(bodies, active_pairs, 0.5 * dt, G);

        apply_block_level(hernandez, bodies, schedule, level+ 1, max_level, 0.5 * dt, G);
        apply_block_level(hernandez, bodies, schedule, level+ 1, max_level, 0.5 * dt, G);

        hernandez.apply_pair_group(bodies, active_pairs, 0.5 * dt, G);
    }
    void apply_checked_pair_map(std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        const Vec3 dr = bodies[pair.j].position - bodies[pair.i].position;
        const Vec3 dv = bodies[pair.j].velocity - bodies[pair.i].velocity;
        const double distance = dr.norm();
        const double relative_speed = dv.norm();

        int last_iterations = 0;
        std::string last_error = "unknown pair-map failure";
        bool retry_succeeded = false;
        int retry_depth_used = 0;
        int substeps_used = 1;

        auto make_failure_message = [&](const std::string& reason) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Hernandez pair Kepler map failed. "
                << "converged = false, "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_pair_i = " << pair.i << ", "
                << "failed_pair_j = " << pair.j << ", "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "failed_distance = " << distance << ", "
                << "failed_relative_speed = " << relative_speed << ", "
                << "failed iterations = " << last_iterations << ", "
                << "retry_enabled = " << (PAIR_MAP_RETRY_ENABLED ? "true" : "false") << ", "
                << "retry_succeeded = " << (retry_succeeded ? "true" : "false") << ", "
                << "retry_depth_used = " << retry_depth_used << ", "
                << "substeps_used = " << substeps_used << ", "
                << "reason = " << reason << ", "
                << "last_error = " << last_error << ", ";
            return msg.str();
        };

        try {
            const HernandezPairMapResult result = apply_hernandez_pair_kepler_map(bodies, pair.i, pair.j, dt, G);
            last_iterations = result.iterations;

            if (result.converged) {
                return;
            }

            last_error = "universal-variable solve did not converge";
        }

        catch (const std::exception& exc) {
            last_error = exc.what();
        }

        if (!PAIR_MAP_RETRY_ENABLED || PAIR_MAP_RETRY_DEPTH <= 0) {
            throw std::runtime_error(make_failure_message("retry disabled"));
        }

        const std::vector<Body> original_bodies = bodies;
        int substeps = 2;

        for (int retry_depth = 1; retry_depth <= PAIR_MAP_RETRY_DEPTH; ++retry_depth) {
            std::vector<Body> trial_bodies = original_bodies;
            const double sub_dt = dt / static_cast<double>(substeps);
            bool all_substeps_converged = true;
            int total_iterations = 0;
            for (int substep = 0; substep < substeps; ++substep) {
                try {
                    const HernandezPairMapResult result = apply_hernandez_pair_kepler_map(trial_bodies, pair.i, pair.j, sub_dt, G);
                    total_iterations += result.iterations;
                    if (!result.converged) {
                        all_substeps_converged = false;
                        last_iterations = result.iterations;
                        last_error = "universal-variable solve did not converge during retry";
                        break;
                    }
                }
                catch (const std::exception& exc) {
                    all_substeps_converged = false;
                    last_error = exc.what();
                    break;
                }
            }

            last_iterations = total_iterations;
            retry_depth_used = retry_depth;
            substeps_used = substeps;

            if (all_substeps_converged) {
                retry_succeeded = true;
                bodies = trial_bodies;
                return;
            }
            substeps *= 2;
        }
        throw std::runtime_error(make_failure_message("retry depth exhausted"));
    }
    /*
        Flow of the Hernandez kinetic remainder Hamiltonian.
        In the all-pairs Hernandez split, the sum of pair Kepler Hamiltonians counts each body's kinetic energy N - 1 times.
        The true Newtonian Hamiltonian needs one copy, so the remainder/correction term is:

            H_remainder = -(N - 2) * T
        
        where:

            T = sum_i[(p_i)^2 / (2 * m_i)]
        
        Since this remainder is purely kinetic, its exact flow is a drift:

            r_i -> r_i + remainder_dt * v_i
            v_i -> v_i
        
        The coefficient -(N - 2) is already included in remainder_dt
    */
    void apply_hernandez_remainder_flow(std::vector<Body>& bodies, double remainder_dt) {
        for (Body& body : bodies) {
            body.position += remainder_dt * body.velocity;
            body.updateMomentumFromVelocity();
        }
    }

    HernandezHamiltonianBookKeeping make_hernandez_bookkeeping(const std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt) {
        HernandezHamiltonianBookKeeping bookkeeping;

        bookkeeping.body_count = static_cast<int>(bodies.size());
        bookkeeping.pair_count = static_cast<int>(pairs.size());

        if (bookkeeping.body_count <= 1) {
            bookkeeping.kinetic_correction_coefficient = 0.0;
            bookkeeping.pair_half_dt = 0.0;
            bookkeeping.correction_half_dt = 0.0;
            return bookkeeping;
        }

        bookkeeping.kinetic_correction_coefficient = -static_cast<double>(bookkeeping.body_count - 2);
        bookkeeping.pair_half_dt = 0.5 * dt;
        bookkeeping.correction_half_dt = 0.5 * bookkeeping.kinetic_correction_coefficient * dt;

        return bookkeeping;
    }
}

HernandezBodyStepper::HernandezBodyStepper(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs_preserve_order(fixed_pairs)) {}

void HernandezBodyStepper::apply_pair_group(std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const {
    const std::vector<Pair> ordered_pairs = canonicalize_pairs_preserve_order(active_pairs);
    const double pair_half_dt = 0.5 * dt;

    for (const Pair& pair : ordered_pairs) {
        apply_checked_pair_map(bodies, pair, pair_half_dt, G);
    }
    for (auto it = ordered_pairs.rbegin(); it != ordered_pairs.rend(); ++it) {
        apply_checked_pair_map(bodies, *it, pair_half_dt, G);
    }
}

void HernandezBodyStepper::step(std::vector<Body>& bodies, double dt, double G) {
    const HernandezHamiltonianBookKeeping bookkeeping = make_hernandez_bookkeeping(bodies, pairs_, dt);
    if (bookkeeping.body_count <= 1 || bookkeeping.pair_count == 0) {
        return;
    }

    apply_hernandez_remainder_flow(bodies, bookkeeping.correction_half_dt);
    apply_pair_group(bodies, pairs_, dt, G);
    apply_hernandez_remainder_flow(bodies, bookkeeping.correction_half_dt);
}

void HernandezBodyStepper::step_block(std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, double dt, double G) const {
    const HernandezHamiltonianBookKeeping bookkeeping = make_hernandez_bookkeeping(bodies, pairs_, dt);
    const int max_level = deepest_nonempty_level(schedule);

    if (bookkeeping.body_count <= 1 || bookkeeping.pair_count == 0) {
        return;
    }
    if (schedule.levels.empty()) {
        return;
    }

    apply_hernandez_remainder_flow(bodies, bookkeeping.correction_half_dt);
    apply_block_level(*this, bodies, schedule, 0, max_level, dt, G);
    apply_hernandez_remainder_flow(bodies, bookkeeping.correction_half_dt);
}

const std::vector<Pair>& HernandezBodyStepper::pairs() const {
    return pairs_;
}