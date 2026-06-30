#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

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
    
    constexpr bool PAIR_MAP_RETRY_ENABLED = true;
    constexpr int PAIR_MAP_MAX_SUBSTEPS = 1024;
    constexpr double PAIR_MAP_TIMESCALE_FRACTION = 0.25;
    constexpr double PAIR_MAP_NEAR_ZERO_DISTANCE = 1e-14;

    double positive_radius(const Body& body) {
        return std::max(0.0, body.radius);
    }
    double collision_distance_for_pair(const std::vector<Body>& bodies, const Pair& pair) {
        return positive_radius(bodies[pair.i]) + positive_radius(bodies[pair.j]);
    }
    int next_power_of_two_substeps(int requested) {
        int substeps = 1;
        while (substeps < requested && substeps < PAIR_MAP_MAX_SUBSTEPS) {
            substeps *= 2;
        }
        return std::max(1, std::min(substeps, PAIR_MAP_MAX_SUBSTEPS));
    }
    int recommended_pair_substeps(const std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        const Body& bi = bodies[pair.i];
        const Body& bj = bodies[pair.j];
        const Vec3 dr = bj.position - bi.position;
        const Vec3 dv = bj.velocity - bi.velocity;
        const double distance = dr.norm();
        const double relative_speed = dv.norm();
        const double collision_distance = collision_distance_for_pair(bodies, pair);
        const double grav_mu = G * (bi.mass + bj.mass);

        if (!std::isfinite(distance) || distance <= PAIR_MAP_NEAR_ZERO_DISTANCE) {
            return PAIR_MAP_MAX_SUBSTEPS;
        }

        double local_timescale = std::numeric_limits<double>::infinity();

        if (relative_speed > 0.0) {
            const double clearance = std::max(distance - collision_distance, PAIR_MAP_NEAR_ZERO_DISTANCE);
            local_timescale = std::min(local_timescale, clearance / relative_speed);
        }
        if (grav_mu > 0.0) {
            local_timescale = std::min(local_timescale, std::sqrt((distance * distance * distance) / grav_mu));
        }
        if (!std::isfinite(local_timescale) || local_timescale <= 0.0) {
            return 1;
        }

        const double safe_dt = PAIR_MAP_TIMESCALE_FRACTION * local_timescale;
        const double abs_dt = std::abs(dt);

        if (abs_dt <= safe_dt) {
            return 1;
        }

        const int requested = static_cast<int>(std::ceil(abs_dt / safe_dt));

        return next_power_of_two_substeps(requested);
    }
    void validate_pair_before_map(const std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        const Body& bi = bodies[pair.i];
        const Body& bj = bodies[pair.j];
        const Vec3 dr = bj.position - bi.position;
        const Vec3 dv = bj.velocity - bi.velocity;
        const double distance = dr.norm();
        const double relative_speed = dv.norm();
        const double collision_distance = collision_distance_for_pair(bodies, pair);

        if (!std::isfinite(distance) || !std::isfinite(relative_speed)) {
            std::ostringstream msg;
            msg << "Hernandez pair map received a non-finite encounter state. "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "distance = " << distance << ", "
                << "relative_speed = " << relative_speed;
            throw std::runtime_error(msg.str());            
        }
        if (collision_distance > 0.0 && distance <= collision_distance) {
            std::ostringstream msg;
            msg << "Hernandez collision event detected. "
                << "collision_detected = true, "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_pair_i = " << pair.i << ", "
                << "failed_pair_j = " << pair.j << ", "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "distance = " << distance << ", "
                << "relative_speed = " << relative_speed << ", "
                << "collision_distance = " << collision_distance << ", "
                << "radius_i = " << positive_radius(bi) << ", "
                << "radius_j = " << positive_radius(bj);
            throw std::runtime_error(msg.str());
        }
        if (distance <= PAIR_MAP_NEAR_ZERO_DISTANCE) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Hernandez near-singular pair encounter detected. "
                << "near_singular_encounter = true, "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_pair_i = " << pair.i << ", "
                << "failed_pair_j = " << pair.j << ", "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "distance = " << distance << ", "
                << "relative_speed = " << relative_speed;
            throw std::runtime_error(msg.str());         
        }
    }
    bool try_pair_map_substeps(const std::vector<Body>& original_bodies, std::vector<Body>& accepted_bodies, const Pair& pair, double dt, double G, int substeps, int& total_iterations, std::string& last_error) {
            std::vector<Body> trial_bodies = original_bodies;
            const double sub_dt = dt / static_cast<double>(substeps);
            total_iterations = 0;

            for (int substep = 0; substep < substeps; ++substep) {
                try {
                    validate_pair_before_map(trial_bodies, pair, sub_dt, G);
                    const HernandezPairMapResult result = apply_hernandez_pair_kepler_map(trial_bodies, pair.i, pair.j, sub_dt, G);
                    total_iterations += result.iterations;
                    if (!result.converged) {
                        last_error = "universal-variable solve did not converge during pair-local substepping";
                        return false;
                    }
                }
                catch (const std::exception& exc) {
                    last_error = exc.what();
                    return false;
                }
            }

            accepted_bodies = trial_bodies;
            return true;
        }
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
        validate_pair_before_map(bodies, pair, dt, G);

        const std::vector<Body> original_bodies = bodies;
        const Vec3 dr = bodies[pair.j].position - bodies[pair.i].position;
        const Vec3 dv = bodies[pair.j].velocity - bodies[pair.i].velocity;
        const double distance = dr.norm();
        const double relative_speed = dv.norm();
        const double collision_distance = collision_distance_for_pair(bodies, pair);

        int last_iterations = 0;
        std::string last_error = "unknown pair-map failure";
        bool retry_succeeded = false;
        int substeps_used = 1;
        const int recommended_substeps = recommended_pair_substeps(bodies, pair, dt, G);

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
                << "collision_distance = " << collision_distance << ", "
                << "failed_iterations = " << last_iterations << ", "
                << "retry_enabled = " << (PAIR_MAP_RETRY_ENABLED ? "true" : "false") << ", "
                << "retry_succeeded = " << (retry_succeeded ? "true" : "false") << ", "
                << "recommended_substeps = " << recommended_substeps << ", "
                << "substeps_used = " << substeps_used << ", "
                << "max_substeps = " << PAIR_MAP_MAX_SUBSTEPS << ", "
                << "reason = " << reason << ", "
                << "last_error = " << last_error << ", ";
            return msg.str();
        };

        if (recommended_substeps > 1) {
            std::vector<Body> accepted_bodies;
            if (try_pair_map_substeps(original_bodies, accepted_bodies, pair, dt, G, recommended_substeps, last_iterations, last_error)) {
                bodies = accepted_bodies;
                retry_succeeded = true;
                substeps_used = recommended_substeps;
                return;
            }
            substeps_used = recommended_substeps;
        }
        else {
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
        }

        if (!PAIR_MAP_RETRY_ENABLED) {
            throw std::runtime_error(make_failure_message("retry disabled"));
        }

        int substeps = std::max(2, recommended_substeps * 2);
        substeps = next_power_of_two_substeps(substeps);

        while (substeps <= PAIR_MAP_MAX_SUBSTEPS) {
            std::vector<Body> accepted_bodies;
            if (try_pair_map_substeps(bodies, accepted_bodies, pair, dt, G, recommended_substeps, last_iterations, last_error)) {
                bodies = accepted_bodies;
                retry_succeeded = true;
                substeps_used = substeps;
                return;
            }

            substeps_used = substeps;
            if (substeps == PAIR_MAP_MAX_SUBSTEPS) {
                break;
            }

            substeps = std::min(substeps * 2, PAIR_MAP_MAX_SUBSTEPS);
        }

        throw std::runtime_error(make_failure_message("maximum pair-local substeps exhausted"));
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