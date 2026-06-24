#include <stdexcept>
#include <algorithm>

#include "integrators/hb15.h"
#include "integrators-helper/hb15/pair_map.h"
#include "dynamics/pairing.h"
#include "dynamics/timestep_planner.h"

namespace {
    struct HB15HamiltonianBookKeeping {
        int body_count = 0;
        int pair_count = 0;

        /*
        
        In the all-pairs split, each body's kinetic energy appears once in every pair that contains that body.
        For N bodies, that is N - 1 appearances.
        The Newtonian Hamiltonian needs one copy, so the correction Hamiltonian is:

            H_correction = -(N - 2) T
        
        where T is the total Cartesian kinetic energy.
        The second-order symmetric HB15 step applies pair maps in two half sweeps:

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

    int deepest_nonempty_level(const HB15PairLevelSchedule& schedule) {
        int deepest_level = 0;
        for (const HB15PairLevelGroup& group : schedule.levels) {
            if (!group.pairs.empty()) {
                deepest_level = std::max(deepest_level, group.level);
            }
        }
        return deepest_level;
    }
    void apply_block_level(const HB15& hb15, std::vector<Body>& bodies, const HB15PairLevelSchedule& schedule, int level, int max_level, double dt, double G) {
        if (level > max_level || level >= static_cast<int>(schedule.levels.size())) {
            return;
        }

        const std::vector<Pair>& active_pairs = schedule.levels[static_cast<std::size_t>(level)].pairs;

        if (level == max_level) {
            hb15.apply_pair_group(bodies, active_pairs, dt, G);
            return;
        }

        hb15.apply_pair_group(bodies, active_pairs, 0.5 * dt, G);

        apply_block_level(hb15, bodies, schedule, level+ 1, max_level, 0.5 * dt, G);
        apply_block_level(hb15, bodies, schedule, level+ 1, max_level, 0.5 * dt, G);

        hb15.apply_pair_group(bodies, active_pairs, 0.5 * dt, G);
    }
    void apply_checked_pair_map(std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        HB15PairMapResult result = apply_hb15_pair_kepler_map(bodies, pair.i, pair.j, dt, G);
        if (!result.converged) {
            throw std::runtime_error("HB15 pair Kepler map failed to converge.");
        }
    }
    /*
        Flow of the HB15 kinetic remainder Hamiltonian.
        In the all-pairs HB15 split, the sum of pair Kepler Hamiltonians counts each body's kinetic energy N - 1 times.
        The true Newtonian Hamiltonian needs one copy, so the remainder/correction term is:

            H_remainder = -(N - 2) * T
        
        where:

            T = sum_i[(p_i)^2 / (2 * m_i)]
        
        Since this remainder is purely kinetic, its exact flow is a drift:

            r_i -> r_i + remainder_dt * v_i
            v_i -> v_i
        
        The coefficient -(N - 2) is already included in remainder_dt
    */
    void apply_hb15_remainder_flow(std::vector<Body>& bodies, double remainder_dt) {
        for (Body& body : bodies) {
            body.position += remainder_dt * body.velocity;
            body.updateMomentumFromVelocity();
        }
    }

    HB15HamiltonianBookKeeping make_hb15_bookkeeping(const std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt) {
        HB15HamiltonianBookKeeping bookkeeping;

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

HB15::HB15(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs_preserve_order(fixed_pairs)) {}

void HB15::apply_pair_group(std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const {
    const std::vector<Pair> ordered_pairs = canonicalize_pairs_preserve_order(active_pairs);
    const double pair_half_dt = 0.5 * dt;

    for (const Pair& pair : ordered_pairs) {
        apply_checked_pair_map(bodies, pair, pair_half_dt, G);
    }
    for (auto it = ordered_pairs.rbegin(); it != ordered_pairs.rend(); ++it) {
        apply_checked_pair_map(bodies, *it, pair_half_dt, G);
    }
}

void HB15::step(std::vector<Body>& bodies, double dt, double G) {
    const HB15HamiltonianBookKeeping bookkeeping = make_hb15_bookkeeping(bodies, pairs_, dt);
    if (bookkeeping.body_count <= 1 || bookkeeping.pair_count == 0) {
        return;
    }

    apply_hb15_remainder_flow(bodies, bookkeeping.correction_half_dt);
    apply_pair_group(bodies, pairs_, dt, G);
    apply_hb15_remainder_flow(bodies, bookkeeping.correction_half_dt);
}

void HB15::step_block(std::vector<Body>& bodies, const HB15PairLevelSchedule& schedule, double dt, double G) const {
    const HB15HamiltonianBookKeeping bookkeeping = make_hb15_bookkeeping(bodies, pairs_, dt);
    const int max_level = deepest_nonempty_level(schedule);

    if (bookkeeping.body_count <= 1 || bookkeeping.pair_count == 0) {
        return;
    }
    if (schedule.levels.empty()) {
        return;
    }

    apply_hb15_remainder_flow(bodies, bookkeeping.correction_half_dt);
    apply_block_level(*this, bodies, schedule, 0, max_level, dt, G);
    apply_hb15_remainder_flow(bodies, bookkeeping.correction_half_dt);
}

const std::vector<Pair>& HB15::pairs() const {
    return pairs_;
}