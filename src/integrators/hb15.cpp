#include <stdexcept>

#include "integrators/hb15.h"
#include "integrators-helper/hb15/pair_map.h"
#include "dynamics/pairing.h"

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
        The second-order symmetric HB16 step applies pair maps in two half sweeps:

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


    void apply_checked_pair_map(std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        HB15PairMapResult result = apply_hb15_pair_kepler_map(bodies, pair.i, pair.j, dt, G);
        if (!result.converged) {
            throw std::runtime_error("HB15 pair Kepler map failed to converge.");
        }
    }
    /*
        Flow of a pure kinetic Hamiltonian * c * T.
        For a Hamiltonian * c * T, where

            T = sum_i([p_i]^2 / [2 * m_i])
        
        the exact flow is:

            r_i -> r_i + c * dt * v_t
            v_i -> unchanged
        
        In the all-pairs HB15 split, c = -(N - 2), because the sum of all pair Kepler Hamiltonians counts each body's kinetic energy N - 1 times.
        While the true Newtonian Hamiltonian needs it once.
    */
    void apply_kinetic_correction_drift(std::vector<Body>& bodies, double drift_dt) {
        for (Body& body : bodies) {
            body.position += drift_dt * body.velocity;
            body.updateMomentumFromVelocity();
        }
    }
}

HB15::HB15(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)) {}

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

void HB15::step(std::vector<Body>& bodies, double dt, double G) {
    const HB15HamiltonianBookKeeping bookkeeping = make_hb15_bookkeeping(bodies, pairs_, dt);
    if (bookkeeping.body_count <= 1 || bookkeeping.pair_count == 0) {
        return;
    }

    apply_kinetic_correction_drift(bodies, bookkeeping.correction_half_dt);

    for (const Pair& pair : pairs_) {
        apply_checked_pair_map(bodies, pair, bookkeeping.pair_half_dt, G);
    }
    for (auto it = pairs_.rbegin(); it != pairs_.rend(); ++it) {
        apply_checked_pair_map(bodies, *it, bookkeeping.pair_half_dt, G);
    }

    apply_kinetic_correction_drift(bodies, bookkeeping.correction_half_dt);
}

const std::vector<Pair>& HB15::pairs() const {
    return pairs_;
}