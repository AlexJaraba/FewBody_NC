#include <stdexcept>

#include "integrators-helper/hb15/pair_map.h"
#include "integrators-helper/hb15/pair_state.h"
#include "numerics/propagator.h"

/* ====================================================================================

    Apply one physical pairwsie Kepler map.

    The universal-variab;le propagator expects canonical relative momentum:

        p_rel = mu * u
    
    where:

        mu = reduced mass
        u = relative velocity

    The gravitational parameter for the relative Kepler problem is:

        G * (m_i + m_j)

    After Propagation:
        q_new = propagated relative position
        p_new = propagated relative velocity
        u_new = p_new / mu

    The COM evolves freely:
        R_new = R + dt * V
        V_new = V

   ==================================================================================== */

HB15PairMapResult apply_hb15_pair_kepler_map(std::vector<Body>& bodies, int i, int j, double dt, double G) {
    HB15PairState pair = HB15PairState::from_bodies(bodies, i, j);

    if (pair.relative_position.norm() <= 1e-14) {
        throw std::runtime_error("HB15 pair Kepler map received a pair with near-zero separation.");
    }

    const Vec3 relative_momentum = pair.reduced_mass * pair.relative_velocity;

    CanonicalStateVector propagated = propagate_universal(pair.gravitational_parameter(G), pair.reduced_mass, pair.relative_position, relative_momentum, dt);

    if (!propagated.converged) {
        return {false, propagated.iterations};
    }

    pair.relative_position = propagated.q;
    pair.relative_velocity = propagated.p / pair.reduced_mass;
    pair.com_position += dt * pair.com_velocity;
    pair.write_to_bodies(bodies);

    return {true, propagated.iterations};
}