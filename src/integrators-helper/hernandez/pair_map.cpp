#include <stdexcept>

#include "integrators-helper/hernandez/pair_map.h"
#include "integrators-helper/hernandez/pair_state.h"
#include "numerics/propagator.h"

/* ====================================================================================

    Apply one physical pairwise Kepler map.

    The universal-variable propagator expects canonical relative momentum:

        p_rel = mu * u
    
    where:

        mu = reduced mass
        u = relative velocity

    The gravitational parameter for the relative Kepler problem is:

        G * (m_i + m_j)

    After Propagation:
        q_new = propagated relative position
        p_new = propagated relative momentum
        u_new = p_new / mu

    The COM evolves freely:
        R_new = R + dt * V
        V_new = V

   ==================================================================================== */

HernandezPairMapResult propagatePairKepler(std::vector<Body>& bodies, int i, int j, double dt, double G) {
    HernandezPairState pair = HernandezPairState::pairState(bodies, i, j);

    if (pair.relative_position.norm() <= 1e-14) {
        throw std::runtime_error("Hernandez pair Kepler map received a pair with near-zero separation.");
    }

    const Vec3 relative_momentum = pair.reduced_mass * pair.relative_velocity;

<<<<<<< Updated upstream
    CanonicalStateVector propagated = propagate_universal(pair.gravitational_parameter(G), pair.reduced_mass, pair.relative_position, relative_momentum, dt);
=======
    KeplerPropagationResult propagated = propagateUniversal(pair.gravitationalParameter(G), pair.reduced_mass, pair.relative_position, relative_momentum, dt);
>>>>>>> Stashed changes

    if (!propagated.converged) {
        return {false, propagated.iterations};
    }

    pair.relative_position = propagated.q;
    pair.relative_velocity = propagated.p / pair.reduced_mass;
    pair.com_position += dt * pair.com_velocity;
    pair.writeToBodies(bodies);

    return {true, propagated.iterations};
}