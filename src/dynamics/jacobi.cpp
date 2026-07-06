#include <vector>

#include "dynamics/jacobi.h"

/* ===============================================================

    Jacobi Coordinate Transformations:

    This file implements the transformations between Cartesian coordinates and Jacobi coordinates, 
    as well as the reconstruction of Cartesian states from Jacobi states.

    The main functions are:
    - compute_jacobi_state: Converts a vector of Body objects (with Cartesian positions and velocities) into a CanonicalState in Jacobi coordinates.
    - reconstruct_bodies: Converts a CanonicalState in Jacobi coordinates back into Cartesian positions and velocities, 
      updating the Body objects accordingly.
    
    For body index k > 0:

        Q_k = r_k - R_{0...k-1}
    
    where:
    
        R_{0...k-1} is the center of mass of all previous bodies.
    
    The conjugate momentum P_k is built using the reduced mass:

        P_k = mu_k * dQ_k/dt
    
    with:

        mu_k = m_k - m_k * M_{k-1} / M_k
    
    Q_0 and P_0 represent the total center-of-mass coordinate and momentum respectively.

   =============================================================== */

/*
    Build a CanonicalState from physical Cartesian bodies.
    The physical masses are preserved so the system can later be reconstructed back into Cartesian coordinates.
*/

CanonicalState compute_jacobi_state(const std::vector<Body>& bodies) {
    CanonicalState state;

    const int N = bodies.size();

    state.Q.resize(N);
    state.P.resize(N);
    state.mu.resize(N);
    state.M.resize(N);
    state.physical_mass.resize(N);

    double total_mass = 0.0;

    for (const auto& body : bodies) {
        total_mass += body.mass;
        state.com_position += body.mass * body.position;
        state.com_velocity += body.mass * body.velocity;
    }

    state.com_position /= total_mass;
    state.com_velocity /= total_mass;

    state.Q[0] = Vec3();
    state.P[0] = Vec3();
    state.mu[0] = bodies[0].mass;
    state.M[0] = bodies[0].mass;
    state.physical_mass[0] = bodies[0].mass;

    Vec3 com_pos;
    Vec3 com_vel;

    double enclosed_mass = bodies[0].mass;
    
    com_pos = bodies[0].mass * bodies[0].position;
    com_vel = bodies[0].mass * bodies[0].velocity;

    for (int i = 1; i < N; ++i) {
        const double mi = bodies[i].mass;
        const double mu = (mi * enclosed_mass) / (mi + enclosed_mass);

        Vec3 prev_com = com_pos / enclosed_mass;
        Vec3 prev_vel = com_vel / enclosed_mass;

        state.Q[i] = bodies[i].position - prev_com;
        state.P[i] = mu * (bodies[i].velocity - prev_vel);

        state.mu[i] = mu;
        state.M[i] = mi + enclosed_mass;
        state.physical_mass[i] = mi;

        com_pos += mi * bodies[i].position;
        com_vel += mi * bodies[i].velocity;

        enclosed_mass += mi;
    }
    return state;
}

/*
    Reconstruct physical Cartesian bodies from the current CanonicalState.
    This is the only place Jacobi-mode evolution returns to physical body coordinates during normal output/diagnostics.
*/

void reconstruct_bodies(const CanonicalState& state, std::vector<Body>& bodies) {
    const int N = static_cast<int>(state.Q.size());
    if (N == 0) return;  // Handle empty state case

    auto r = reconstruct_cartesian_positions(state);
    auto v = reconstruct_cartesian_velocities(state);

    for (int i = 0; i < N; ++i) {
        bodies[i].position = r[i];
        bodies[i].velocity = v[i];
        bodies[i].updateMomentumFromVelocity();
    }
}