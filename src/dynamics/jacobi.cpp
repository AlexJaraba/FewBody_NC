#include <vector>

#include "dynamics/jacobi.h"

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