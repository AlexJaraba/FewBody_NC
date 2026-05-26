#include <vector>

#include "jacobi.h"

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
    const int N = bodies.size();
    if (N == 0) return;

    Vec3 R_prev;
    Vec3 V_prev;

    double M_prev = bodies[0].mass;

    //Recursive jacobi reconstruction
    for (int i = 1; i < N; ++i) {
        const double mi = bodies[i].mass;
        const double mu = state.mu[i];

        Vec3 r_com_prev;
        Vec3 v_com_prev;

        r_com_prev = R_prev / M_prev;
        v_com_prev = V_prev / M_prev;

        bodies[i].position = state.Q[i] + r_com_prev;
        bodies[i].velocity = (state.P[i] / mu) + v_com_prev;

        R_prev += mi * bodies[i].position;
        V_prev += mi * bodies[i].velocity;

        M_prev += mi;
    }

    // Recover Cartesian momenta
    Vec3 weighted_pos;
    Vec3 weighted_vel;
    for (int i = 1; i < N; ++i) {
        weighted_pos += bodies[i].mass * bodies[i].position;
        weighted_vel += bodies[i].mass * bodies[i].velocity;
    }

    bodies[0].position = (state.com_position * M_prev - weighted_pos) / bodies[0].mass;
    bodies[0].velocity = (state.com_velocity * M_prev - weighted_vel) / bodies[0].mass;


    for (int i = 0; i < N; ++i) {
        bodies[i].updateMomentumFromVelocity();
    }
}

std::vector<Vec3> reconstruct_cartesian_position(const CanonicalState& state) {
    const int N = state.Q.size();
    std::vector<Vec3> r(N);
    Vec3 R_prev;
    double M_prev = state.physical_mass[0];
    
    for (int i = 1; i < N; ++i) {
        Vec3 r_com_prev = R_prev / M_prev;
        r[i] = r_com_prev + state.Q[i];
        R_prev += state.physical_mass[i] * r[i];
        M_prev += state.physical_mass[i];
    }
    return r;
}