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

    state.com_position.resize(3, 0.0);
    state.com_velocity.resize(3, 0.0);

    double total_mass = 0.0;

    for (const auto& body : bodies) {
        total_mass += body.mass;
        for (int k = 0; k < 3; ++k) {
            state.com_position[k] += body.mass * body.position[k];
            state.com_velocity[k] += body.mass * body.velocity[k];
        }
    }

    for (int k = 0; k < 3; ++k) {
        state.com_position[k] /= total_mass;
        state.com_velocity[k] /= total_mass;
    }

    state.Q[0] = {0.0, 0.0, 0.0};
    state.P[0] = {0.0, 0.0, 0.0};
    state.mu[0] = bodies[0].mass;
    state.M[0] = bodies[0].mass;
    state.physical_mass[0] = bodies[0].mass;

    std::vector<double> com_pos(3, 0.0);
    std::vector<double> com_vel(3, 0.0);

    double enclosed_mass = bodies[0].mass;
    
    for (int k = 0; k < 3; ++k) {
        com_pos[k] = bodies[0].mass * bodies[0].position[k];
        com_vel[k] = bodies[0].mass * bodies[0].velocity[k];
    }

    for (int i = 1; i < N; ++i) {
        const double mi = bodies[i].mass;
        const double mu = (mi * enclosed_mass) / (mi + enclosed_mass);

        state.Q[i].resize(3);
        state.P[i].resize(3);

        for (int k = 0; k < 3; ++k) {
            const double prev_com = com_pos[k] / enclosed_mass;
            const double prev_vel = com_vel[k] / enclosed_mass;

            state.Q[i][k] = bodies[i].position[k] - prev_com;
            state.P[i][k] = mu * (bodies[i].velocity[k] - prev_vel);
        }

        state.mu[i] = mu;
        state.M[i] = mi + enclosed_mass;
        state.physical_mass[i] = mi;

        for (int k = 0; k <3; ++k) {
            com_pos[k] += mi * bodies[i].position[k];
            com_vel[k] += mi * bodies[i].velocity[k];
        }
        enclosed_mass += mi;
    }
    return state;
}

void reconstruct_bodies(const CanonicalState& state, std::vector<Body>& bodies) {
    const int N = bodies.size();
    if (N == 0) return;

    std::vector<double> R_prev(3, 0.0);
    std::vector<double> V_prev(3, 0.0);

    double M_prev = bodies[0].mass;

    for (int k = 0; k < 3; ++k) {
        R_prev[k] = bodies[0].mass * bodies[0].position[k];
        V_prev[k] = bodies[0].mass * bodies[0].velocity[k];
    }

    //Recursive jacobi reconstruction
    for (int i = 1; i < N; ++i) {
        const double mi = bodies[i].mass;
        const double mu = state.mu[i];

        std::vector<double> r_com_prev(3);
        std::vector<double> v_com_prev(3);

        for (int k = 0; k < 3; ++k) {
            r_com_prev[k] = R_prev[k] / M_prev;
            v_com_prev[k] = V_prev[k] / M_prev;
        }

        for (int k = 0; k < 3; ++k) {
            bodies[i].position[k] = state.Q[i][k] + r_com_prev[k];
            bodies[i].velocity[k] = (state.P[i][k] / mu) + v_com_prev[k];
        }

        for (int k = 0; k < 3; ++k) {
            R_prev[k] += mi * bodies[i].position[k];
            V_prev[k] += mi * bodies[i].velocity[k];
        }
        M_prev += mi;
    }

    // Recover Cartesian momenta
    for (int k = 0; k < 3; ++k) {
        double weighted_pos = 0.0;
        double weighted_vel = 0.0;
        for (int i = 1; i < N; ++i) {
            weighted_pos += bodies[i].mass * bodies[i].position[k];
            weighted_vel += bodies[i].mass * bodies[i].velocity[k];
        }
        bodies[0].position[k] = state.com_position[k] - (weighted_pos / bodies[0].mass);
        bodies[0].velocity[k] = state.com_velocity[k] - (weighted_vel / bodies[0].mass);
    }
    for (int i = 0; i < N; ++i) {
        bodies[i].updateMomentumFromVelocity();
    }
}