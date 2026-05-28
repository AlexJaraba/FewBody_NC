#include "core/reconstruction.h"

std::vector<Vec3> reconstruct_cartesian_positions(const CanonicalState& state) {
    const int N = state.Q.size();
    
    std::vector<Vec3> r(N);
    Vec3 weighted_pos;
    Vec3 R_prev;

    double M_prev = state.physical_mass[0];

    for (int i = 1; i < N; ++i) {
        Vec3 r_com_prev = R_prev / M_prev;
        r[i] = r_com_prev + state.Q[i];
        weighted_pos += state.physical_mass[i] * r[i];
        R_prev += state.physical_mass[i] * r[i];
        M_prev += state.physical_mass[i];
    }
    r[0] = (state.com_position * M_prev - weighted_pos) / state.physical_mass[0];
    return r;
};

std::vector<Vec3> reconstruct_cartesian_velocities(const CanonicalState& state) {
    const int N = state.P.size();

    std::vector<Vec3> v(N);
    Vec3 weighted_vel;
    Vec3 V_prev;

    double M_prev = state.physical_mass[0];

    for (int i = 1; i < N; ++i) {
        Vec3 v_com_prev = V_prev / M_prev;
        v[i] = v_com_prev + (state.P[i] / state.mu[i]);
        weighted_vel += state.physical_mass[i] * v[i];
        V_prev += state.physical_mass[i] * v[i];
        M_prev += state.physical_mass[i];
    }
    v[0] = (state.com_velocity * M_prev - weighted_vel) / state.physical_mass[0];
    return v;
};