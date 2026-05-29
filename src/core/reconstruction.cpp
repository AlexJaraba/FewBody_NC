#include "core/reconstruction.h"

std::vector<Vec3> reconstruct_cartesian_positions(const CanonicalState& state) {
    const int N = static_cast<int>(state.Q.size());
    
    std::vector<Vec3> r(N);
    if (N == 0) return r;  // Handle empty state case

    Vec3 r0 = state.com_position;  // Start with the center of mass position
    double M_prev = state.physical_mass[0];

    for (int k = 1; k < N; ++k) {
        const double mk = state.physical_mass[k];
        const double Mk = M_prev + mk;
        const double beta = mk / Mk;

        r0 -= beta * state.Q[k];  // Update the center of mass position
        M_prev = Mk;  // Update the total mass
    }
    r[0] = r0;  // The first position is the center of mass position

    Vec3 com_prev = r0;
    M_prev = state.physical_mass[0];

    for (int k = 1; k < N; ++k) {
        const double mk = state.physical_mass[k];
        const double Mk = M_prev + mk;

        r[k] = com_prev + state.Q[k];  // Compute the position of the k-th particle
        com_prev = (M_prev * com_prev + mk * r[k]) / Mk;  // Update the center of mass position
        M_prev = Mk;  // Update the total mass
    }
    return r;
};

std::vector<Vec3> reconstruct_cartesian_velocities(const CanonicalState& state) {
    const int N = static_cast<int>(state.P.size());

    std::vector<Vec3> v(N);
    if (N == 0) return v;  // Handle empty state case

    Vec3 v0 = state.com_velocity;  // Start with the center of mass velocity
    double M_prev = state.physical_mass[0];

    for (int k = 1; k < N; ++k) {
        const double mk = state.physical_mass[k];
        const double Mk = M_prev + mk;
        const double beta = mk / Mk;

        v0 -= beta * (state.P[k] / state.mu[k]);  // Update the center of mass velocity
        M_prev = Mk;  // Update the total mass
    }
    v[0] = v0;  // The first velocity is the center of mass velocity

    Vec3 vcom_prev = v0;
    M_prev = state.physical_mass[0];

    for (int k = 1; k < N; ++k) {
        const double mk = state.physical_mass[k];
        const double Mk = M_prev + mk;

        v[k] = vcom_prev + (state.P[k] / state.mu[k]);  // Compute the velocity of the k-th particle
        vcom_prev = (M_prev * vcom_prev + mk * v[k]) / Mk;  // Update the center of mass velocity
        M_prev = Mk;  // Update the total mass
    }
    return v;
};