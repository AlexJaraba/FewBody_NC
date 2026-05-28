#include <cmath>

#include "dynamics/kepler_hamiltonian.h"

KeplerHamiltonian::KeplerHamiltonian(double G_) : G(G_) {}

double KeplerHamiltonian::energy(const CanonicalState& state) const {
    double H = 0.0;
    for (size_t i =1; i < state.Q.size(); ++i) {
        const double r = state.Q[i].norm();
        const double p2 = state.P[i].norm2();
        const double mu = state.mu[i];
        const double M = state.M[i];

        H += (p2 / (2.0 * mu)) - ((G * mu * M) / r);
    }
    return H;
}

ForceResult KeplerHamiltonian::forces(const CanonicalState& state) const {
    ForceResult result;
    return result;
}