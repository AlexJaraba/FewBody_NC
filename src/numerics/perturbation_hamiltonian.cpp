#include <cmath>

#include "canonical_state.h"
#include "pairing.h"
#include "jacobi.h"

double compute_perturbation_hamiltonian(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    auto r = reconstruct_cartesian_position(state);
    double H = 0.0;

    for (const auto& pair : pairs) {
        const int i = pair.i;
        const int j = pair.j;
        double r2 = 0.0;

        for (int k = 0; k < 3; ++k) {
            const double dr =
                r[j][k] - r[i][k];
            r2 += dr * dr;
        }

        const double dist = std::sqrt(r2) + 1e-15;
        H -= (G * state.physical_mass[i] * state.physical_mass[j]) / dist;
        
    }
    return H;
}

std::vector<std::vector<double>> compute_perturbation_gradient(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    const int N = state.Q.size();
    std::vector<std::vector<double>> grad(N, std::vector<double>(3, 0.0));
    auto r = reconstruct_cartesian_position(state);
    for (const auto& pair : pairs) {
        const int i = pair.i;
        const int j = pair.j;
        std::vector<double> dr(3);
        double r2 = 0.0;

        for (int k = 0; k < 3; ++k) {
            dr[k] = r[j][k] - r[i][k];
            r2 += dr[k] * dr[k];
        }

        const double dist = std::sqrt(r2) + 1e-15;
        const double coeff = (G * state.physical_mass[i] * state.physical_mass[j]) / (dist * dist * dist);

        for (int k = 0; k < 3; ++k) {
            const double f = coeff * dr[k];

            grad[i][k] -= f;
            grad[j][k] += f;
        }
    }
    return grad;
}