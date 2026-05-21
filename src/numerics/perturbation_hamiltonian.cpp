#include <cmath>

#include "canonical_state.h"
#include "pairing.h"

static std::vector<std::vector<double>> reconstruct_positions(const CanonicalState& state) {
    const int N = state.Q.size();

    std::vector<std::vector<double>> r(N, std::vector<double>(3, 0.0));
    std::vector<double> R_prev(3, 0.0);

    double M_prev = state.physical_mass[0];

    for (int i = 1; i < N; ++i) {
        std::vector<double> r_com_prev(3);
        for (int k = 0; k < 3; ++k) {
            r_com_prev[k] = R_prev[k] / M_prev;
        }
        for (int k = 0; k < 3; ++k) {
            r[i][k] = r_com_prev[k] + state.Q[i][k];
        }
        for (int k = 0; k < 3; ++k) {
            R_prev[k] += state.physical_mass[i] * r[i][k];
        }
        M_prev += state.physical_mass[i];
    }

    return r;
}

double compute_perturbation_hamiltonian(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    auto r = reconstruct_positions(state);
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
        H -= G * state.physical_mass[i] * state.physical_mass[j] / dist;
        
    }
    return H;
}