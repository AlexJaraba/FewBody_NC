#include <cmath>
#include <vector>

#include "corrector.h"

void SymplecticCorrector::apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const{
    apply(state, pairs, dt, G, +1.0);
}

void SymplecticCorrector::apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
    apply(state, pairs, dt, G, -1.0);
}

void SymplecticCorrector::apply(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G, double sign) const {
    const double c = sign * (dt * dt) / 12.0;
    const int N = static_cast<int>(state.Q.size());

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

    for (const auto& pair : pairs) {
        int i = pair.i;
        int j = pair.j;

        std::vector<double> dr(3);
        for (int k = 0; k < 3; ++k) {
            dr[k] = r[j][k] - r[i][k];
        }
        double r2 = 0.0;
        for (int k = 0; k < 3; ++k) {
            r2 += dr[k] * dr[k];
        }

        const double dist = std::sqrt(r2) + 1e-15;
        const double coeff = (c * G * state.physical_mass[i] * state.physical_mass[j]) / (dist * dist * dist);

        for (int k = 0; k < 3; ++k) {
            const double dp = coeff * dr[k];
            state.P[i][k] += dp;
            state.P[j][k] -= dp;
        }
    }
}
