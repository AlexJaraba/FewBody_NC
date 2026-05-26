#include <cmath>

#include "corrector.h"
#include "vec3.h"

void SymplecticCorrector::apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const{
    apply(state, pairs, dt, G, +1.0);
}

void SymplecticCorrector::apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
    apply(state, pairs, dt, G, -1.0);
}

void SymplecticCorrector::apply(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G, double sign) const {
    const double c = sign * (dt * dt) / 12.0;
    const int N = static_cast<int>(state.Q.size());

    std::vector<Vec3> r(N);
    Vec3 R_prev;
    double M_prev = state.physical_mass[0];

    for (int i = 1; i < N; ++i) {
        Vec3 r_com_prev;
        r_com_prev = R_prev / M_prev;
        r[i] = r_com_prev + state.Q[i];
        R_prev += state.physical_mass[i] * r[i];
        M_prev += state.physical_mass[i];
    }

    for (const auto& pair : pairs) {
        int i = pair.i;
        int j = pair.j;

        Vec3 dr;
        dr = r[j] - r[i];
        double r2 = 0.0;
        r2 += dr.norm2();

        const double dist = std::sqrt(r2) + 1e-15;
        const double coeff = (c * G * state.physical_mass[i] * state.physical_mass[j]) / (dist * dist * dist);
        Vec3 dp = coeff * dr;
        
        state.P[i] += dp;
        state.P[j] -= dp;
    }
}
