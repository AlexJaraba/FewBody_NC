#include <cmath>

#include "dynamics/corrector.h"
#include "numerics/perturbation_forces.h"

/* ===============================================================

    Symplectic Corrector Prototype

    This file contains corrector hooks intended to reduce bounded oscillatory energy error in composed symplectic methods.
    The infrastructure exists, but correctors should remain disabled unless a specific corrector has been revalidated against the current Jacobi split:
        H = H_kepler_jacobi + H_perturbation

   =============================================================== */

void SymplecticCorrector::apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const{
    apply(state, pairs, dt, G, +1.0);
}

void SymplecticCorrector::apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
    apply(state, pairs, dt, G, -1.0);
}

void SymplecticCorrector::apply(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G, double sign) const {
    const double c = (sign * (dt * dt)) / 12.0;
    auto forces = compute_perturbation_forces(state, pairs, G);
    const auto& grad = forces.gradient;
    const int N = static_cast<int>(state.P.size());

    for (int i = 1; i < N; ++i) {
        state.P[i] += c * grad[i];
    }
}
