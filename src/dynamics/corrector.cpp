#include <cmath>

#include "dynamics/corrector.h"
#include "math/vec3.h"
#include "core/reconstruction.h"

void SymplecticCorrector::apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const{
    apply(state, pairs, dt, G, +1.0);
}

void SymplecticCorrector::apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
    apply(state, pairs, dt, G, -1.0);
}

void SymplecticCorrector::apply(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G, double sign) const {
    const double c = sign * (dt * dt) / 12.0;
    const int N = static_cast<int>(state.Q.size());

    auto r = reconstruct_cartesian_positions(state);
}
