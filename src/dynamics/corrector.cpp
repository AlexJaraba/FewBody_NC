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
    auto r = reconstruct_cartesian_positions(state);
}
