#include "corrector.h"
#include "operators.h"

SymplecticCorrector::SymplecticCorrector(double coefficient) : coefficient_(coefficient) {}

void SymplecticCorrector::apply_forward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
//     const double h = coefficient_ * dt;

//     kick_operator(bodies, pairs, 0.5 * h, G);
//     for (auto& body : bodies){
//         body.updateVelocityFromMomentum();
//     }

//     drift_operator(bodies, h);

//     kick_operator(bodies, pairs, 0.5 * h, G);
//     for (auto& body : bodies){
//         body.updateVelocityFromMomentum();
//     }
}

void SymplecticCorrector::apply_backward(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
    // apply_forward(bodies, pairs, -dt, G);
}