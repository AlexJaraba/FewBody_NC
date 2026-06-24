#include "integrators/hernandez.h"
#include "dynamics/pairing.h"
#include "dynamics/operators.h"

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)), composition_({
    {OperatorType::KEPLER, 0.5},
    {OperatorType::KICK, 1.0},
    {OperatorType::KEPLER, 0.5},
}), cartesian_core_(fixed_pairs) {};

void Hernandez::step(CanonicalState& state, double dt, double G) {
    composition_.execute(state, pairs_, dt, G);
}

void Hernandez::step(std::vector<Body>& bodies, double dt, double G) {
    cartesian_core_.step(bodies, dt, G);
}
void Hernandez::apply_pair_group( std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const {
    cartesian_core_.apply_pair_group(bodies, active_pairs, dt, G);
}
void Hernandez::step_block(std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, double dt, double G) const {
    cartesian_core_.step_block(bodies, schedule, dt, G);
}
const std::vector<Pair>& Hernandez::pairs() const {
    return cartesian_core_.pairs();
}