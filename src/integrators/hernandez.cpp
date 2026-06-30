#include "integrators/hernandez.h"
#include "dynamics/pairing.h"
#include "dynamics/operators.h"

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)), canonical_composition_({
    {OperatorType::KEPLER, 0.5},
    {OperatorType::KICK, 1.0},
    {OperatorType::KEPLER, 0.5},
}), body_stepper_(fixed_pairs) {};

void Hernandez::step(CanonicalState& state, double dt, double G) {
    step_canonical(state, dt, G);
}
void Hernandez::step_canonical(CanonicalState& state, double dt, double G) {
    canonical_composition_.execute(state, pairs_, dt, G);
}

void Hernandez::step(std::vector<Body>& bodies, double dt, double G) {
    step_bodies(bodies, dt, G);
}
void Hernandez::step_bodies(std::vector<Body>& bodies, double dt, double G) {
    body_stepper_.step(bodies, dt, G);
}

void Hernandez::apply_pair_group( std::vector<Body>& bodies, const std::vector<Pair>& active_pairs, double dt, double G) const {
    body_stepper_.apply_pair_group(bodies, active_pairs, dt, G);
}
void Hernandez::step_block(std::vector<Body>& bodies, const HernandezPairLevelSchedule& schedule, double dt, double G) const {
    body_stepper_.step_block(bodies, schedule, dt, G);
}
const std::vector<Pair>& Hernandez::pairs() const {
    return body_stepper_.pairs();
}