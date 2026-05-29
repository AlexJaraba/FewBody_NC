#include <vector> 

#include "integrators/hernandez.h"
#include "dynamics/pairing.h"
#include "dynamics/operators.h"
#include "dynamics/corrector.h"

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)), composition_({
    {OperatorType::KEPLER, 0.5},
    {OperatorType::KICK, 1.0},
    {OperatorType::KEPLER, 0.5},
}) {};

void Hernandez::step(CanonicalState& state, double dt, double G) {
    composition_.execute(state, pairs_, dt, G);
}