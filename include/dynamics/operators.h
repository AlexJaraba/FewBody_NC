#pragma once

#include <vector>

#include "core/body.h"
#include "core/canonical_state.h"
#include "dynamics/pairing.h"
#include "dynamics/jacobi.h"

void drift_operator(
    CanonicalState& state,
    double dt
);

void kick_operator(
    CanonicalState& state,
    const std::vector<Pair>& pairs,
    double dt,
    double G
);

void kepler_operator(
    CanonicalState& state,
    const std::vector<Pair>& pairs,
    double dt,
    double G
);

void symmetric_kepler_operator(
    CanonicalState& state,
    const std::vector<Pair>& pairs,
    double dt,
    double G
);