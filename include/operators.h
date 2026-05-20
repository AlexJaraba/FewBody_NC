#pragma once

#include <vector>

#include "body.h"
#include "pairing.h"
#include "jacobi.h"
#include "canonical_state.h"

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