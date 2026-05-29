#pragma once

#include "core/canonical_state.h"
#include "core/variational_state.h"
#include "dynamics/pairing.h"

void variational_drift_operator(const CanonicalState& state, VariationalState& var_state, double dt);
void variational_kick_operator(const CanonicalState& state, VariationalState& var_state, const std::vector<Pair>& pairs, double dt, double G);