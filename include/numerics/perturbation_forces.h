#pragma once

#include <vector>

#include "core/canonical_state.h"
#include "dynamics/pairing.h"
#include "numerics/force_result.h"

ForceResult compute_perturbation_forces(const CanonicalState& state, const std::vector<Pair>& pairs, double G);