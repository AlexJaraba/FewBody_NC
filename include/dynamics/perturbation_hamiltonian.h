#pragma once

#include <vector>

#include "core/canonical_state.h"
#include "dynamics/pairing.h"

double compute_perturbation_hamiltonian(
    const CanonicalState& state,
    const std::vector<Pair>& pairs,
    double G
);

std::vector<Vec3> compute_perturbation_gradient(const CanonicalState& state, const std::vector<Pair>& pairs, double G);