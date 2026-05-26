#pragma once

#include <vector>

#include "canonical_state.h"
#include "pairing.h"

double compute_perturbation_hamiltonian(
    const CanonicalState& state,
    const std::vector<Pair>& pairs,
    double G
);

std::vector<Vec3> compute_perturbation_gradient(const CanonicalState& state, const std::vector<Pair>& pairs, double G);