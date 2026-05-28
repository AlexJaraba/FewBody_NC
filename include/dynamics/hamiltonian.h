#pragma once

#include "core/canonical_state.h"
#include "numerics/force_result.h"

class Hamiltonian {
public:
    virtual ~Hamiltonian() = default;
    virtual double energy(const CanonicalState& state) const = 0;
    virtual ForceResult forces(const CanonicalState& state) const = 0;
};