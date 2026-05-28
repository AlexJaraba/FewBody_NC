#pragma once

#include <vector>

#include "dynamics/hamiltonian.h"
#include "dynamics/pairing.h"

class PerturbationHamiltonianTerm : public Hamiltonian {
public:
    PerturbationHamiltonianTerm(const std::vector<Pair>& pairs_, double G_);;
    double energy(const CanonicalState& state) const override;
    ForceResult forces(const CanonicalState& state) const override;
private:
    std::vector<Pair> pairs;
    double G;
};