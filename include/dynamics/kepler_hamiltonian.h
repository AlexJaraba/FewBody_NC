#pragma once

#include "dynamics/hamiltonian.h"

class KeplerHamiltonian : public Hamiltonian {
public:
    explicit KeplerHamiltonian(double G_);
    double energy(const CanonicalState& state) const override;
    ForceResult forces(const CanonicalState& state) const override;
private:
    double G;
};