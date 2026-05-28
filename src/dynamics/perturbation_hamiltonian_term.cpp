#include "dynamics/perturbation_hamiltonian_term.h"
#include "numerics/perturbation_forces.h"

PerturbationHamiltonianTerm::PerturbationHamiltonianTerm(const std::vector<Pair>& pairs_, double G_) : pairs(pairs_), G(G_) {}

double PerturbationHamiltonianTerm::energy(const CanonicalState& state) const {
    auto result = compute_perturbation_forces(state, pairs, G);
    return result.potential;
}

ForceResult PerturbationHamiltonianTerm::forces(const CanonicalState& state) const {
    return compute_perturbation_forces(state, pairs, G);
}