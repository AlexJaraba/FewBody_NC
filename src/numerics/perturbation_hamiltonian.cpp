#include <cmath>

#include "canonical_state.h"
#include "pairing.h"
#include "jacobi.h"
#include "vec3.h"

double compute_perturbation_hamiltonian(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    auto r = reconstruct_cartesian_position(state);
    double H = 0.0;

    for (const auto& pair : pairs) {
        const int i = pair.i;
        const int j = pair.j;

        Vec3 dr = r[j] - r[i];
        double r2 = dr.norm2();

        const double dist = std::sqrt(r2) + 1e-15;
        H -= (G * state.physical_mass[i] * state.physical_mass[j]) / dist;
        
    }
    return H;
}

std::vector<Vec3> compute_perturbation_gradient(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    const int N = state.Q.size();
    std::vector<Vec3> grad(N);
    auto r = reconstruct_cartesian_position(state);
    for (const auto& pair : pairs) {
        const int i = pair.i;
        const int j = pair.j;
        Vec3 dr;

        dr = r[j] - r[i];
        const double r2 = dr.norm2();

        const double dist = std::sqrt(r2) + 1e-15;
        const double coeff = (G * state.physical_mass[i] * state.physical_mass[j]) / (dist * dist * dist);

        Vec3 f = coeff * dr;

        grad[i] -= f;
        grad[j] += f;
    }
    return grad;
}