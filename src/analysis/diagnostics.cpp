#include <cmath>

#include "analysis/diagnostics.h"
#include "math/vec3.h"

Diagnostics compute_diagnostics(const std::vector<Body>& bodies, double G, double dt) {
    Diagnostics d{};

    // Compute kinetic energy
    for (const auto& body : bodies) {
        d.kinetic_energy += body.kineticEnergy();
    }

    // Compute potential energy
    const int N = bodies.size();
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec3 dr = bodies[j].position - bodies[i].position;
            double r = dr.norm();
            d.potential_energy -= (G * bodies[i].mass * bodies[j].mass) / r;
        }
    }

    d.total_energy = d.kinetic_energy + d.potential_energy;

    // Compute linear momentum
    Vec3 P;
    for (const auto& body : bodies) {
        P += body.momentum;
    }

    d.linear_momentum = P.norm();

    // Compute angular momentum
    Vec3 L;
    for (const auto& body : bodies) {
        L += cross(body.position, body.momentum);
    }

    // Compute center of mass drift
    Vec3 Rcm;
    double total_mass = 0.0;
    for (const auto& body : bodies) {
        total_mass += body.mass;
        Rcm += body.mass * body.position;
    }

    Rcm /= total_mass;
    d.com_drift = Rcm.norm();

    // Second-order shadow Hamiltonian estimate
    // double p2sum = 0.0;
    // for (const auto& body : bodies) {
    //     p2sum += body.momentumMagnitudeSquared();
    // }

    d.shadow_energy = d.total_energy;

    d.timestep = dt;

    return d;
}

double compute_perturbation_energy(const std::vector<Body>& bodies, const std::vector<Pair>& pairs, double G) {
    double E = 0.0;

    for (const auto& pair : pairs) {
        int i = pair.i;
        int j = pair.j;
        Vec3 dr = bodies[j].position - bodies[i].position;
        double r = dr.norm();
        E -= (G * bodies[i].mass * bodies[j].mass) / r;
    }
    return E;
}