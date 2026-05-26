#include <cmath>

#include "diagnostics.h"

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
            double r2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double dr = bodies[j].position[k] - bodies[i].position[k];
                r2 += dr * dr;
            }
            double r = std::sqrt(r2);
            d.potential_energy -= G * bodies[i].mass * bodies[j].mass / r;
        }
    }

    d.total_energy = d.kinetic_energy + d.potential_energy;

    // Compute linear momentum
    double px = 0.0, py = 0.0, pz = 0.0;
    for (const auto& body : bodies) {
        px += body.momentum[0];
        py += body.momentum[1];
        pz += body.momentum[2];
    }

    d.linear_momentum = std::sqrt(px*px + py*py + pz*pz);

    // Compute angular momentum
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;
    for (const auto& body : bodies) {
        Lx += body.position[1] * body.momentum[2] - body.position[2] * body.momentum[1];
        Ly += body.position[2] * body.momentum[0] - body.position[0] * body.momentum[2];
        Lz += body.position[0] * body.momentum[1] - body.position[1] * body.momentum[0];
    }

    d.angular_momentum = std::sqrt(Lx*Lx + Ly*Ly + Lz*Lz);

    // Compute center of mass drift
    double mx = 0.0, my = 0.0, mz = 0.0, total_mass = 0.0;
    for (const auto& body : bodies) {
        total_mass += body.mass;
        mx += body.position[0] * body.mass;
        my += body.position[1] * body.mass;
        mz += body.position[2] * body.mass;
    }

    mx /= total_mass;
    my /= total_mass;
    mz /= total_mass;

    d.com_drift = std::sqrt(mx*mx + my*my + mz*mz);

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
        double r2 = 0.0;

        for (int k = 0; k < 3; ++k) {
            double dr = bodies[j].position[k] - bodies[i].position[k];
            r2 += dr * dr;
        }

        double r = std::sqrt(r2);
        E -= (G * bodies[i].mass * bodies[j].mass) / r;
    }
    return E;
}