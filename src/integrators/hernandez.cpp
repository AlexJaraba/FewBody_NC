#include <vector> 
#include <cmath>
#include <stdexcept>

#include "hernandez.h"
#include "pairing.h"
#include "propagator.h"

static void updateAccelerationPerturbation(
    Body& body,
    const std::vector<Body>& bodies,
    const std::vector<Pair>& pairs,
    int index,
    double G)
{
    std::vector<double> acc(3, 0.0);

    for (size_t j = 0; j < bodies.size(); ++j) {
        if (j == size_t(index)) continue;

        if (is_kepler_pair(index, j, pairs)) continue;

        std::vector<double> dr(3);
        for (int k = 0; k < 3; ++k)
            dr[k] = bodies[j].position[k] - body.position[k];

        double r = std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]) + 1e-12;

        for (int k = 0; k < 3; ++k)
            acc[k] += G * bodies[j].mass * dr[k] / (r*r*r);
    }

    body.acceleration = acc;
}

static void kepler_pair_step(Body& bi, Body& bj, double dt, double G)
{
    std::vector<double> r_rel(3), v_rel(3);

    for (int k = 0; k < 3; ++k) {
        r_rel[k] = bi.position[k] - bj.position[k];
        v_rel[k] = bi.velocity[k] - bj.velocity[k];
    }

    double mu = G * (bi.mass + bj.mass);

    StateVector result = propagate_universal(mu, r_rel, v_rel, dt);

    if (!result.converged) {
        throw std::runtime_error("Kepler solve failed.");
    }
    
    double m_tot = bi.mass + bj.mass;

    std::vector<double> r_cm(3), v_cm(3);
    for (int k = 0; k < 3; ++k) {
        r_cm[k] = (bi.mass * bi.position[k] + bj.mass * bj.position[k]) / (bi.mass + bj.mass);
        v_cm[k] = (bi.mass * bi.velocity[k] + bj.mass * bj.velocity[k]) / (bi.mass + bj.mass);

        r_cm[k] += v_cm[k] * dt; // Drift center of mass
    }

    for (int k = 0; k < 3; ++k) {
        bi.position[k] = r_cm[k] + (bj.mass / m_tot) * result.r[k];
        bj.position[k] = r_cm[k] - (bi.mass / m_tot) * result.r[k];

        bi.velocity[k] = v_cm[k] + (bj.mass / m_tot) * result.v[k];
        bj.velocity[k] = v_cm[k] - (bi.mass / m_tot) * result.v[k];
    }
}

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : pairs_(fixed_pairs) {}

void Hernandez::step(std::vector<Body>& bodies, double dt)
{
    extern double G;
    int N = bodies.size();

    const std::vector<Pair>& pairs = pairs_;

    // Half Kick
    for (int i = 0; i < N; ++i) {
        updateAccelerationPerturbation(bodies[i], bodies, pairs, i, G);
    }

    for (auto& body : bodies) {
        body.updateVelocity(0.5 * dt);
    }

    // Kepler Half Step
    for (const auto& p : pairs) {
        kepler_pair_step(bodies[p.i], bodies[p.j], 0.5 * dt, G);
    }

    // Drift Unpaired Bodies
    std::vector<bool> paired(N, false);
    for (const auto& p : pairs) {
        paired[p.i] = true;
        paired[p.j] = true;
    }

    for (int i = 0; i < N; ++i) {
        if (!paired[i]) {
            bodies[i].updatePosition(dt);
        }
    }

    // Kepler Half Step
    for (const auto& p : pairs) {
        kepler_pair_step(bodies[p.i], bodies[p.j], 0.5 * dt, G);
    }

    // Half Kick
    for (int i = 0; i < N; ++i) {
        updateAccelerationPerturbation(bodies[i], bodies, pairs, i, G);
    }

    for (auto& body : bodies) {
        body.updateVelocity(0.5 * dt);
    }
}