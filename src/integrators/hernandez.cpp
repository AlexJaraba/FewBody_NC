#include <vector> 
#include <cmath>

#include "hernandez.h"
#include "../numerics/pairing.h"
#include "../propagator/propagator.h"

static std::vector<Body> clone_bodies(const std::vector<Body>& bodies) {
    return bodies;
}

static void predict_bodies(std::vector<Body>& bodies, double dt)
{
    for (auto& b : bodies) {
        b.updatePosition(dt);
    }
}

static void updateAccelerationPerturbation(
    Body& body,
    const std::vector<Body>& bodies,
    const std::vector<Pair>& pairs,
    int index,
    double G)
{
    std::vector<double> acc(3, 0.0);

    for (size_t j = 0; j < bodies.size(); ++j) {
        if (j == size_t index) continue;

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

    if (!result.converged) return;
        double m_tot = bi.mass + bj.mass;

    std::vector<double> r_cm(3), v_cm(3);
    for (int k = 0; k < 3; ++k) {
        r_cm[k] = (bi.mass * bi.position[k] + bj.mass * bj.position[k]) / m_tot;
        v_cm[k] = (bi.mass * bi.velocity[k] + bj.mass * bj.velocity[k]) / m_tot;
    }

    for (int k = 0; k < 3; ++k) {
        bi.position[k] = r_cm[k] + (bj.mass / m_tot) * result.r[k];
        bj.position[k] = r_cm[k] - (bi.mass / m_tot) * result.r[k];

        bi.velocity[k] = v_cm[k] + (bj.mass / m_tot) * result.v[k];
        bj.velocity[k] = v_cm[k] - (bi.mass / m_tot) * result.v[k];
    }
}

void Hernandez::step(std::vector<Body>& bodies, double dt)
{
    extern double G;
    int N = bodies.size();

    std::vector<Body> predicted = clone_bodies(bodies);
    predict_bodies(predicted, dt);

    std::vector<Pair> pairs = build_kepler_pairs(predicted, G);

    // Kick
    for (int i = 0; i < N; ++i) {
        updateAccelerationPerturbation(bodies[i], bodies, pairs, i, G);
    }

    for (auto& body : bodies) {
        body.updateVelocity(0.5 * dt);
    }

    // Kepler Drift
    for (const auto& p : pairs) {
        kepler_pair_step(bodies[p.i], bodies[p.j], dt, G);
    }

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

    // Kick
    for (int i = 0; i < N; ++i) {
        updateAccelerationPerturbation(bodies[i], bodies, pairs, i, G);
    }

    for (auto& body : bodies) {
        body.updateVelocity(0.5 * dt);
    }
}