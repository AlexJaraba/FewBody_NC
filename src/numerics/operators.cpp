#include <cmath>
#include <stdexcept>

#include "operators.h"
#include "pairing.h"
#include "propagator.h"
#include "jacobi.h"

void drift_operator(std::vector<Body>& bodies, double dt) {
    for (auto& body: bodies) {
        for (int k = 0; k < 3; ++k) {
            body.position[k] += body.velocity[k] * dt;
        }
    }
}

static bool pair_contains(int i, int j, const std::vector<Pair>& pairs) {
    for (const auto& p : pairs){
        if ((p.i == i && p.j == j) || (p.i == j && p.j == i)) {
            return true;
        }
    }
    return false;
}

void kick_operator(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) {
    int N = bodies.size();
    std::vector<std::vector<double>> acc(N, std::vector<double>(3, 0.0));

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (pair_contains(i, j, pairs))
                continue;
            std::vector<double> dr(3);

            for (int k = 0; k < 3; ++k)
                dr[k] = bodies[j].position[k] - bodies[i].position[k];
            
            double r = std::sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]) + 1e-12;
            double factor = G / (r*r*r);

            for (int k = 0; k < 3; ++k) {
                double force = factor * dr[k];
                acc[i][k] += force * bodies[j].mass;
                acc[j][k] -= force * bodies[i].mass;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 3; ++k) {
            bodies[i].momentum[k] += bodies[i].mass * acc[i][k] * dt;
        }
    }
}

static void kepler_pair_step(Body& bi, Body& bj, double dt, double G) {
    std::vector<double> r_rel(3), p_rel(3), v_rel(3);
    double mu_red = (bi.mass * bj.mass) / (bi.mass + bj.mass);

    for (int k = 0; k < 3; ++k) {
        r_rel[k] = bi.position[k] - bj.position[k];
        p_rel[k] = mu_red * (bi.velocity[k] - bj.velocity[k]);
        v_rel[k] = p_rel[k] / mu_red;
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

void kepler_operator(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) {
    for (const auto& p : pairs) {
        kepler_pair_step(bodies[p.i], bodies[p.j], dt, G);
    }
}

void symmetric_kepler_operator(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) {
    //Forward half step
    for (const auto& p : pairs) {
        kepler_pair_step(bodies[p.i], bodies[p.j], dt, G);
    }

    //Backward half step
    std::vector<Pair> reversed = reverse_pairs(pairs);
    for (const auto& p : reversed) {
        kepler_pair_step(bodies[p.i], bodies[p.j], dt, G);
    }

}