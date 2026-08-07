#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "integrators-helper/hernandez/body_stepper.h"
#include "integrators-helper/hernandez/pair_map.h"
#include "dynamics/pairing.h"

namespace {
    constexpr double PAIR_MAP_NEAR_ZERO_DISTANCE = 1e-14;

    void validate_pair_before_map(const std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        if (pair.i < 0 || pair.j < 0 || pair.i >= static_cast<int>(bodies.size()) || pair.j >= static_cast<int>(bodies.size()) || pair.i == pair.j) {
            throw std::runtime_error("Hernandez pair map received invalid body indices.");
        }

        const Body& body_i = bodies[pair.i];
        const Body& body_j = bodies[pair.j];
        const Vec3 relative_position = body_j.position - body_i.position;
        const Vec3 relative_velocity = body_j.velocity - body_i.velocity;
        const double distance = relative_position.norm();
        const double relative_speed = relative_velocity.norm();

        if (!std::isfinite(distance) || !std::isfinite(relative_speed)) {
            std::ostringstream msg;
            msg << "Hernandez pair map received a non-finite encounter state. "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "distance = " << distance << ", "
                << "relative_speed = " << relative_speed;
            throw std::runtime_error(msg.str());            
        }
        if (distance <= PAIR_MAP_NEAR_ZERO_DISTANCE) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Hernandez near-singular pair encounter detected. "
                << "near_singular_encounter = true, "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_pair_i = " << pair.i << ", "
                << "failed_pair_j = " << pair.j << ", "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "distance = " << distance << ", "
                << "relative_speed = " << relative_speed;
            throw std::runtime_error(msg.str());         
        }
    }

    void apply_checked_pair_map(std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        validate_pair_before_map(bodies, pair, dt, G);

        const Vec3 relative_position = bodies[pair.j].position - bodies[pair.i].position;
        const Vec3 relative_velocity = bodies[pair.j].velocity - bodies[pair.i].velocity;
        const double distance = relative_position.norm();
        const double relative_speed = relative_velocity.norm();

        try {
            const HernandezPairMapResult result = apply_hernandez_pair_kepler_map(bodies, pair.i, pair.j, dt, G);
            if (result.converged) {
                return;
            }

            std::ostringstream msg;
            msg << std::setprecision(17)
                << "Hernandez pair Kepler map failed. "
                << "converged = false, "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_pair_i = " << pair.i << ", "
                << "failed_pair_j = " << pair.j << ", "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "failed_distance = " << distance << ", "
                << "failed_relative_speed = " << relative_speed << ", "
                << "failed_iterations = " << result.iterations << ", "
                << "reason = universal-variable solve did not converge";
            throw std::runtime_error(msg.str());
        } catch (const std::exception& exc) {
            std::ostringstream msg;
            msg << std::setprecision(17)
                << "Hernandez pair Kepler map failed. "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_pair_i = " << pair.i << ", "
                << "failed_pair_j = " << pair.j << ", "
                << "failed_dt = " << dt << ", "
                << "G = " << G << ", "
                << "failed_distance = " << distance << ", "
                << "failed_relative_speed = " << relative_speed << ", "
                << "reason = " << exc.what();
            throw std::runtime_error(msg.str());
        }
    }

    void drift_all_bodies(std::vector<Body>& bodies, double drift_dt) {
        for (Body& body : bodies) {
            body.position += drift_dt * body.velocity;
            body.updateMomentumFromVelocity();
        }
    }

    void drift_pair(std::vector<Body>& bodies, const Pair& pair, double drift_dt) {
        bodies[pair.i].position += drift_dt * bodies[pair.i].velocity;
        bodies[pair.j].position += drift_dt * bodies[pair.j].velocity;
        bodies[pair.i].updateMomentumFromVelocity();
        bodies[pair.j].updateMomentumFromVelocity();
    }

    void apply_phi(std::vector<Body>& bodies, const std::vector<Pair>& ordered_pairs, double h, double G) {
        drift_all_bodies(bodies, h);
        for (const Pair& pair : ordered_pairs) {
            drift_pair(bodies, pair, -h);
            apply_checked_pair_map(bodies, pair, h, G);
        }
    }

    void apply_phi_adjoint(std::vector<Body>& bodies, const std::vector<Pair>& ordered_pairs, double h, double G) {
        for (auto it = ordered_pairs.rbegin(); it != ordered_pairs.rend(); ++it) {
            apply_checked_pair_map(bodies, *it, h, G);
            drift_pair(bodies, *it, -h);
        }
        drift_all_bodies(bodies, h);
    }
}

HernandezBodyStepper::HernandezBodyStepper(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs_preserve_order(fixed_pairs)) {}

void HernandezBodyStepper::step(std::vector<Body>& bodies, double dt, double G) {
    if (bodies.size() <= 1 || pairs_.empty()) {
        return;
    }
    if (pairs_.empty()) {
        drift_all_bodies(bodies, dt);
        return;
    }
    const double half_dt = 0.5 * dt;
    apply_phi(bodies, pairs_, half_dt, G);
    apply_phi_adjoint(bodies, pairs_, half_dt, G);
}

const std::vector<Pair>& HernandezBodyStepper::pairs() const {
    return pairs_;
}