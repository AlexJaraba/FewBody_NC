#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "integrators-helper/hernandez/body_stepper.h"
#include "integrators-helper/hernandez/pair_map.h"
#include "integrators-helper/hernandez/pair_state.h"
#include "dynamics/pairing.h"
#include "dynamics/external_potential.h"
#include "core/body.h"

namespace {
    constexpr double PAIR_MAP_NEAR_ZERO_DISTANCE = 1e-14;

    void validatePair(const std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        if (pair.i < 0 || pair.j < 0 || pair.i >= static_cast<int>(bodies.size()) || pair.j >= static_cast<int>(bodies.size()) || pair.i == pair.j) {
            throw std::runtime_error("Hernandez pair map received invalid body indices.");
        }
        const std::size_t i = static_cast<std::size_t>(pair.i);
        const std::size_t j = static_cast<std::size_t>(pair.j);
        const Body& body_i = bodies[i];
        const Body& body_j = bodies[j];
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
        const double collision_distance = body_i.radius + body_j.radius;
        if (collision_distance > 0.0 && distance <= collision_distance) {
            std::ostringstream msg;
            msg << std::setprecision(17);
            msg << "Hernandez physical collision detected. "
                << "collision_detected = true, "
                << "pair=(" << pair.i << "," << pair.j << "), "
                << "failed_dt = " << dt << ", "
                << "distance = " << distance << ", "
                << "collision_distance = " << collision_distance << ", "
                << "radius_i = " << body_i.radius << ", "
                << "radius_j = " << body_j.radius;
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

    void keplerSolver(std::vector<Body>& bodies, const Pair& pair, double dt, double G) {
        validatePair(bodies, pair, dt, G);
        const std::size_t i = static_cast<std::size_t>(pair.i);
        const std::size_t j = static_cast<std::size_t>(pair.j);
        const Vec3 relative_position = bodies[j].position - bodies[i].position;
        const Vec3 relative_velocity = bodies[j].velocity - bodies[i].velocity;
        const double distance = relative_position.norm();
        const double relative_speed = relative_velocity.norm();

        try {
            const HernandezPairMapResult result = propagatePairKepler(bodies, pair.i, pair.j, dt, G);
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

    void externalPotentialKick(std::vector<Body>& bodies, const potential::ExternalPotential& potential, double G, double dt) {
        for (auto& body : bodies) {
            const Vec3 acceleration = potential.accelerationAt(body.position, G);
            body.velocity += dt * acceleration;
            body.updateMomentumFromVelocity();
        }
    }

    void driftAllBodies(std::vector<Body>& bodies, double drift_dt) {
        for (Body& body : bodies) {
            body.position += drift_dt * body.velocity;
            body.updateMomentumFromVelocity();
        }
    }

    void driftPair(std::vector<Body>& bodies, const Pair& pair, double drift_dt) {
        const std::size_t i = static_cast<std::size_t>(pair.i);
        const std::size_t j = static_cast<std::size_t>(pair.j);
        bodies[i].position += drift_dt * bodies[i].velocity;
        bodies[j].position += drift_dt * bodies[j].velocity;
        bodies[i].updateMomentumFromVelocity();
        bodies[j].updateMomentumFromVelocity();
    }

    void driftCOM(std::vector<Body>& bodies, const Pair& pair, double drift_dt) {
        HernandezPairState pair_state = HernandezPairState::pairState(bodies, pair.i, pair.j);
        pair_state.com_position += drift_dt * pair_state.com_velocity;
        pair_state.writeToBodies(bodies);
    }

        // pair.relative_position = propagated.q;
        // pair.relative_velocity = propagated.p / pair.reduced_mass;
        // pair.com_position += dt * pair.com_velocity;
        // pair.writeToBodies(bodies);
} // namespace hernandez

HernandezBodyStepper::HernandezBodyStepper(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalizePairsPreserveOrder(fixed_pairs)) {}

void HernandezBodyStepper::step(std::vector<Body>& bodies, double dt, double G) {
    if (bodies.size() <= 1 || pairs_.empty()) {
        return;
    }
    if (pairs_.empty()) {
        driftAllBodies(bodies, dt);
        return;
    }
    const double half_dt = 0.5 * dt;
    const double potential_mass = 10.0; // Replace with the actual mass value as needed
    const Vec3 scale_length(1.0, 2.0, 3.0); // Replace with the actual vector as needed
    potential::ExternalPotential external_potential(potential_mass, scale_length);
    double total_mass = 0.0;
    for (const auto& body : bodies) {
        total_mass += body.mass;
    }

    externalPotentialKick(bodies, external_potential, G, half_dt);
    driftAllBodies(bodies, half_dt);
    for (const Pair& pair : pairs_) {
        driftPair(bodies, pair, -half_dt);
        keplerSolver(bodies, pair, half_dt, G);
        driftCOM(bodies, pair, half_dt);
    }
    for (auto it = pairs_.rbegin(); it != pairs_.rend(); ++it) {
        driftCOM(bodies, *it, half_dt);
        keplerSolver(bodies, *it, half_dt, G);
        driftPair(bodies, *it, -half_dt);
    }
    driftAllBodies(bodies, half_dt);
    externalPotentialKick(bodies, external_potential, G, half_dt);
}

const std::vector<Pair>& HernandezBodyStepper::pairs() const {
    return pairs_;
}