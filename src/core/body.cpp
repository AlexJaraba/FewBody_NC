#include <cmath>
#include <stdexcept>
#include <numbers>

#include "core/body.h"
#include "dynamics/external_potential.h"

namespace {
    void compensatedAdd(double& value, double& compensation, double increment) {
        const double corrected_increment = increment - compensation;
        const double updated_value = value + corrected_increment;
        compensation = (updated_value - value) - corrected_increment;
        value = updated_value;
    }
}

Body::Body(double m, Vec3 pos, Vec3 vel, double r)
    : Body(-1, m, pos, vel, r) {}

Body::Body(int id_, double m, Vec3 pos, Vec3 vel, double r)
    : id(id_), mass(m), radius(r), position(pos), velocity(vel), acceleration(), momentum(m * vel), momentum_compensation() {}

void Body::updateAcceleration(const std::vector<Body>& bodies, double G) {
    acceleration = Vec3();

    for (const auto& other : bodies) {
        if (&other == this) continue;

        Vec3 dr = other.position - position;
        const double r2 = dr.norm2();
        if (!std::isfinite(r2) || r2 <= 0.0) {
            throw std::runtime_error("updateAcceleration requires a finite, non-zero pair separation.");
        }
        const double r = dr.norm();
        const double inv_r3 = 1.0 / (r * r2);
        acceleration += G * other.mass * inv_r3 * dr;
    }
}

void Body::updatePosition(double dt) {
    position += velocity * dt;
}

void Body::updateVelocity(double dt) {
    velocity += acceleration * dt;
}

void Body::updateVelocityFromMomentum() {
    velocity = momentum / mass;
}

void Body::updateMomentumFromVelocity() {
    momentum = mass * velocity;
    momentum_compensation = Vec3();
}

void Body::addMomentumIncrement(const Vec3& delta) {
    compensatedAdd(momentum.x, momentum_compensation.x, delta.x);
    compensatedAdd(momentum.y, momentum_compensation.y, delta.y);
    compensatedAdd(momentum.z, momentum_compensation.z, delta.z);
}

BodyState Body::toState(double time) const {
    return BodyState{time, id, {position.x, position.y, position.z}, {velocity.x, velocity.y, velocity.z}, mass, radius};
}

double Body::kineticEnergy() const {
    return 0.5 * momentum.norm2() / mass;
}