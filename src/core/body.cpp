#include <cmath>

#include "core/body.h"

Body::Body(double m, Vec3 pos, Vec3 vel)
    : mass(m), position(pos), velocity(vel), acceleration(), momentum(m * vel) {}

void Body::updateAcceleration(const std::vector<Body>& bodies, double G) {
    acceleration = Vec3();
    
    for (const auto& other : bodies) {
        if (&other == this) continue;

        Vec3 dr = other.position - position;
        double r2 = dr.norm2();
        double r = dr.norm();

        if (r < 1e-10) {
            continue;
        }

        double inv_r3 = 1.0 / (r * r2);
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
}

BodyState Body::toState(double time) const {
    return BodyState{
        time,
        {position.x, position.y, position.z},
        {velocity.x, velocity.y, velocity.z},
        mass
    };
}

double Body::kineticEnergy() const {
    return 0.5 * mass * velocity.norm2();
}