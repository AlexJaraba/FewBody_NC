#include <cmath>

#include "body.h"
#include "globals.h"
#include "body_state.h"


Body::Body(double m, std::vector<double> pos, std::vector<double> vel)
    : mass(m), position(pos), velocity(vel), acceleration(3, 0.0) {
        momentum.resize(3);
        for (int i = 0; i < 3; ++i)
            momentum[i] = mass * velocity[i];
    }

void Body::updateAcceleration(const std::vector<Body>& bodies) {
    acceleration = {0.0, 0.0, 0.0};
    
    for (const auto& other : bodies) {
        if (&other == this) continue;
        
        double diff[3];
        for (int i = 0; i < 3; ++i)
            diff[i] = other.position[i] - position[i];
        
        double r2 = (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        double distance = std::sqrt(r2);
        if (distance < 1e-10) continue;  // Avoid division by zero

        double inv_r3 = 1.0 / (distance * r2);  // 1/r^3 for force calculation
        double force = G * other.mass * inv_r3;
        
        for (int i = 0; i < 3; ++i)
            acceleration[i] += force * diff[i];
    }
}

void Body::updatePosition(double dt) {
  for (int i = 0; i < 3; ++i)
    position[i] += velocity[i] * dt;
}

void Body::updateVelocity(double dt) {
    for (int i = 0; i < 3; ++i)
        velocity[i] += acceleration[i] * dt;
}

void Body::updateVelocityFromMomentum() {
    for (int k = 0; k < 3; ++k)
        velocity[k] = momentum[k] / mass;
}

void Body::updateMomentumFromVelocity() {
    for (int k = 0; k < 3; ++k)
        momentum[k] = mass * velocity[k];
}

BodyState Body::toState(double time) const {
    return BodyState{
        time,
        {position[0], position[1], position[2]},
        {velocity[0], velocity[1], velocity[2]},
        mass
    };
}

double Body::kineticEnergy() const {
    double v2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        v2 += velocity[k] * velocity[k];
    }
    return 0.5 * mass * v2;
}

double Body::momentumMagnitudeSquared() const {
    double p2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        p2 += momentum[k] * momentum[k];
    }
    return p2;
}
