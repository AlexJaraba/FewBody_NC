#include <cmath>
#include <iostream>

#include "body.h"
#include "globals.h"
#include "body_state.h"


Body::Body(double m, std::vector<double> pos, std::vector<double> vel)
    : mass(m), position(pos), velocity(vel), acceleration(3, 0.0) {}

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

void Body::updateCenterofMass(const std::vector<Body>& bodies) {
    double total_mass = 0.0;
    std::vector<double> center_of_mass(3, 0.0);
    
    for (const auto& body : bodies) {
        total_mass += body.mass;
        for (int i = 0; i < 3; ++i)
            center_of_mass[i] += body.position[i] * body.mass;
    }
    
    for (int i = 0; i < 3; ++i)
        center_of_mass[i] /= total_mass;
}

void Body::updatePosition(double dt) {
  for (int i = 0; i < 3; ++i)
    position[i] += velocity[i] * dt;
}

void Body::updateVelocity(double dt) {
    for (int i = 0; i < 3; ++i)
        velocity[i] += acceleration[i] * dt;
}

BodyState Body::toState(double time) const {
    return BodyState{
        time,
        {position[0], position[1], position[2]},
        {velocity[0], velocity[1], velocity[2]},
        mass
    };
}

