#include "body.h"
#include "globals.h"
#include <cmath>
#include <iostream>


Body::Body(double m, std::vector<double> pos, std::vector<double> vel)
    : mass(m), position(pos), velocity(vel), acceleration(3, 0.0) {}

void Body::updateAcceleration(const std::vector<Body>& bodies) {
    acceleration = {0.0, 0.0, 0.0};
    
    for (const auto& other : bodies) {
        if (&other == this) continue;
        
        std::vector<double> diff(3);
        for (int i = 0; i < 3; ++i)
            diff[i] = other.position[i] - position[i];
        
        double distance = std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        double force = G * other.mass / (distance * distance * distance);
        
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
  
  std::cout << position[0]<< std::endl;

}

void Body::updateVelocity(double dt) {
    for (int i = 0; i < 3; ++i)
        velocity[i] += acceleration[i] * dt;
}
