#include "integrators/leapfrog.h"

void Leapfrog::step(std::vector<Body>& bodies, double dt, double G, const ExternalPotential* external_potential) {
    for (auto& body : bodies) {
        body.updateAcceleration(bodies, G);
        if (external_potential != nullptr) {
            body.acceleration += external_potential->accelerationAt(body.position, G);
        }
    }
    for (auto& body : bodies) {
        body.addMomentumIncrement(0.5 * dt * body.mass * body.acceleration);
        body.updateVelocityFromMomentum();
        body.position += dt * (body.momentum / body.mass);
    }
    for (auto& body : bodies) {
        body.updateAcceleration(bodies, G);
        if (external_potential != nullptr) {
            body.acceleration += external_potential->accelerationAt(body.position, G);
        }
    }
    for (auto& body : bodies) {
        body.addMomentumIncrement(0.5 * dt * body.mass * body.acceleration);
        body.updateVelocityFromMomentum();
    }
}