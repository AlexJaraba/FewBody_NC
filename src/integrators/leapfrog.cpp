#include "integrators/leapfrog.h"

void Leapfrog::step(std::vector<Body>& bodies, double dt, double G) {
    for (auto& body : bodies) {
        body.updateAcceleration(bodies, G);
    }
    for (auto& body : bodies) {
        body.velocity += 0.5 * dt * body.acceleration;
        body.position += dt * body.velocity;
    }
    for (auto& body : bodies) {
        body.updateAcceleration(bodies, G);
    }
    for (auto& body : bodies) {
        body.velocity += 0.5 * dt * body.acceleration;
        body.updateMomentumFromVelocity();
    }
}