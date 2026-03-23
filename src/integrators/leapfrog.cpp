#include "integrator.h"
#include "body.h"

void Leapfrog::step(std::vector<Body>& bodies, double dt) {
    // Initialize half-step velocities for the leapfrog integrator
    for (auto& body : bodies) {
        body.updateAcceleration(bodies);
        for (int i = 0; i < 3; ++i) {
            body.velocity[i] += 0.5 * body.acceleration[i] * dt;
        }
    }

    // Update positions
    for (auto& body : bodies)
        body.updatePosition(dt);

    // Recompute accelerations
    for (auto& body : bodies)
        body.updateAcceleration(bodies);

    // Update velocities
    //for (auto& body : bodies)
    //    body.updateVelocity(dt);

    // Finalize velocities after the last step
    for (auto& body : bodies) {
        for (int i = 0; i < 3; ++i) {
            body.velocity[i] += 0.5 * body.acceleration[i] * dt;
        }
    }
}
