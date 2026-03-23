#include "integrator.h"
#include "body.h"

void Hernandez::step(std::vector<Body>& bodies, double dt) {
    // Step 1: Update all accelerations (global)
    for (auto& body : bodies) {
        body.updateAcceleration(bodies);  // full pairwise update
    }

    // Step 2: Kick - half velocity step
    for (auto& body : bodies) {
        body.updateVelocity(+0.5 * dt);
    }

    // Step 3: Drift - full position step
    for (auto& body : bodies) {
        body.updatePosition(dt);
    }


    // Custom Propagator


    // Step 4: Recompute acceleration at new positions
    for (auto& body : bodies) {
        body.updateAcceleration(bodies);
    }

    // Step 5: Kick - remaining half velocity step
    for (auto& body : bodies) {
        body.updateVelocity(+0.5 * dt);
    }
}
    // Return values for position and velocity