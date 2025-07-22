#include "integrator.h"

void Hernandez::step(std::vector<Body>& bodies, double dt) {
    for (auto& Body : bodies) {
        // Drift bodies forward by dt/2
        // Use Propagate on Bodies
        // Move Center of Mass according to position and velocity
        // Drift bodies backward by dt/2
    }
    // Return values for position and velocity
}

//Drift should be body.updatePosition(dt/2) and body.updateVelocity(dt/2) for each body