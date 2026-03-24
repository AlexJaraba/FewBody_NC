#include "energy.h"

double compute_energy(const std::vector<Body>& bodies, double G)
{
    double kinetic = 0.0;
    double potential = 0.0;

    // Kinetic energy
    for (const auto& b : bodies) {
        kinetic += 0.5 * b.mass * b.velocity.norm2();
    }

    // Potential energy
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            Vec3 dr = bodies[j].position - bodies[i].position;
            double r = dr.norm();
            potential -= G * bodies[i].mass * bodies[j].mass / r;
        }
    }

    return kinetic + potential;
}