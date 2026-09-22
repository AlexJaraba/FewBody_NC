#include "integrators/hernandez.h"

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : body_stepper_(fixed_pairs) {}

void Hernandez::step(std::vector<Body>& bodies, double dt, double G, const ExternalPotential* external_potential) {
    body_stepper_.step(bodies, dt, G, external_potential);
}

const std::vector<Pair>& Hernandez::pairs() const {
    return body_stepper_.pairs();
}