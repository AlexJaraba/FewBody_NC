#include "integrators/hernandez.h"

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : body_stepper_(fixed_pairs) {};

void Hernandez::step(std::vector<Body>& bodies, double dt, double G) {
    body_stepper_.step(bodies, dt, G);
}

const std::vector<Pair>& Hernandez::pairs() const {
    return body_stepper_.pairs();
}