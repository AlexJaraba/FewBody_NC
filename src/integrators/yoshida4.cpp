#include "yoshida4.h"
#include "hernandez.h"

Yoshida4::Yoshida4(const std::vector<Pair>& pairs) : pairs_(pairs) {}

void Yoshida4::step(std::vector<Body>& bodies, double dt) {
    Hernandez hernandez(pairs_);
    hernandez.step(bodies, w1_ * dt);
    hernandez.step(bodies, w0_ * dt);
    hernandez.step(bodies, w1_ * dt);
}