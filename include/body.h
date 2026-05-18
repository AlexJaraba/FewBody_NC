#pragma once

#include <vector>

#include "body_state.h"

struct Body {
    double mass;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> acceleration;
    std::vector<double> momentum;
    std::vector<double> jacobi_position;
    std::vector<double> jacobi_momentum;
    BodyState toState(double time) const;

    Body(double m, std::vector<double> pos, std::vector<double> vel);
    void updateAcceleration(const std::vector<Body>& bodies);
    void updatePosition(double dt);
    void updateVelocity(double dt);
    void updateCenterofMass(const std::vector<Body>& bodies);
    void updateVelocityFromMomentum();
    void updateMomentumFromVelocity();
    double kineticEnergy() const;
    double momentumMagnitudeSquared() const;
};
