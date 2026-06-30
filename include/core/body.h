#pragma once

#include <vector>

#include "core/body_state.h"
#include "math/vec3.h"

struct Body {
    double mass;
    double radius;
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    Vec3 momentum;

    BodyState toState(double time) const;

    Body(double m, Vec3 pos, Vec3 vel, double r = 0.0);

    void updateAcceleration(const std::vector<Body>& bodies, double G);
    void updatePosition(double dt);
    void updateVelocity(double dt);
    void updateVelocityFromMomentum();
    void updateMomentumFromVelocity();
    double kineticEnergy() const;
};
