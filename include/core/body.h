#pragma once

#include "core/body_state.h"
#include "math/vec3.h"

struct Body {
    int id = -1;
    double mass = 0.0;
    double radius = 0.0;
    Vec3 position;
    Vec3 velocity;

    Body(double mass, Vec3 position, Vec3 velocity, double radius = 0.0);
    Body(int id_, double mass, Vec3 position, Vec3 velocity, double radius = 0.0);

    [[nodiscard]] BodyState to_state(double time) const;
    [[nodiscard]] Vec3 linear_momentum() const noexcept;
    [[nodiscard]] double kineticEnergy() const noexcept;
};
