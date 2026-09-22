#pragma once

#include <memory>

#include "math/vec3.h"

class ExternalPotential {
public:
    virtual ~ExternalPotential() = default;
    virtual double computePotentialEnergy(const Vec3& position, double mass) const = 0;
    virtual Vec3 accelerationAt(const Vec3& position, double G) const = 0;
};