#pragma once

#include <memory>

#include "math/vec3.h"

class ExternalPotential {
public:
    virtual ~ExternalPotential() = default;
    // Returns the potential energy at a given position for a mass.
    virtual double computePotentialEnergy(const Vec3& position, double mass) const = 0;
    // Returns the acceleration at a given position due to this potential.
    virtual Vec3 accelerationAt(const Vec3& position, double G) const = 0;
};