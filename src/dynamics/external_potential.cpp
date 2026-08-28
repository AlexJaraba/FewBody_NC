#include <cmath>

#include "dynamics/external_potential.h"

namespace potential {
    ExternalPotential::ExternalPotential(double mass, Vec3 b) : mass_(mass), b_(b) {}

    Vec3 ExternalPotential::accelerationAt(const Vec3& position, double G) const {
        const double b2 = b_.norm2();
        const double r2 = position.norm2();
        const double denominator = std::pow(r2 + b2, 3.0);
        const double coeff = -G * mass_ / denominator;
        const Vec3 external_acceleration = coeff * position;
        
        return external_acceleration;
    }
}