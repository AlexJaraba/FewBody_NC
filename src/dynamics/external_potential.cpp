#include <cmath>

#include "dynamics/external_potential.h"

namespace potential {
    ExternalPotential::ExternalPotential(double mass, double scale_length) : mass_(mass), scale_length_(scale_length) {}

    Vec3 ExternalPotential::accelerationAt(const Vec3& position, double G) const {
        const double b2 = scale_length_ * scale_length_;
        const double r2 = position.norm2();
        const double softened_r2 = r2 + b2;
        const double denominator = softened_r2 * std::sqrt(softened_r2);
        const double coeff = -G * mass_ / denominator;
        const Vec3 external_acceleration = coeff * position;
        
        return external_acceleration;
    }

    double ExternalPotential::potentialAt(const Vec3& position, double G) const {
        const double b2 = scale_length_ * scale_length_;
        const double r2 = position.norm2();
        const double softened_r2 = r2 + b2;
        const double potential = -G * mass_ / std::sqrt(softened_r2);
        return potential;
    }
}