#pragma once

#include "math/vec3.h"

namespace potential {
class ExternalPotential {
public:
    explicit ExternalPotential(double mass, double scale_length);

    [[nodiscard]] Vec3 accelerationAt(const Vec3& position, double G) const;
    [[nodiscard]] double potentialAt(const Vec3& position, double G) const;
private:
    const double mass_;
    const double scale_length_;
};
} // namespace potential