#pragma once

#include "math/vec3.h"

namespace potential {
class ExternalPotential {
public:
    explicit ExternalPotential(double mass, Vec3 b);

    [[nodiscard]] Vec3 accelerationAt(const Vec3& position, double G) const;
private:
    const double mass_;
    const Vec3 b_;
};
} // namespace potential