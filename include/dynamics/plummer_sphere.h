#pragma once

#include "dynamics/extern_potentials.h"

class Plummer : public ExternalPotential {
    public:
        Plummer(double mass, double scale_length);
        [[nodiscard]] Vec3 accelerationAt(const Vec3& position, double G) const override;
        [[nodiscard]] double computePotentialEnergy(const Vec3& position, double mass) const override;
    private:
        const double mass_;
        const double scale_length_;
}; // class Plummer