#pragma once

#include <vector>

#include "core/body.h"
#include "dynamics/pairing.h"
#include "math/vec3.h"

struct TwoBodyState {
    Pair pair;
    double first_mass = 0.0;
    double second_mass = 0.0;
    double total_mass = 0.0;
    double reduced_mass = 0.0;
    Vec3 COM_position;
    Vec3 COM_veklocity;
    Vec3 relative_position;
    Vec3 relative_momentum;

    [[nodiscard]] static TwoBodyState from_body_pair(const std::vector<Body>& bodies, const Pair& body_pair);
    [[nodiscard]] double relative_gravitational_parameter(double gravitational_constant) const;
    [[nodiscard]] double relative_orbital_energy(double gravitational_constant) const;
    [[nodiscard]] Vec3 relative_angular_momentum() const;
    [[nodiscard]] Vec3 total_linear_momentum() const;

    void write_back_to_bodies(std::vector<Body>& bodies) const;
};