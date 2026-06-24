#pragma once

#include <cstddef>
#include <vector>

#include "core/body.h"
#include "math/vec3.h"

/* ============================================================================

    HernandezState stores the full Cartesian canonical state for the Hernandez path.

   ============================================================================ */

struct HernandezState {
    std::vector<double> masses;
    std::vector<Vec3> positions;
    std::vector<Vec3> momenta;
    static HernandezState from_bodies(const std::vector<Body>& bodies);
    void write_to_bodies(std::vector<Body>& bodies) const;
    std::vector<Body> to_bodies() const;
    std::size_t size() const;
    bool empty() const;
    void validate() const;
    double total_mass() const;
    Vec3 total_momentum() const;
    Vec3 com_positions() const;
    Vec3 com_velocity() const;
};