#pragma once

#include <vector>

#include "core/body.h"
#include "math/vec3.h"

/* ==========================================================

    HB15PairState stores the physical two-body variables for one Cartesian pair.

    It converts the physical bodies into:
        center-of-mass position and velocity
        relative position and velocity
        total mass
        reduced mass

    This class only handles the exact coordinate conversion and reconstruction.

   ==========================================================*/

struct HB15PairState {
    int i = -1;
    int j = -1;
    
    double mass_i = 0.0;
    double mass_j = 0.0;
    double total_mass = 0.0;
    double reduced_mass = 0.0;

    Vec3 com_position;
    Vec3 com_velocity;
    Vec3 relative_position;
    Vec3 relative_velocity;

    static HB15PairState from_bodies(const std::vector<Body>& bodies, int i, int j);

    void write_to_bodies(std::vector<Body>& bodies) const;

    double gravitational_parameter(double G) const;
    double two_body_energy(double G) const;

    Vec3 two_body_angular_momentum() const;
    Vec3 total_momentum() const;
};