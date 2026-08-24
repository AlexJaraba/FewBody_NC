#pragma once

#include <vector>

#include "core/body.h"
#include "math/vec3.h"

/* ==========================================================

    HernandezPairState stores the physical two-body variables for one body pair.

    It converts the physical bodies into:
        center-of-mass position and velocity
        relative position and velocity
        total mass
        reduced mass

    This class only handles the exact coordinate conversion and reconstruction.

   ==========================================================*/

struct HernandezPairState {
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

    static HernandezPairState pairState(const std::vector<Body>& bodies, int i, int j);

    void writeToBodies(std::vector<Body>& bodies) const;

    double gravitationalParameter(double G) const;
    double twoBodyEnergy(double G) const;

    Vec3 twoBodyAngularMomentum() const;
    Vec3 totalMomentum() const;
};