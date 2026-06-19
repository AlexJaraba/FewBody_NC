#include <stdexcept>

#include "integrators/hb15_pair_state.h"

namespace {
    void validate_pair_indices(const std::vector<Body>& bodies, int i, int j) {
        const int N = static_cast<int>(bodies.size());

        if (i < 0 || j < 0) {
            throw std::runtime_error("HB15PairState received a negative body index.");
        }
        if (i >= N || j >= N) {
            throw std::runtime_error("HB15PairState body index is out of range.");
        }
        if (i == j) {
            throw std::runtime_error("HB15PairState requires two distinct bodies.");
        }
        if (bodies[i].mass <= 0.0 || bodies[j].mass <= 0.0) {
            throw std::runtime_error("HB15PairState requires positive body masses.");
        }
    }
}

HB15PairState HB15PairState::from_bodies(const std::vector<Body>& bodies, int i, int j) {
    validate_pair_indices(bodies, i, j);

    HB15PairState pair;

    pair.i = i;
    pair.j = j;
    pair.mass_i = bodies[i].mass;
    pair.mass_j = bodies[j].mass;
    pair.total_mass = pair.mass_i + pair.mass_j;
    pair.reduced_mass = (pair.mass_i * pair.mass_j) / pair.total_mass;
    pair.com_position = ((pair.mass_i * bodies[i].position) + (pair.mass_j * bodies[j].position)) / pair.total_mass;
    pair.com_velocity = ((pair.mass_i * bodies[i].velocity) + (pair.mass_j * bodies[j].velocity)) / pair.total_mass;
    pair.relative_position = bodies[i].position - bodies[j].position;
    pair.relative_velocity = bodies[i].velocity - bodies[j].velocity;

    return pair;
}

void HB15PairState::write_to_bodies(std::vector<Body>& bodies) const {
    validate_pair_indices(bodies, i, j);

    const double coeff_i = mass_j / total_mass;
    const double coeff_j = mass_i / total_mass;

    bodies[i].position = com_position + coeff_i * relative_position;
    bodies[j].position = com_position - coeff_j * relative_position;
    bodies[i].velocity = com_velocity + coeff_i * relative_velocity;
    bodies[j].velocity = com_velocity - coeff_j * relative_velocity;

    bodies[i].updateMomentumFromVelocity();
    bodies[j].updateMomentumFromVelocity();
}

double HB15PairState::gravitational_parameter(double G) const {
    return G * total_mass;
}
double HB15PairState::two_body_energy(double G) const {
    const double r = relative_position.norm();

    if (r <= 0.0) {
        throw std::runtime_error("HB15PairState two-body energy has zero separation.");
    }

    const double kinetic = 0.5 * reduced_mass * relative_velocity.norm2();
    const double potential = (-G * mass_i * mass_j)/ r;

    return kinetic + potential;
}

Vec3 HB15PairState::two_body_angular_momentum() const {
    return reduced_mass * cross(relative_position, relative_velocity);
}
Vec3 HB15PairState::total_momentum() const {
    return total_mass * com_velocity;
}