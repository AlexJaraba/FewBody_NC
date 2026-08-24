#include <stdexcept>

#include "integrators-helper/hernandez/pair_state.h"

namespace {
    void validatePairIndices(const std::vector<Body>& bodies, int i, int j) {
        const int N = static_cast<int>(bodies.size());

        if (i < 0 || j < 0) {
            throw std::runtime_error("HernandezPairState received a negative body index.");
        }
        if (i >= N || j >= N) {
            throw std::runtime_error("HernandezPairState body index is out of range.");
        }
        if (i == j) {
            throw std::runtime_error("HernandezPairState requires two distinct bodies.");
        }
        const std::size_t index_i = static_cast<std::size_t>(i);
        const std::size_t index_j = static_cast<std::size_t>(j);
        if (bodies[index_i].mass <= 0.0 || bodies[index_j].mass <= 0.0) {
            throw std::runtime_error("HernandezPairState requires positive body masses.");
        }
    }
}

HernandezPairState HernandezPairState::pairState(const std::vector<Body>& bodies, int i, int j) {
    validatePairIndices(bodies, i, j);

    HernandezPairState pair;
    const std::size_t index_i = static_cast<std::size_t>(i);
    const std::size_t index_j = static_cast<std::size_t>(j);

    pair.i = i;
    pair.j = j;
    pair.mass_i = bodies[index_i].mass;
    pair.mass_j = bodies[index_j].mass;
    pair.total_mass = pair.mass_i + pair.mass_j;
    pair.reduced_mass = (pair.mass_i * pair.mass_j) / pair.total_mass;
    pair.com_position = ((pair.mass_i * bodies[index_i].position) + (pair.mass_j * bodies[index_j].position)) / pair.total_mass;
    pair.com_velocity = ((pair.mass_i * bodies[index_i].velocity) + (pair.mass_j * bodies[index_j].velocity)) / pair.total_mass;
    pair.relative_position = bodies[index_i].position - bodies[index_j].position;
    pair.relative_velocity = bodies[index_i].velocity - bodies[index_j].velocity;

    return pair;
}

void HernandezPairState::writeToBodies(std::vector<Body>& bodies) const {
    validatePairIndices(bodies, i, j);

    const std::size_t index_i = static_cast<std::size_t>(i);
    const std::size_t index_j = static_cast<std::size_t>(j);
    const double coeff_i = mass_j / total_mass;
    const double coeff_j = mass_i / total_mass;

    bodies[index_i].position = com_position + coeff_i * relative_position;
    bodies[index_j].position = com_position - coeff_j * relative_position;
    bodies[index_i].velocity = com_velocity + coeff_i * relative_velocity;
    bodies[index_j].velocity = com_velocity - coeff_j * relative_velocity;

    bodies[index_i].updateMomentumFromVelocity();
    bodies[index_j].updateMomentumFromVelocity();
}

double HernandezPairState::gravitationalParameter(double G) const {
    return G * total_mass;
}
double HernandezPairState::twoBodyEnergy(double G) const {
    const double r = relative_position.norm();

    if (r <= 0.0) {
        throw std::runtime_error("HernandezPairState two-body energy has zero separation.");
    }

    const double kinetic = 0.5 * reduced_mass * relative_velocity.norm2();
    const double potential = (-G * mass_i * mass_j) / r;

    return kinetic + potential;
}

Vec3 HernandezPairState::twoBodyAngularMomentum() const {
    return reduced_mass * cross(relative_position, relative_velocity);
}
Vec3 HernandezPairState::totalMomentum() const {
    return total_mass * com_velocity;
}