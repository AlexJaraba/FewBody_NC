#include <stdexcept>

#include "integrators-helper/hernandez/state.h"

HernandezState HernandezState::from_bodies(const std::vector<Body>& bodies) {
    HernandezState state;
    
    state.masses.reserve(bodies.size());
    state.positions.reserve(bodies.size());
    state.momenta.reserve(bodies.size());

    for (const Body& body : bodies) {
        state.masses.push_back(body.mass);
        state.positions.push_back(body.position);
        state.momenta.push_back(body.mass * body.velocity);
    }
    state.validate();
    return state;
}

void HernandezState::write_to_bodies(std::vector<Body>& bodies) const {
    validate();

    if (bodies.size() != size()) {
        throw std::runtime_error("HernandezState::write_to_bodies size mismatch.");
    }

    for (std::size_t i = 0; i < size(); ++i) {
        bodies[i].mass = masses[i];
        bodies[i].position = positions[i];
        bodies[i].momentum = momenta[i];
        bodies[i].velocity = momenta[i] / masses[i];
    }
}

std::vector<Body> HernandezState::to_bodies() const {
    validate();

    std::vector<Body> bodies;
    bodies.reserve(size());

    for (std::size_t i = 0; i < size(); ++i) {
        const Vec3 velocity = momenta[i] / masses[i];
        bodies.emplace_back(masses[i], positions[i], velocity);
    }
    return bodies;
}

std::size_t HernandezState::size() const {
    return masses.size();
}

bool HernandezState::empty() const {
    return size() == 0;
}

void HernandezState::validate() const {
    if (masses.size() != positions.size() || masses.size() != momenta.size()) {
        throw std::runtime_error("HernandezState has inconsistent vector size.");
    }
    for (double mass : masses) {
        if (mass <= 0.0) {
            throw std::runtime_error("HernandezState contains a non-positive mass.");
        }
    }
}

double HernandezState::total_mass() const {
    validate();

    double total = 0.0;

    for (double mass : masses) {
        total += mass;
    }
    return total;
}

Vec3 HernandezState::total_momentum() const {
    validate();

    Vec3 total;

    for (const Vec3& momentum : momenta) {
        total += momentum;
    }
    return total;
}

Vec3 HernandezState::com_positions() const {
    validate();

    const double total = total_mass();
    Vec3 center;
    for(std::size_t i = 0; i < size(); ++i) {
        center += masses[i] * positions[i];
    }
    return center / total;
}

Vec3 HernandezState::com_velocity() const {
    validate();
    return total_momentum() / total_mass();
}