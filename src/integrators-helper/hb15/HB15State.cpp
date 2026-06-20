#include <stdexcept>

#include "integrators-helper/hb15/state.h"

HB15State HB15State::from_bodies(const std::vector<Body>& bodies) {
    HB15State state;
    
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

void HB15State::write_to_bodies(std::vector<Body>& bodies) const {
    validate();

    if (bodies.size() != size()) {
        throw std::runtime_error("HB15State::write_to_bodies size mismatch.");
    }

    for (std::size_t i = 0; i < size(); ++i) {
        bodies[i].mass = masses[i];
        bodies[i].position = positions[i];
        bodies[i].momentum = momenta[i];
        bodies[i].velocity = momenta[i] / masses[i];
    }
}

std::vector<Body> HB15State::to_bodies() const {
    validate();

    std::vector<Body> bodies;
    bodies.reserve(size());

    for (std::size_t i = 0; i < size(); ++i) {
        const Vec3 velocity = momenta[i] / masses[i];
        bodies.emplace_back(masses[i], positions[i], velocity);
    }
    return bodies;
}

std::size_t HB15State::size() const {
    return masses.size();
}

bool HB15State::empty() const {
    return size() == 0;
}

void HB15State::validate() const {
    if (masses.size() != positions.size() || masses.size() != momenta.size()) {
        throw std::runtime_error("HB15State has inconsistent vector size.");
    }
    for (double mass : masses) {
        if (mass <= 0.0) {
            throw std::runtime_error("HB15State contains a non-positive mass.");
        }
    }
}

double HB15State::total_mass() const {
    validate();

    double total = 0.0;

    for (double mass : masses) {
        total += mass;
    }
    return total;
}

Vec3 HB15State::total_momenta() const {
    validate();

    Vec3 total;

    for (const Vec3& momentum : momenta) {
        total += momentum;
    }
    return total;
}

Vec3 HB15State::com_positions() const {
    validate();

    const double total = total_mass();
    Vec3 center;
    for(std::size_t i = 0; i < size(); ++i) {
        center += masses[i] * positions[i];
    }
    return center / total;
}

Vec3 HB15State::com_velocity() const {
    validate();
    return total_momenta() / total_mass();
}