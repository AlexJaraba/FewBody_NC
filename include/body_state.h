#pragma once

#include <array>

struct BodyState {
    double time;
    std::array<double, 3> position;
    std::array<double, 3> velocity;
    double mass;
};