#pragma once

#include <vector>

struct CanonicalState{
    std::vector<std::vector<double>> Q;
    std::vector<std::vector<double>> P;
    std::vector<double> com_position;
    std::vector<double> com_velocity;
    std::vector<double> mu;
    std::vector<double> M;
    std::vector<double> physical_mass;
};