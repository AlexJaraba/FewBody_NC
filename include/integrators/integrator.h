#pragma once

#include <string_view>
#include <vector>

#include "core/body.h"

class Integrator {
public:
    virtual ~Integrator() = default;

    [[nodiscard]] virtual std::string_view name()  const noexcept = 0;

    virtual void step(std::vector<Body>& bodies, double timestep, double gravitational_constant) = 0;
    
};