#pragma once

#include "integrators/integrator.h"

class LeapfrogIntegrator final : public Integrator {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    void step(std::vector<Body>& bodies, double timestep, gravitational_constant) override;
};