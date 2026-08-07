#pragma once

#include <vector>

#include "integrators/integrator.h"
#include "dynamics/pairing.h"


class HernandezIntegrator final : public Integrator {
public:
    explicit HernandezIntegrator(const std::vector<Pair>& fixed_pairs);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nosdiscard]] std::vector<Pair>& fixed_pair_order() const noexcept;
    void step(std::vector<Body>& bodies, double timestep, double gravitational_constant) override;

private:
    std::vector<Pair> fixed_pairs_;
    void apply_phi(std::vector<Body>& bodies, double timestep, double gravitational_constant) const;
    void apply_phi_adjoint(std::vector<Body>& bodies, double timestep, double gravitational_constant) const;
};