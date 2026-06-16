#include <stdexcept>

#include "numerics/composition.h"
#include "dynamics/operators.h"

/* ===================================================================================

    Symmetric Composition Executor

    A composition is a sequence of primitive operators with coefficients.

     - Example
        KEPLER(0.5)
        KICK(1.0)
        KEPLER(0.5)
    
    This structure allows the Hernandez and Yoshida-style methods to be built from reusable drift/kick/Kepler operators.

   =================================================================================== */

SymmetricComposition::SymmetricComposition(const std::vector<CompositionStep>& steps)
    : steps_(steps) {}

void SymmetricComposition::set_corrector(const SymplecticCorrector& corrector) {
    corrector_ = corrector;
    has_corrector_ = true;
}

const std::vector<CompositionStep>& SymmetricComposition::steps() const {
    return steps_;
}

/*
    Execute each composition step in order.
    Correctors are optional and should only be enabled after validation for the current Hamiltonian split.
*/

void SymmetricComposition::execute(CanonicalState& state, const std::vector<Pair>& pairs, double dt, double G) const {
    if (has_corrector_) {
        corrector_.apply_forward(state, pairs, dt, G);
    }

    for (const auto& step : steps_) {
        const double h = step.coefficient * dt;

        switch (step.type) {
            case OperatorType::DRIFT:
                drift_operator(state, h);
                break;
            case OperatorType::KICK:
                kick_operator(state, pairs, h, G);
                break;
            case OperatorType::KEPLER:
                symmetric_kepler_operator(state, pairs, h, G);
                break;
            default:
                throw std::runtime_error("Unknown operator type in composition.");
        }
    }

    if (has_corrector_) {
        corrector_.apply_backward(state, pairs, dt, G);
    }
}