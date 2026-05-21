#include <stdexcept>

#include "composition.h"
#include "operators.h"

SymmetricComposition::SymmetricComposition(const std::vector<CompositionStep>& steps)
    : steps_(steps) {}

void SymmetricComposition::set_corrector(const SymplecticCorrector& corrector) {
    corrector_ = corrector;
    has_corrector_ = true;
}

const std::vector<CompositionStep>& SymmetricComposition::steps() const {
    return steps_;
}

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