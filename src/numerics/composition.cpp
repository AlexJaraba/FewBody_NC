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

void SymmetricComposition::execute(std::vector<Body>& bodies, const std::vector<Pair>& pairs, double dt, double G) const {
    if (has_corrector_) {
        corrector_.apply_forward(bodies, pairs, dt, G);
    }

    for (const auto& step : steps_) {
        const double h = step.coefficient * dt;

        switch (step.type) {
            case OperatorType::DRIFT:
                drift_operator(bodies, h);
                break;
            case OperatorType::KICK:
                kick_operator(bodies, pairs, h, G);
                for (auto& body : bodies){
                    body.updateVelocityFromMomentum();
                }
                break;
            case OperatorType::KEPLER:
                symmetric_kepler_operator(bodies, pairs, h, G);
                for (auto& body: bodies){
                    body.updateMomentumFromVelocity();
                }
                break;
            default:
                throw std::runtime_error("Unknown operator type in composition.");
        }
    }

    if (has_corrector_) {
        corrector_.apply_backward(bodies, pairs, dt, G);
    }
}