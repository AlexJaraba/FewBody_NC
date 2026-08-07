#pragma once

#include <vector>
#include <cstdint>

#include "core/body.h"
#include "io/output_writer.h"
#include "io/io.h"
#include "integrators/integrator.h"

void move_to_COM_frame(std::vector<Body>& bodies);

class Solver {
public:
    Solver(std::vector<Body>& bodies, OutputWriter& output_writer);
    void run(const SolverParams& params);

private:
    std::vector<Body>& bodies_;
    OutputWriter& output_writer_;

    void run_fixed_step_integration(Integrator& integrator, const SolverParams& params);
    void validate_body_states_are_finite(std::uint64_t step, double time, const char* context) const;
    void write_body_snapshot(double time);
};
