#pragma once

#include <vector>
#include <string>
#include <memory>

#include "core/body.h"
#include "io/csv_output_writer.h"
#include "io/io.h"
#include "dynamics/pairing.h"
#include "integrators/integrator.h"


class CSVOutputWriter;

void recenter_system(std::vector<Body>& bodies);

class Solver {
public:
    Solver(std::vector<Body>& bodies, CSVOutputWriter& writer);
    void run();

private:
    std::vector<Body>& bodies;
    std::unique_ptr<Integrator> integrator;
    std::vector<Pair> fixed_pairs;
    CSVOutputWriter& writer;
    SolverParams params_;
    
    std::string effective_pair_order = "canonical";

    void run_fixed_step(const SolverParams& params);
    void write_current_bodies(double time);
};
