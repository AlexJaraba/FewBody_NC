#pragma once

#include <vector>
#include <memory>
#include <string>

#include "core/body.h"
#include "io/csv_output_writer.h"
#include "io/io.h"
#include "integrators/integrator.h"
#include "integrators/leapfrog.h"
#include "integrators/hernandez.h"
#include "dynamics/pairing.h"


class CSVOutputWriter;

void recenterSystem(std::vector<Body>& bodies);

class Solver {
public:
    Solver(std::vector<Body>& bodies, CSVOutputWriter& writer);
    void run();

private:
    std::vector<Body>& bodies;
    std::vector<Pair> fixed_pairs;
    std::unique_ptr<Integrator> integrator;
    CSVOutputWriter& writer;
    
    std::string effective_pair_order = "canonical";
    double hierarchy_ratio = 0.0;

<<<<<<< Updated upstream
    void run_jacobi(const SolverParams& params);
    void run_cartesian(const SolverParams& params);
    void cartesian_step(double dt, double G);
    void write_current_bodies(double time);
=======
    void runFixedStep(const SolverParams& params);
    void writeCurrentBodies(double time);
>>>>>>> Stashed changes
};
