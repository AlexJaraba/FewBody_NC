#pragma once

#include <vector>
#include <memory>

#include "core/body.h"
#include "io/csv_output_writer.h"
#include "io/io.h"
#include "integrators/integrator.h"
#include "integrators/leapfrog.h"
#include "integrators/hernandez.h"
#include "integrators/yoshida4.h"
#include "dynamics/pairing.h"

class CSVOutputWriter;

void recenter_system(std::vector<Body>& bodies);

class Solver {
public:
    Solver(std::vector<Body>& bodies, CSVOutputWriter& writer);

    void run();
    void ReversibilityTest();
    void TestHernandezAdjoint(double dt);
    void TestLocalOrder();
    void TestHB15PairStateRoundTrip();
    void TestHB15PairKeplerMap();
    void TestHB15PairKeplerSuite();
    void TestHB15SymmetricOrdering();
    void TestHB15FixedStepValidation();

private:
    std::vector<Body>& bodies;
    std::vector<Pair> fixed_pairs;
    std::unique_ptr<Integrator> integrator;
    CSVOutputWriter& writer;

    void run_jacobi(const SolverParams& params);
    void run_cartesian(const SolverParams& params);

    void cartesian_step(double dt, double G);
    void write_current_bodies(double time);
};
