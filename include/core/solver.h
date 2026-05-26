#pragma once

#include <vector>
#include <memory>

#include "core/body.h"
#include "io/csv_output_writer.h"
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

private:
    std::vector<Body>& bodies;
    std::unique_ptr<Integrator> integrator;
    CSVOutputWriter& writer;
};
