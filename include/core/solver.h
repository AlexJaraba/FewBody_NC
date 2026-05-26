#pragma once

#include <vector>
#include <memory>

#include "body.h"
#include "csv_output_writer.h"
#include "integrator.h"
#include "leapfrog.h"
#include "hernandez.h"
#include "yoshida4.h"
#include "pairing.h"

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
