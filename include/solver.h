#pragma once

#include <vector>
#include <memory>

#include "body.h"
#include "csv_output_writer.h"
#include "integrator.h"
#include "leapfrog.h"
#include "hernandez.h"

class CSVOutputWriter;

class Solver {
public:
    Solver(std::vector<Body>& bodies, CSVOutputWriter& writer);
    void run();

private:
    std::vector<Body>& bodies;
    std::unique_ptr<Integrator> integrator;
    CSVOutputWriter& writer;
};
