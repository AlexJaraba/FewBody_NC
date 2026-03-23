#pragma once

#include <vector>
#include <memory>

#include "body.h"
#include "integrator.h"

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
