#include <iostream>
#include <vector>
#include <string>

#include "core/body.h"
#include "core/solver.h"
#include "io/io.h"
#include "io/csv_output_writer.h"
#include "integrators/hernandez.h"
#include "integrators/yoshida4.h"
#include "dynamics/jacobi.h"
#include "dynamics/operators.h"

int main() {
    // Read initial conditions and create bodies
    std::vector<Body> bodies;
    readInitialConditions("data/initial_conditions.txt", bodies);

    // Recenter system to the center of mass frame
    recenter_system(bodies);

    // Create output writer
    std::string outputfilename = "output.csv";
    CSVOutputWriter output(outputfilename);
    
    // Create solver and run simulation
    Solver sim(bodies, output);

    //sim.run();

    // Run Optional Tests for Validation (example: sim.tests.[]())
    sim.tests.TestHB15AdaptiveBlockValidation();

    // Close output file
    output.close();
    
    return 0;
}
