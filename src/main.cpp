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
    std::vector<Body> bodies;
    readInitialConditions("data/initial_conditions.txt", bodies);

    recenter_system(bodies);

    std::string outputfilename = "output.csv";
    CSVOutputWriter output(outputfilename);
    
    Solver sim(bodies, output);
    sim.run();
    sim.ReversibilityTest();

    output.close();
    
    return 0;
}
