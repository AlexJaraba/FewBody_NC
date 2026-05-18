#include <iostream>
#include <vector>
#include <string>

#include "body.h"
#include "solver.h"
#include "io.h"
#include "globals.h"
#include "csv_output_writer.h"

int main() {
    std::vector<Body> bodies;
    readInitialConditions("data/initial_conditions.txt", bodies);

    std::string outputfilename = "output.csv";
    CSVOutputWriter output(outputfilename);
    
    Solver sim(bodies, output);
    sim.run();
    sim.ReversibilityTest();

    output.close();
    
    return 0;
}
