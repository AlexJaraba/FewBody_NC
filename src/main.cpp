#include <iostream>
#include <vector>
#include <string>

#include "body.h"
#include "solver.h"
#include "io.h"
#include "globals.h"
#include "csv_output_writer.h"
#include "hernandez.h"
#include "yoshida4.h"
#include "jacobi.h"
#include "solver.h"
#include "operators.h"

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
