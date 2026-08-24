#include <iostream>
#include <vector>
#include <string>

#include "core/body.h"
#include "core/solver.h"
#include "io/io.h"
#include "io/csv_output_writer.h"

int main() {
    // Read initial conditions and create bodies
    std::vector<Body> bodies;
    readInitialConditions("data/initial_conditions.txt", bodies);

<<<<<<< Updated upstream
    // Recenter system to the center of mass frame
    recenter_system(bodies);
=======
        // Recenter system to the center of mass frame
        recenterSystem(bodies);
>>>>>>> Stashed changes

    // Create output writer
    std::string outputfilename = "output.csv";
    CSVOutputWriter output(outputfilename);
    
    // Create solver and run simulation
    Solver sim(bodies, output);

    sim.run();

    // Close output file
    output.close();
    
    return 0;
}
