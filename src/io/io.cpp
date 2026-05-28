#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "io/io.h"
#include "math/vec3.h"

void readInitialConditions(const std::string& filename, std::vector<Body>& bodies) {
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double mass;
        Vec3 position; 
        Vec3 velocity;

        if (!(iss >> mass >> position.x >> position.y >> position.z 
                   >> velocity.x >> velocity.y >> velocity.z)) { break; }
        
        bodies.emplace_back(mass, position, velocity);
    }
    std::cout << "Number of bodies read: " << bodies.size() << std::endl;
}

void writeOutput(const std::string& filename, const std::vector<Body>& bodies, double time) {
    std::ofstream outfile(filename, std::ios_base::app);  // Append to the file

    outfile << time;  // Start with the current time

    for (const auto& body : bodies) {
        outfile 
            << " " << body.mass
            << " " << body.position.x
            << " " << body.position.y
            << " " << body.position.z
            << " " << body.velocity.x
            << " " << body.velocity.y
            << " " << body.velocity.z;
    }
    outfile << "\n";  // Newline at the end of each timestep
}

SolverParams readParams(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line, param;
    SolverParams params = {100, 10.0, 0.01, "leapfrog", 6.67430e-11};  // Default values


    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        if (iss >> param) {
            if (param == "output_frequency") {
                iss >> params.output_frequency;
            } else if (param == "runtime") {
                iss >> params.runtime;
            } else if (param == "timestep") {
                iss >> params.timestep;
            } else if (param == "integrator") {
                iss >> params.integrator;
            } else if (param == "gravitational_constant") {
	            iss >> params.gravitational_constant;
	    }
        }
    }

    return params;
}
