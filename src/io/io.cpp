#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

#include "io/io.h"
#include "math/vec3.h"

/* ============================================================================================================= 

    Input Parsing

    initial_conditions.txt format: mass x y z vx vy vz [radius]
    param.txt format:
        output_frequency <int>
        runtime <double>
        timestep <double>
        integrator <string>
        coordinate_mode <jacobi|cartesian>
        gravitational_constant <double>
    
    Lines are parsed as whitespace-separated key/value pairs.

   ============================================================================================================= */

void readInitialConditions(const std::string& filename, std::vector<Body>& bodies) {
    std::ifstream infile(filename);
    std::string line;
    int next_id = 0;

    if (!infile) {throw std::runtime_error("Could not open " + filename); }
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double mass;
        double radius = 0.0;
        Vec3 position; 
        Vec3 velocity;

        if (!(iss >> mass >> position.x >> position.y >> position.z 
                          >> velocity.x >> velocity.y >> velocity.z)) { continue; }
        
        if (iss >> radius) {
            if (radius < 0.0) {
                throw std::runtime_error("Initial condition radius must be non-negative.");
            }
        }

        bodies.emplace_back(next_id, mass, position, velocity, radius);
        ++next_id;
    }
    std::cout << "Number of bodies read: " << bodies.size() << std::endl;
}

SolverParams readParams(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line, param;
    SolverParams params = {100, 10.0, 0.01, "hernandez", 0.000296014912, "canonical"};  // Default values

    if (!infile) {throw std::runtime_error("Could not open " + filename); }
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        if (!(iss >> param)) {
            continue;
        }
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
        } else if (param == "pair_order") {
            iss >> params.pair_order;
        }
    }

    return params;
}

