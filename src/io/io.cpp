#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

#include "io/io.h"
#include "math/vec3.h"

/* ============================================================================================================= 

    Input Parsing

    initial_conditions.txt format: mass x y z vx vy vz
    param.txt format:
        output_frequency <int>
        runtime <double>
        timestep <double>
        integrator <string>
        coordinate_mode <jacobi|cartesian>
        gravitational_constant <double>
    
    Lines are parsed as whitespace-separated key/value pairs.

   ============================================================================================================= */

namespace {bool parse_bool(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {return static_cast<char>(std::tolower(c)); });
    return value == "true" || value == "1" || value == "yes" || value == "on";
}}

void readInitialConditions(const std::string& filename, std::vector<Body>& bodies) {
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double mass;
        Vec3 position; 
        Vec3 velocity;

        if (!(iss >> mass >> position.x >> position.y >> position.z 
                          >> velocity.x >> velocity.y >> velocity.z)) { continue; }
        
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
    SolverParams params = {100, 10.0, 0.01, "hernandez", "cartesian", 6.67430e-11, "canonical",false, 0, 0.03, 1, 3};  // Default values


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
            } else if (param == "coordinate_mode") {
                iss >> params.coordinate_mode;
            } else if (param == "gravitational_constant") {
	            iss >> params.gravitational_constant;
            } else if (param == "pair_order") {
                iss >> params.pair_order;
            } else if (param == "adaptive_timesteps") {
                std::string value;
                iss >> value;
                params.adaptive_timesteps = parse_bool(value);
            } else if (param == "timestep_levels") {
                iss >> params.timestep_levels;
            } else if (param == "timestep_eta") {
                iss >> params.timestep_eta;
            } else if (param == "timestep_refresh_interval") {
                iss >> params.timestep_refresh_interval;
            } else if (param == "timestep_level_decrease_delay") {
                iss >> params.timestep_level_decrease_delay;
            }
        }
    }

    return params;
}
