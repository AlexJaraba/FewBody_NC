#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cmath>
#include <vector>

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
        gravitational_constant <double>
    
    Lines are parsed as whitespace-separated key/value pairs.

   ============================================================================================================= */

namespace {
    bool is_blank_or_comment(const std::string& line) {
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        return first == std::string::npos || line[first] == '#';
    }
    void require_no_extra_tokens(std::istringstream& iss, const std::string& filename, std::size_t line_number) {
        std::string extra;
        if (iss >> extra) {
            throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " contains unexpected extra data: " + extra);
        }
    }
}

void readInitialConditions(const std::string& filename, std::vector<Body>& bodies) {
    std::ifstream infile(filename);
    std::string line;
    int next_id = 0;
    std::size_t line_number = 0;

    if (!infile) {throw std::runtime_error("Could not open " + filename); }
    while (std::getline(infile, line)) {
        ++line_number;
        if (is_blank_or_comment(line)) {
            continue;
        }
        std::istringstream iss(line);
        double mass;
        double radius = 0.0;
        Vec3 position; 
        Vec3 velocity;

        if (!(iss >> mass >> position.x >> position.y >> position.z >> velocity.x >> velocity.y >> velocity.z)) { 
            throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " must contain mass x y z vx vy vz [radius]."); 
        }
        if (iss >> radius) {
            require_no_extra_tokens(iss, filename, line_number);
        }
        if (!std::isfinite(mass) || mass <= 0.0) {
            throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " mass must be finite and positive.");
        }
        if (!position.is_finite() || !velocity.is_finite()) {
            throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " position and velocity must be finite.");
        }
        if (!std::isfinite(radius) || radius < 0.0) {
            throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " radius must be finite and non-negative.");
        }

        bodies.emplace_back(next_id, mass, position, velocity, radius);
        ++next_id;
    }
    if (bodies.empty()) {
        throw std::runtime_error("No bodies were read from " + filename);
    }
    std::cout << "Number of bodies read: " << bodies.size() << std::endl;
}

SolverParams readParams(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;
    std::size_t line_number = 0;
    SolverParams params = {};
    bool have_output_frequency = false;
    bool have_runtime = false;
    bool have_timestep = false;
    bool have_integrator = false;
    bool have_gravitational_constant = false;
    bool have_pair_order = false;

    if (!infile) {
        throw std::runtime_error("Could not open " + filename);
    }
    while (std::getline(infile, line)) {
        ++line_number;
        if (is_blank_or_comment(line)) {
            continue;
        }
        std::istringstream iss(line);
        std::string param;
        if (!(iss >> param)) {
            continue;
        }
        if (param == "output_frequency") {
            if (!(iss >> params.output_frequency)) {
                throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an invalid output_frequency.");
            }
            have_output_frequency = true;
        } else if (param == "runtime") {
            if (!(iss >> params.runtime)) {
                throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an invalid runtime.");
            }
            have_runtime = true;
        } else if (param == "timestep") {
            if (!(iss >> params.timestep)) {
                throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an invalid timestep.");
            }
            have_timestep = true;
        } else if (param == "integrator") {
            if (!(iss >> params.integrator)) {
                throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an invalid integrator.");
            }
            have_integrator = true;
        } else if (param == "gravitational_constant") {
            if (!(iss >> params.gravitational_constant)) {
                throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an invalid gravitational_constant.");
            }
            have_gravitational_constant = true;
        } else if (param == "pair_order") {
            if (!(iss >> params.pair_order)) {
                throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an invalid pair_order.");
            }
            have_pair_order = true;
        } else {
            throw std::runtime_error(filename + ": line " + std::to_string(line_number) + " has an unrecognized parameter: " + param);
        }
        require_no_extra_tokens(iss, filename, line_number);
    }
    std::vector<std::string> missing;
    if (!have_output_frequency) missing.push_back("output_frequency");
    if (!have_runtime) missing.push_back("runtime");
    if (!have_timestep) missing.push_back("timestep");
    if (!have_integrator) missing.push_back("integrator");
    if (!have_gravitational_constant) missing.push_back("gravitational_constant");
    if (!have_pair_order) missing.push_back("pair_order");
    if (!missing.empty()) {
        std::ostringstream msg;
        msg << filename << " is missing required parameters: ";
        for (std::size_t i = 0; i < missing.size(); ++i) {
            if (i > 0) msg << ", ";
            msg << missing[i];
        }
        throw std::runtime_error(msg.str());
    }
    if (params.output_frequency <= 0.0) throw std::runtime_error(filename + ": output_frequency must be positive.");
    if (!std::isfinite(params.runtime) || params.runtime < 0.0) throw std::runtime_error(filename + ": runtime must be finite and non-negative.");
    if (!std::isfinite(params.timestep) || params.timestep <= 0.0) throw std::runtime_error(filename + ": timestep must be finite and positive.");
    if (!std::isfinite(params.gravitational_constant) || params.gravitational_constant <= 0.0) throw std::runtime_error(filename + ": gravitational_constant must be finite and positive.");
    
    if (params.integrator != "hernandez" && params.integrator != "leapfrog") {
        throw std::runtime_error(filename + ": integrator must be 'hernandez' or 'leapfrog'.");
    }
    if (params.pair_order != "canonical" && params.pair_order != "strength" && params.pair_order != "auto") {
        throw std::runtime_error(filename + ": pair_order must be 'canonical', 'strength', or 'auto'.");
    }

    return params;
}

