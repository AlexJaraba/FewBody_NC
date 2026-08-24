#pragma once

#include <fstream>
#include <string>

#include "analysis/diagnostics.h"
#include "math/vec3.h"

class DiagnosticsWriter {
public:
    explicit DiagnosticsWriter(const std::string& filename);
    void write (double time, const Diagnostics& diagnostics);
    void close();
private:
    std::ofstream file_;
    bool header_written_ = false;
    bool reference_set_ = false;
    double reference_time_ = 0.0;
    double reference_energy_ = 0.0;
    Vec3 reference_angular_momentum_;
    Vec3 reference_linear_momentum_;
    Vec3 reference_center_of_mass_;
    Vec3 reference_center_of_mass_velocity_;

    void writeHeader();
    void setReference(double time, const Diagnostics& diagnostics);
};