#pragma once

#include <fstream>
#include <string>

#include "analysis/diagnostics.h"

class DiagnosticsWriter {
public:
    explicit DiagnosticsWriter(const std::string& filename);
    ~DiagnosticsWriter();

    DiagnosticsWriter(const DiagnosticsWriter&) = delete;
    DiagnosticsWriter& operator=(const DiagnosticsWriter&) = delete;

    void write (double time, const SystemDiagnostics& diagnostics);
    void close();
private:
    std::ofstream file_;
    bool header_written_ = false;
    bool reference_set_ = false;
    double reference_time_ = 0.0;
    double reference_energy_ = 0.0;
    Vec3 reference_angular_momentum_;
    Vec3 reference_linear_momentum_;
    Vec3 reference_COM_position_;
    Vec3 reference_COM_velocity_;

    void write_header_if_needed();
    void set_reference(double time, const SystemDiagnostics& diagnostics);
    void require_successful_write() const;
};