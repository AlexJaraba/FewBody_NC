#pragma once

#include <fstream>
#include <string>

#include "diagnostics.h"

class DiagnosticsWriter {
public:
    explicit DiagnosticsWriter(const std::string& filename);
    void write (double time, const Diagnostics& diagnostics);
    void close();
private:
    std::ofstream file_;
    bool header_written_;
};