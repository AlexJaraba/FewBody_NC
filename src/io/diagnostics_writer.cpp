#include <iomanip>

#include "io/diagnostics_writer.h"

/* ============================================================================================================= 

    Diagnostics CSV Writer

    Writes scalar diagnostic quantities to diagnostics.csv so Python can plot conservation behavior without recomputing diagnostics from output.csv.

   ============================================================================================================= */

DiagnosticsWriter::DiagnosticsWriter(const std::string& filename) : file_(filename), header_written_(false) {file_ << std::setprecision(17);}

void DiagnosticsWriter::write(double time, const Diagnostics& d) {
    if (!header_written_) {
        file_ << "time," << "total_energy," << "kinetic_energy," << "potential_energy," << "linear_momentum," << "angular_momentum," << "com_drift," << "shadow_energy," << "timestep\n";
        header_written_ = true;
    }

    file_ << time << ","
          << d.total_energy << ","
          << d.kinetic_energy << ","
          << d.potential_energy << ","
          << d.linear_momentum << ","
          << d.angular_momentum << ","
          << d.com_drift << ","
          << d.shadow_energy << ","
          << d.timestep << "\n";
}

void DiagnosticsWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}