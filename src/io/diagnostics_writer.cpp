#include <algorithm>
#include <cmath>
#include <iomanip>

#include "io/diagnostics_writer.h"

/* ============================================================================================================= 

    Diagnostics CSV Writer

    Writes scalar diagnostic quantities to diagnostics.csv so Python can plot conservation behavior without recomputing diagnostics from output.csv.

   ============================================================================================================= */

namespace {
    double absolute_difference(double value, double reference) {
        return std::abs(value - reference);
    }
    double relative_energy_error(double value, double reference) {
        const double scale = std::max(std::abs(reference), 1e-300);
        return std::abs(value - reference) / scale;
    }
}

DiagnosticsWriter::DiagnosticsWriter(const std::string& filename) : file_(filename) {file_ << std::setprecision(17);}

void DiagnosticsWriter::write_header_if_needed() {
    if (header_written_) {
        return;
    }

    file_ << "time,"
          << "total_energy,kinetic_energy,potential_energy,dE_over_E0,dE_abs,"
          << "linear_momentum,linear_momentum_x,linear_momentum_y,linear_momentum_z,dPx,dPy,dPz,"
          << "angular_momentum,angular_momentum_x,angular_momentum_y,angular_momentum_z,dLx,dLy,dLz,"
          << "com_drift,com_x,com_y,com_z,com_vx,com_vy,com_vz,"
          << "com_integral_x,com_integral_y,com_integral_z,dCcm_x,dCcm_y,dCcm_z,dRcm_x,dRcm_y,dRcm_z,"
          << "nine_integral_error_max,shadow_energy,dShadow_over_Shadow0,timestep\n";
        
    header_written_ = true;
}

void DiagnosticsWriter::set_reference(double time, const Diagnostics& diagnostics) {
    reference_time_ = time;
    reference_energy_ = diagnostics.total_energy;
    reference_angular_momentum_ = diagnostics.angular_momentum_vec;
    reference_linear_momentum_ = diagnostics.linear_momentum_vec;
    reference_center_of_mass_ = diagnostics.center_of_mass;
    reference_center_of_mass_velocity_ = diagnostics.center_of_mass_velocity;
    reference_set_ = true;
}

void DiagnosticsWriter::write(double time, const Diagnostics& d) {
    write_header_if_needed();
    if (!reference_set_) {
        set_reference(time, d);
    }

    const double dt_from_reference = time - reference_time_;
    const Vec3 com_integral = d.center_of_mass - reference_center_of_mass_ - dt_from_reference * reference_center_of_mass_velocity_;
    const double dE_abs = absolute_difference(d.total_energy, reference_energy_);
    const double dE_rel = relative_energy_error(d.total_energy, reference_energy_);
    const double dPx = absolute_difference(d.linear_momentum_vec.x, reference_linear_momentum_.x);
    const double dPy = absolute_difference(d.linear_momentum_vec.y, reference_linear_momentum_.y);
    const double dPz = absolute_difference(d.linear_momentum_vec.z, reference_linear_momentum_.z);
    const double dLx = absolute_difference(d.angular_momentum_vec.x, reference_angular_momentum_.x);
    const double dLy = absolute_difference(d.angular_momentum_vec.y, reference_angular_momentum_.y);
    const double dLz = absolute_difference(d.angular_momentum_vec.z, reference_angular_momentum_.z);
    const double dCcm_x = std::abs(com_integral.x);
    const double dCcm_y = std::abs(com_integral.y);
    const double dCcm_z = std::abs(com_integral.z);
    const double dRcm_x = absolute_difference(d.center_of_mass.x, reference_center_of_mass_.x);
    const double dRcm_y = absolute_difference(d.center_of_mass.y, reference_center_of_mass_.y);
    const double dRcm_z = absolute_difference(d.center_of_mass.z, reference_center_of_mass_.z);

    file_ << time << ","
          << d.total_energy << "," << d.kinetic_energy << "," << d.potential_energy << "," << dE_rel << "," << dE_abs << ","
          << d.linear_momentum << "," << d.linear_momentum_vec.x << "," << d.linear_momentum_vec.y << "," << d.linear_momentum_vec.z << "," << dPx << "," << dPy << "," << dPz << ","
          << d.angular_momentum << "," << d.angular_momentum_vec.x << "," << d.angular_momentum_vec.y << "," << d.angular_momentum_vec.z << "," << dLx << "," << dLy << "," << dLz << ","
          << d.com_drift << "," << d.center_of_mass.x << "," << d.center_of_mass.y << "," << d.center_of_mass.z << ","
          << d.center_of_mass_velocity.x << "," << d.center_of_mass_velocity.y << "," << d.center_of_mass_velocity.z << ","
          << com_integral.x << "," << com_integral.y << "," << com_integral.z << ","
          << dCcm_x << "," << dCcm_y << "," << dCcm_z << ","
          << dRcm_x << "," << dRcm_y << "," << dRcm_z << ","
          << d.timestep << "\n";
}

void DiagnosticsWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}