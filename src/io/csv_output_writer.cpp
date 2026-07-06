#include <iomanip>

#include "io/csv_output_writer.h"
#include "core/body.h"

/* ============================================================================================================= 

    CSV Output Writer

    Writes physical Cartesian body states to output.csv

    Output columns: time,id,x,y,z,vx,vy,vz,mass

    This format is consumed by python/functions.py and python/plot_output.py

   ============================================================================================================= */

CSVOutputWriter::CSVOutputWriter(const std::string& filename)
    : file_(filename), header_written_(false) {file_ << std::setprecision(17);}

void CSVOutputWriter::write(const std::vector<BodyState>& bodies) {
    if (!header_written_) {
        file_ << "time,id,x,y,z,vx,vy,vz,mass,radius\n";
        header_written_ = true;
    }

    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& b = bodies[i];
        file_ << b.time << "," 
              << b.id << ","
              << b.position[0] << "," << b.position[1] << "," << b.position[2] << ","
              << b.velocity[0] << "," << b.velocity[1] << "," << b.velocity[2] << ","
              << b.mass << "," << b.radius << "\n";
    }
}

void CSVOutputWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}
