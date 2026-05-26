#include <iomanip>

#include "io/csv_output_writer.h"
#include "core/body.h"

CSVOutputWriter::CSVOutputWriter(const std::string& filename)
    : file_(filename), header_written_(false) {file_ << std::setprecision(17);}

void CSVOutputWriter::write(const std::vector<BodyState>& bodies) {
    if (!header_written_) {
        file_ << "time,id,x,y,z,vx,vy,vz,mass\n";
        header_written_ = true;
    }

    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& b = bodies[i];
        file_ << b.time << "," << i << ","
             << b.position[0] << "," << b.position[1] << "," << b.position[2] << ","
             << b.velocity[0] << "," << b.velocity[1] << "," << b.velocity[2] << ","
             << b.mass << "\n";
    }
}

void CSVOutputWriter::close() {
    if (file_.is_open()) file_.close();
}
