#pragma once

#include <fstream>
#include <vector>
#include <string>

#include "io/output_writer.h"

class CSVOutputWriter final: public OutputWriter {
public:
    explicit CSVOutputWriter(const std::string& filename);
    ~CSVOutputWriter() override;

    CSVOutputWriter(const CSVOutputWriter&) = delete;
    CSVOutputWriter& operator=(const CSVOutputWriter&) = delete;

    void write(const std::vector<BodyState>& bodies) override;
    void close() override;

private:
    std::ofstream file_;
    bool header_written_ = false;
};
