// output_writer.h
#pragma once

#include <vector>

struct BodyState;

class OutputWriter {
public:
    virtual ~OutputWriter() = default;
    virtual void write(const std::vector<BodyState>& bodies) = 0;
    virtual void close() = 0;
};

