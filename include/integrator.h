#pragma once

#include <vector>

class Body;

class Integrator {
public:
    virtual void step(std::vector<Body>& bodies, double dt) = 0;
    virtual ~Integrator() = default;
};

class Leapfrog : public Integrator {
public:
    void step(std::vector<Body>& bodies, double dt) override;
};

class Hernandez : public Integrator {
public:
    void step(std::vector<Body>& bodies, double dt) override;
    virtual ~Hernandez() override = default;
};
