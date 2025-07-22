#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <vector>
#include "body.h"

class Integrator {
public:
    virtual void step(std::vector<Body>& bodies, double dt) = 0;
    virtual ~Integrator() {}
};

class Leapfrog : public Integrator {
public:
    void step(std::vector<Body>& bodies, double dt) override;
};

class Hernandez : public Integrator {
public:
    virtual void step(std::vector<Body>& bodies, double dt) = 0;
    virtual ~Hernandez() {}
};

#endif // INTEGRATOR_H
