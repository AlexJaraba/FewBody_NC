#ifndef BODY_H
#define BODY_H

#include <vector>

struct Body {
    double mass;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> acceleration;

    Body(double m, std::vector<double> pos, std::vector<double> vel);
    void updateAcceleration(const std::vector<Body>& bodies);
    void updatePosition(double dt);
    void updateVelocity(double dt);
    void updateCenterofMass(const std::vector<Body>& bodies);
};

#endif // BODY_H
