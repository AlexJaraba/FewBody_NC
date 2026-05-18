#include <vector>

#include "jacobi.h"

void compute_jacobi_coordinates(std::vector<Body>& bodies) {
    const int N = bodies.size();
    if (N < 2) return;

    std::vector<std::vector<double>> pos_snapshot(N);
    std::vector<std::vector<double>> mom_snapshot(N);

    for (int i = 0; i < N; ++i) {
        pos_snapshot[i] = bodies[i].position;
        mom_snapshot[i] = bodies[i].momentum;
    }

    std::vector<double> cumulative_mass(N, 0.0);
    cumulative_mass[0] = bodies[0].mass;

    for (int i = 1; i < N; ++i) {
        cumulative_mass[i] = cumulative_mass[i-1] + bodies[i].mass;
    }

    for (int k = 0; k < 3; ++k) {
        bodies[0].jacobi_position[k] = pos_snapshot[0][k];
        bodies[0].jacobi_momentum[k] = mom_snapshot[0][k];
    }

    for (int i = 1; i < N; ++i) {

        std::vector<double> com(3, 0.0);
        std::vector<double> pcom(3, 0.0);

        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < 3; ++k) {
                com[k] += pos_snapshot[j][k] * bodies[j].mass;
                pcom[k] += mom_snapshot[j][k];
            }
        }

        for (int k = 0; k < 3; ++k) {
            com[k] /= cumulative_mass[i-1];

            bodies[i].jacobi_position[k] = pos_snapshot[i][k] - com[k];
            bodies[i].jacobi_momentum[k] = mom_snapshot[i][k] - (bodies[i].mass / cumulative_mass[i-1]) * pcom[k];
        }
    }
}

void reconstruct_barycentric_coordinates(std::vector<Body>& bodies) {
    const int N = bodies.size();
    if (N < 2) return;

    std::vector<std::vector<double>> jacobi_pos(N);
    std::vector<std::vector<double>> jacobi_mom(N);

    for (int i = 0; i < N; ++i) {
        jacobi_pos[i] = bodies[i].jacobi_position;
        jacobi_mom[i] = bodies[i].jacobi_momentum;
    }

    std::vector<double> cumulative_mass(N, 0.0);
    cumulative_mass[0] = bodies[0].mass;

    for (int i = 1; i < N; ++i) {
        cumulative_mass[i] = cumulative_mass[i-1] + bodies[i].mass;
    }

    for (int k = 0; k < 3; ++k) {
        bodies[0].position[k] = jacobi_pos[0][k];
        bodies[0].momentum[k] = jacobi_mom[0][k];
    }

    for (int i = 1; i < N; ++i) {
        std::vector<double> com(3, 0.0);
        std::vector<double> pcom(3, 0.0);

        for (int j = 0; j < i; ++j) {
            for (int k = 0; k < 3; ++k) {
                com[k] += bodies[j].mass * bodies[j].position[k];
                pcom[k] += bodies[j].momentum[k];
            }
        }

        for (int k = 0; k < 3; ++k) {
            com[k] /= cumulative_mass[i-1];

            bodies[i].position[k] = com[k] + jacobi_pos[i][k];
            bodies[i].momentum[k] = jacobi_mom[i][k] + (bodies[i].mass / cumulative_mass[i-1]) * pcom[k];
        }
    }

    for (auto& body : bodies) {
        body.updateVelocityFromMomentum();
    }
}