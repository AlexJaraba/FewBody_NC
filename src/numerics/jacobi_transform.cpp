#include "jacobi_transform.h"

std::vector<std::vector<double>> build_jacobi_projection_matrix(const std::vector<double>& masses) {
    const int N = masses.size();

    std::vector<std::vector<double>> A(N, std::vector<double>(N,0.0));

    double enclosed = masses[0];

    A[0][0] = 1.0;

    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            A[i][j] = -masses[j] / enclosed;
        }
        A[i][i] = 1.0;

        enclosed+= masses[i];
    }

    return A;
}