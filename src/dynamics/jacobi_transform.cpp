#include "dynamics/jacobi_transform.h"

std::vector<std::vector<double>> build_jacobi_projection_matrix(const CanonicalState& state) {
    const int N = static_cast<int>(state.physical_mass.size());

    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));

    for (int k = 1; k < N; ++k) {
        double M_prev = 0.0;
        for (int a = 0; a < k; ++a) {
            M_prev += state.physical_mass[a];
        }
        const double mk = state.physical_mass[k];
        const double Mk = M_prev + mk;
        for (int a = 0; a < k; ++a) {
            A[a][k] = -mk / Mk;
        }
        A[k][k] = M_prev / Mk;
    }
    return A;
}