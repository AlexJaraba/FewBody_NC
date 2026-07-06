#include "dynamics/jacobi_transform.h"

/* ========================================================================

    Jacobi Projection Matrix

    This file builds the matrix:

        A[a][k] = d r_a / d Q_k

    at a fixed total center of mass.

    This matrix is used to project Cartesian forces and force Jacobians into Jacobi generalized coordinates:

        F_Q = A^T F_r

        J_Q = A^T J_r A
    
    Correct signs in this matrix are essential for the perturbation force to cancel the Kepler force correctly in the pure two-body limit.

   ========================================================================*/

/*
    Build A[a][k] = d r_a / d Q_k
    For a < k, previous bodies move opposite to Q_k through the partial COM.
    For a == k, body k moves with coefficient M_{k-1} / M_k.
*/

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