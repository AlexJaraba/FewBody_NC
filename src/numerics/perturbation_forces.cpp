#include <cmath>
#include <stdexcept>

#include "numerics/perturbation_forces.h"
#include "core/reconstruction.h"
#include "dynamics/jacobi_transform.h"

namespace {
    Mat3 identity_matrix() {
        Mat3 I;
        I.xx = 1.0; 
        I.yy = 1.0;
        I.zz = 1.0;
        return I;
    }

    void add_scaled(Mat3& dst, const Mat3& src, double s) {
        dst.xx += s * src.xx;
        dst.xy += s * src.xy;
        dst.xz += s * src.xz;
        dst.yx += s * src.yx;
        dst.yy += s * src.yy;
        dst.yz += s * src.yz;
        dst.zx += s * src.zx;
        dst.zy += s * src.zy;
        dst.zz += s * src.zz;
    }

    Mat3 force_jacobian_from_relative_vector(const Vec3& dr, double coeff) {
        const double r2 = dr.norm2();
        const double r = std::sqrt(r2);

        if (r < 1e-14) {
            throw std::runtime_error("Singualr pair distance in for Jacobian.");
        }

        const double inv_r3 = 1.0 / (r2 * r);
        const double inv_r5 = inv_r3 / r2;

        Mat3 I = identity_matrix();
        Mat3 rrT = outer(dr, dr);
        Mat3 J;

        J.xx = coeff * (3.0 * rrT.xx * inv_r5 - I.xx * inv_r3);
        J.xy = coeff * (3.0 * rrT.xy * inv_r5 - I.xy * inv_r3);
        J.xz = coeff * (3.0 * rrT.xz * inv_r5 - I.xz * inv_r3);
        J.yx = coeff * (3.0 * rrT.yx * inv_r5 - I.yx * inv_r3);
        J.yy = coeff * (3.0 * rrT.yy * inv_r5 - I.yy * inv_r3);
        J.yz = coeff * (3.0 * rrT.yz * inv_r5 - I.yz * inv_r3);
        J.zx = coeff * (3.0 * rrT.zx * inv_r5 - I.zx * inv_r3);
        J.zy = coeff * (3.0 * rrT.zy * inv_r5 - I.zy * inv_r3);
        J.zz = coeff * (3.0 * rrT.zz * inv_r5 - I.zz * inv_r3);

        return J;
    }
}

ForceResult compute_perturbation_forces(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    const int N = static_cast<int>(state.Q.size());

    ForceResult result;
    result.potential = 0.0;
    result.gradient.resize(N);
    result.hessian.resize(N, std::vector<Mat3>(N));

    const auto r = reconstruct_cartesian_positions(state);
    const auto A = build_jacobi_projection_matrix(state);

    std::vector<Vec3> cartesian_forces(N);
    std::vector<std::vector<Mat3>> cartesian_force_jacobians(N, std::vector<Mat3>(N));

    (void)pairs; // Current Jacobi split requires all physical Cartesian pairs.

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec3 dr = r[j] - r[i];

            const double r2 = dr.norm2();
            const double dist = std::sqrt(r2);

            if (dist < 1e-14) {
                throw std::runtime_error("Singular pair distance in perturbation force.");
            }

            const double coeff = G * state.physical_mass[i] * state.physical_mass[j];
            const double inv_r3 = 1.0 / (r2 * dist);

            result.potential -= coeff / dist;

            const Vec3 f = coeff * inv_r3 * dr;
            cartesian_forces[i] += f;
            cartesian_forces[j] -= f;

            const Mat3 J = force_jacobian_from_relative_vector(dr, coeff);
            cartesian_force_jacobians[i][i] += J;
            cartesian_force_jacobians[i][j] -= J;
            cartesian_force_jacobians[j][i] -= J;
            cartesian_force_jacobians[j][j] += J;
        }
    }

    for (int k = 1; k < N; ++k) {
        Vec3 Fk;
        for (int a = 0; a < N; ++a) {
            Fk += A[a][k] * cartesian_forces[a];
        }
        result.gradient[k] = Fk;
    }

    for (int k = 1; k < N; ++k) {
        for (int l = 1; l < N; ++l) {
            for (int a = 0; a < N; ++a) {
                if (A[a][k] == 0.0) continue;
                for (int b = 0; b < N; ++b) {
                    if (A[b][l] == 0.0) continue;
                    add_scaled(result.hessian[k][l], cartesian_force_jacobians[a][b], A[a][k] * A[b][l]);
                }
            }
        }
    }
    
    for (int k = 1; k < N; ++k) {
        const Vec3 q = state.Q[k];
        const double r2 = q.norm2();
        const double dist = std::sqrt(r2);

        if (dist < 1e-14) {
            continue;
        }

        const double coeff = G * state.mu[k] * state.M[k];
        const double inv_r3 = 1.0 / (r2 * dist);
        const Vec3 F_kepler = -(coeff * inv_r3) * q;

        result.gradient[k] -= F_kepler;
        result.potential += coeff / dist;
        const Mat3 J_kepler = force_jacobian_from_relative_vector(q, coeff);
        result.hessian[k][k] -= J_kepler;
    }
    return result;
}