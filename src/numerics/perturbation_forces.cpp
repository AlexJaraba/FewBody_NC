#include <cmath>

#include "numerics/perturbation_forces.h"
#include "core/reconstruction.h"

ForceResult compute_perturbation_forces(const CanonicalState& state, const std::vector<Pair>& pairs, double G) {
    const int N = state.Q.size();

    ForceResult result;

    result.gradient.resize(N);
    result.hessian.resize(N, std::vector<Mat3>(N));

    auto r = reconstruct_cartesian_positions(state);
    
    for (const auto& pair : pairs) {
        const int i = pair.i;
        const int j = pair.j;

        Vec3 dr = r[j] - r[i];

        const double r2 = dr.norm2();
        const double dist = std::sqrt(r2) + 1e-15;
        const double inv_r3 = 1.0 / (dist * dist * dist);
        const double inv_r5 = inv_r3 / r2;
        const double coeff = G * state.physical_mass[i] * state.physical_mass[j];

        result.potential -= coeff / dist;
        Vec3 f = coeff * inv_r3 * dr;

        result.gradient[i] -= f;
        result.gradient[j] += f;

        Mat3 I;
        I.xx = 1.0; 
        I.yy = 1.0;
        I.zz = 1.0;

        Mat3 rrT = outer(dr, dr);

        Mat3 H;
        H.xx = coeff * (3.0 * rrT.xx * inv_r5 - I.xx * inv_r3);
        H.xy = coeff * (3.0 * rrT.xy * inv_r5 - I.xy * inv_r3);
        H.xz = coeff * (3.0 * rrT.xz * inv_r5 - I.xz * inv_r3);
        H.yx = coeff * (3.0 * rrT.yx * inv_r5 - I.yx * inv_r3);
        H.yy = coeff * (3.0 * rrT.yy * inv_r5 - I.yy * inv_r3);
        H.yz = coeff * (3.0 * rrT.yz * inv_r5- I.yz * inv_r3);
        H.zx = coeff * (3.0 * rrT.zx * inv_r5 - I.zx * inv_r3);
        H.zy = coeff * (3.0 * rrT.zy * inv_r5 - I.zy * inv_r3);
        H.zz = coeff * (3.0 * rrT.zz * inv_r5 - I.zz * inv_r3);

        result.hessian[i][i] += H;
        result.hessian[j][j] += H;
    }
    return result;
}