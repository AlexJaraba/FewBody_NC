#pragma once

#include <vector>

#include "core/body.h"

/* =====================================================================

    HB15 pairwise Kepler map.

    This applies one exact two-body Kepler evolution to one physical Cartesian pair (i, j), using the existing universal-variable Kepler propagator.

   ===================================================================== */

struct HB15PairMapResult {
    bool converged;
    int iterations;
};

HB15PairMapResult apply_hb15_pair_kepler_map(std::vector<Body>& bodies, int i, int j, double dt, double G);