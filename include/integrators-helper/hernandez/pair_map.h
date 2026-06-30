#pragma once

#include <vector>

#include "core/body.h"

/* =====================================================================

    Hernandez pairwise Kepler map.

    This applies one exact two-body Kepler evolution to one physical body pair (i, j), using the existing universal-variable Kepler propagator.
    Close-encounter hardening is handled by HernandezBodyStepper, which may split one difficult pair map into several smaller exact Kepler maps.

   ===================================================================== */

struct HernandezPairMapResult {
    bool converged;
    int iterations;
};

HernandezPairMapResult apply_hernandez_pair_kepler_map(std::vector<Body>& bodies, int i, int j, double dt, double G);