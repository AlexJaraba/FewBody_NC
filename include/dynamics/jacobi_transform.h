#pragma once

#include <vector>

#include "core/canonical_state.h"

std::vector<std::vector<double>> build_jacobi_projection_matrix(const CanonicalState& state);