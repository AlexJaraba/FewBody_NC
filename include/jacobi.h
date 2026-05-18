#pragma once

#include <vector>

#include "body.h"

void compute_jacobi_coordinates(std::vector<Body>& bodies);
void reconstruct_barycentric_coordinates(std::vector<Body>& bodies);