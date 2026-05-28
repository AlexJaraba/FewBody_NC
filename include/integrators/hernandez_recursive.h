#pragma once

#include <memory>

#include "dynamics/hierarchy_tree.h"
#include "core/canonical_state.h"

void recursive_hernandez_step(std::shared_ptr<HierarchyNode> node, CanonicalState& state, double dt, double G);