#pragma once

#include <memory>
#include <vector>

#include "dynamics/hierarchy_node.h"

class HierarchyTree {
public:
    std::shared_ptr<HierarchyNode> root;
    explicit HierarchyTree(const std::vector<Body>& bodies);
private:
    std::shared_ptr<HierarchyNode> buildTree(const std::vector<Body>& bodies, std::vector<int> indices);
};