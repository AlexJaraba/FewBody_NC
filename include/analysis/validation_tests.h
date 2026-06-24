#pragma once

class Solver;

class Tests {
public:
    explicit Tests(Solver& solver);

    
    void TestHernandezAdjoint(double dt);
    void TestLocalOrder();
    void ReversibilityTest();
    void TestHierarchyTreeModel();
    void TestHierarchySelectionCriteria();

    void TestHB15StateRoundTrip();
    void TestHB15PairStateRoundTrip();
    void TestHB15PairDiagnostics();
    void TestHB15PairKeplerSuite();
    void TestHB15PairOrdering();
    
    void TestHB15HierarchyDiagnostics();
    void TestHB15SymmetricOrdering();
    void TestHB15RemainderOperator();
    void TestHB15FixedStepValidation();
    void TestHB15Reversibility();
    void TestHB15RecursiveOrderingPrototype();

private:
    Solver& solver_;
};