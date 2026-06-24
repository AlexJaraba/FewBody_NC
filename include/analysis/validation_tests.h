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

    void TestHernandezStateRoundTrip();
    void TestHernandezPairStateRoundTrip();
    void TestHernandezPairDiagnostics();
    void TestHernandezPairKeplerSuite();
    void TestHernandezPairOrdering();
    
    void TestHernandezHierarchyDiagnostics();
    void TestHernandezSymmetricOrdering();
    void TestHernandezRemainderOperator();
    void TestHernandezFixedStepValidation();
    void TestHernandezReversibility();
    void TestHernandezRecursiveOrderingPrototype();
    void TestHernandezRecursiveOrderingValidation();
    void TestHernandezPairLevelScheduler();
    void TestHernandezBlockTimestepSequenceDesign();
    void TestHernandezAdaptiveBlockValidation();

private:
    Solver& solver_;
};