#include <vector> 

#include "hernandez.h"
#include "pairing.h"
#include "operators.h"

Hernandez::Hernandez(const std::vector<Pair>& fixed_pairs) : pairs_(canonicalize_pairs(fixed_pairs)), composition_({
    {OperatorType::KEPLER, 0.5},
    {OperatorType::KICK, 0.5},
    {OperatorType::DRIFT, 1.0},
    {OperatorType::KICK, 0.5},
    {OperatorType::KEPLER, 0.5},
}) {};

void Hernandez::step(std::vector<Body>& bodies, double dt)
{
    extern double G;
    composition_.execute(bodies, pairs_, dt, G);
}