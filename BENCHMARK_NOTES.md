### Tests
Test 1 — binary
Purpose: Kepler baseline

Test 2 — circumbinary triple
Purpose: hierarchical-system benchmark

Test 3 — stronger perturbed triple
Purpose: moderate perturbation test

Test 4 — scattering / escape
Purpose: stress test; bounded orbital behavior is not assumed

Test 5 — figure-8
Purpose: non-hierarchical benchmark

Test 6 — close encounter
Purpose: close-encounter stress test

Test 7 — Solar System
Purpose: long-term planetary benchmark; do not include the 4.567-Gyr run in routine benchmark sweeps

Test 8 — inner planets
Purpose: inner fast-orbit planetary benchmark

Test 9 — normalized Hernandez three-body test
Purpose: timestep-scaling and conservation test

Test 10 - Chaotic Bodys
Purpose: Stress Test

### Current param.txt format
```text
output_frequency 20
runtime 365
timestep 0.025
integrator hernandez
gravitational_constant 0.000296014912
pair_order canonical
```

`pair_order auto` currently resolves explicitly to canonical ordering. `strength` remains available as an experimental explicit choice.
