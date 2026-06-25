### Tests
Test 1 — binary
Purpose: Kepler baseline

Test 2 — circumbinary triple
Purpose: best hierarchical showcase

Test 3 — stronger perturbed triple
Purpose: moderate perturbation test

Test 4 — scattering / escape
Purpose: stress test, not expected to remain bounded

Test 5 — figure-8
Purpose: non-hierarchical limit test

Test 6 — close encounter
Purpose: close-encounter limit test

Test 7 — Solar System
Purpose: general planetary benchmark

Test 8 — inner planets
Purpose: inner fast-orbit planetary benchmark

### Default param.txt
```
output_frequency 20
runtime 365
timestep 0.025
coordinate_mode cartesian
integrator hernandez
gravitational_constant 0.000296014912
pair_order canonical

adaptive_timesteps false
timestep_levels 2
timestep_eta 0.001
timestep_refresh_interval 1
timestep_level_decrease_delay 3
```