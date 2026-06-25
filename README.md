# FewBodyNC
Numerical integration package for few massive bodies (black holes) orbiting the center of a nuclear cluster while interacting with each other

Build:
make clean
make

Run simulation:
.\few_body_nc.exe

Plot output:
python python/plot_output.py

Run benchmark suite:
python python/plot_output.py --benchmark

Run convergence study:
python python/plot_output.py --convergence

## Production defaults and integrator status
The default close-encounter/few-body production path is:

```
coordinate_mode cartesian
integrator hernandez
pair_order canonical
adaptive_timesteps false
gravitational_constant 0.000296014912
```

Use this as the main reference mode for binaries, close encounters, scattering tests, and general few-body work.

### Integrator selection guide

System type
    - Recommended mode
    - Status

Binary / isolated Kepler pair 
    - `coordinate_mode cartesian`, `integrator hernandez`, `pair_order canonical`, `adaptive_timesteps false` 
    - Production default
Close encounter / scattering / general few-body | `coordinate_mode cartesian`, `integrator hernandez`, `pair_order canonical`, `adaptive_timesteps false` | Production default
Hierarchical triple / circumbinary triple 
    - `coordinate_mode jacobi`, `integrator hernandez`, `adaptive_timesteps false` 
    - Production hierarchical default
Inner-planets / planetary-style hierarchy 
    - `coordinate_mode jacobi`, `integrator hernandez`, `adaptive_timesteps false` 
    - Production hierarchical default
Full solar-system-like secular benchmark 
    - `coordinate_mode jacobi`, `integrator hernandez`, `adaptive_timesteps true` 
    - Specialized; use only when benchmarked
Figure-8 benchmark 
    - `coordinate_mode cartesian`, `integrator leapfrog`, `adaptive_timesteps false` 
    - Baseline/stress-test comparison

### Pair-order policy

`pair_order` - Status 
    - Notes

`canonical` - Production default 
    - Most robust pair ordering; recommended for normal Cartesian Hernandez runs.
`strength` - Validated optional fixed-step mode 
    - Passed the benchmark crash/failure gate after pair-map retry support, but remains worse than canonical on 
      close-encounter and scattering accuracy.
`auto` - Experimental selector 
    - May resolve to non-canonical ordering. Do not use as the production default yet.

### Adaptive timestep policy

Adaptive timesteps are not a universal default. Use fixed timesteps unless a specific benchmark supports adaptive mode for the system class. Current policy:

- Use `adaptive_timesteps false` for binaries, scattering, close encounters, Figure-8, and general few-body tests.
- Use Jacobi Hernandez adaptive only for validated solar-system-like secular cases.
- Treat Cartesian Hernandez adaptive block mode as experimental until it passes stronger fixed-vs-adaptive comparisons.

### Known limitations

- Cartesian leapfrog is useful as a baseline and Figure-8 comparison method, not the main scientific integrator.
- `pair_order strength` is no longer a crashing experimental path, but it is not more accurate than canonical 
  on the hardest close-encounter/scattering cases.
- `pair_order auto` should not be used as the production default until its selection logic is benchmark-gated.
- Higher-order methods, true adaptive timesteps, tangent maps, chaos indicators, and relativity should wait until 
  the fixed second-order Hernandez path has formal convergence and bounded-energy tests.