# FewBodyNC

## Requirements
- C++17 compiler
- Python 3
- matplotlib
- numpy
- pandas
- pyarrow is optional and is used only for Parquet benchmark tables.

## Plotting and tests
```text
python python/plot_output.py
python python/plot_output.py --benchmark
python python/plot_output.py --convergence
```

The executable recenters the supplied initial conditions into the center-of-mass position and velocity frame before integration.
