# Constrained Optimization Solver

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

Implementation of Lagrangian relaxation and dual gradient ascent algorithms for solving constrained optimization problems.

## Overview

This project demonstrates the application of:
- **Lagrangian Relaxation**: Converting constrained problems to unconstrained
- **Dual Gradient Ascent**: Iterative algorithm for finding optimal dual variables
- **Convergence Analysis**: Visualization of algorithm convergence behavior

## Tech Stack

- **Python 3.8+**
- **NumPy**: Numerical computations
- **Matplotlib**: Convergence visualization

## Project Structure

```
constrained-optimization/
├── src/
│   ├── run.py                      # Main execution script
│   ├── problem_definition.py       # Optimization problem setup
│   ├── lagrangian_relaxation.py    # Lagrangian formulation
│   ├── dual_gradient_ascent.py     # Dual ascent algorithm
│   └── plots.py                    # Visualization utilities
├── results/                        # Auto-generated outputs
│   ├── dual_convergence.png        # Convergence plot
│   └── solution.txt                # Optimal solution
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd src
python run.py
```

Results will be saved to the `results/` directory.

## Features

- Custom optimization problem definition
- Dual gradient ascent implementation
- Convergence visualization
- Solution verification

## License

MIT License - see [LICENSE](../LICENSE)

---

**Note**: This project demonstrates algorithmic implementation for optimization theory, complementing the machine learning work in the bike-demand-prediction project.
