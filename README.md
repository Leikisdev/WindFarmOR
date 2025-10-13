# Wind Farm Layout Optimization using JAX and SciPy

## Overview
This project implements a **gradient-based optimization** framework for wind farm layout design, using **JAX** for automatic differentiation and **SciPy’s SLSQP** for constrained optimization.  
The goal is to **maximize the Annual Energy Production (AEP)** of a wind farm while enforcing physical and spatial constraints on turbine placement.

---

## Approach

### 1. **Objective Function**
The optimization objective is defined as the **negative total AEP** where AEP is computed from wind resource data (wind speed, direction, and frequency) and turbine characteristics (cut-in/out speeds, rated power, etc.) via the `calcAEP()` model.

### 2. **Constraints**
Each turbine must remain **within a circular farm boundary** implemented as inequality constraints:
```python
farm_radius - sqrt(x_i**2 + y_i**2) >= 0