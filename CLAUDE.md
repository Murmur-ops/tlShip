# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the System
```bash
# Default 30-node demo (achieves 18.5mm RMSE)
python run_demo.py

# With specific configuration
python run_demo.py --config configs/example_30node.yaml

# 8-node system
python run_demo.py --config configs/example_8node.yaml
```

### Testing with Noise Model
```bash
# Test with different noise presets
python test_30node_system.py ideal       # No noise (~18.5mm RMSE)
python test_30node_system.py clean       # Lab conditions
python test_30node_system.py realistic   # Typical deployment
python test_30node_system.py harsh       # Challenging environment

# Compare all noise presets
python test_30node_system.py --compare

# Validate noise model
python test_noise_validation.py

# Analyze noise impact
python analyze_noise_impact.py
```

### Legacy Testing
```bash
# Run the original 30-node test
python test_30node_system.py ideal
```

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

## Architecture

### Core Components

The Time-Localization (TL) system is a distributed localization and time synchronization system achieving sub-centimeter accuracy through adaptive optimization.

**Main Entry Points:**
- `run_demo.py`: Main demo script with YAML configuration support and visualization
- `test_30node_system.py`: Verification test for 30-node network (18.5mm RMSE)
- `ftl_enhanced.py`: Enhanced FTL solver with Adaptive Levenberg-Marquardt optimization

**Key Modules:**
- `ftl/optimization/`: Optimization algorithms
  - `adaptive_lm.py`: Adaptive Levenberg-Marquardt implementation
  - `line_search.py`: Line search methods (Armijo, Wolfe)
- `ftl/noise_model.py`: Comprehensive noise model for realistic simulations
  - Thermal, multipath, hardware, and environmental noise sources
  - Configurable presets: ideal, clean, realistic, harsh
- `ftl/`: Core FTL algorithms and utilities for factors, solving, metrics, and initialization

### Configuration System

The system uses YAML configuration files (in `configs/`) with the following structure:
- `network`: Node count and anchor configuration
- `optimization`: Algorithm parameters (adaptive LM, line search, convergence criteria)
- `measurements`: Noise and measurement settings
- `output`: Verbosity and plotting options

### Performance Targets

The codebase is optimized to achieve:
- **30-node network**: 18.5mm RMSE with 5 anchors, 25 unknowns
- **8-node network**: <10mm RMSE with 3 anchors, 5 unknowns
- Convergence within ~100 iterations using Adaptive Levenberg-Marquardt

### Output Files

Running demos generates:
- `convergence.png`: Position RMSE and time offset convergence plots
- `positions.png`: Visualization of estimated vs actual positions