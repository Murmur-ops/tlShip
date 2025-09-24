# Time-Localization (TL) System

Production-ready distributed localization and time synchronization achieving **18.5mm RMSE** accuracy.

## Performance

- **30-node network**: 18.5mm RMSE (5 anchors, 25 unknowns)
- **8-node network**: <10mm RMSE (3 anchors, 5 unknowns)
- **Convergence**: ~100 iterations
- **Algorithm**: Adaptive Levenberg-Marquardt

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Demo

```bash
# Default 30-node system (achieves 18.5mm)
python run_demo.py

# Using YAML configuration
python run_demo.py --config configs/example_30node.yaml

# 8-node system
python run_demo.py --config configs/example_8node.yaml
```

### Outputs

The demo generates two visualization files:
1. **convergence.png**: Shows position RMSE and time offset convergence
2. **positions.png**: Shows estimated vs actual positions

## Files

- `ftl_enhanced.py`: Main FTL solver with Adaptive LM
- `test_30node_system.py`: 30-node test that achieves 18.5mm
- `run_demo.py`: Demo script with YAML support and visualizations
- `ftl/`: Core optimization algorithms (Adaptive LM, line search)
- `configs/`: Example YAML configurations

## YAML Configuration

```yaml
network:
  n_nodes: 30        # Total nodes
  n_anchors: 5       # Known positions

optimization:
  use_adaptive_lm: true
  max_iterations: 100
  lm_initial_lambda: 0.001
  gradient_tol: 1.0e-8

measurements:
  measurement_std: 0.01  # 1cm noise

output:
  verbose: true
  plot_results: true
```

## Verified Performance

```
30-Node Network Test
Position RMSE: 18.539 mm
Time RMSE: 4.145 ps
```

This is the exact working code that achieves the published performance.