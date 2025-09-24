# Comprehensive Noise Model for TL System

## Overview

The TL system now includes a comprehensive, configurable noise model that simulates realistic UWB ranging conditions. This allows testing system performance under various environmental conditions while maintaining the ability to run in ideal (noise-free) mode for algorithm development.

## Quick Start

### Using Noise Presets

```python
# Test with different noise levels
python test_30node_system.py ideal       # ~18.5mm RMSE (no noise)
python test_30node_system.py clean       # ~50mm RMSE (lab conditions)
python test_30node_system.py realistic   # ~1.7m RMSE (typical deployment)
python test_30node_system.py harsh       # ~3m RMSE (challenging environment)

# Compare all presets
python test_30node_system.py --compare
```

### Running Analysis

```bash
# Comprehensive noise impact analysis
python analyze_noise_impact.py

# Validate noise model
python test_noise_validation.py
```

## Noise Sources

### 1. Measurement Noise

#### Thermal Noise (AWGN)
- Based on SNR and system bandwidth
- Configurable noise figure
- Models receiver thermal noise
- **Parameters:**
  - `snr_db`: Signal-to-noise ratio (dB)
  - `bandwidth_mhz`: System bandwidth
  - `noise_figure_db`: Receiver noise figure

#### Distance-Dependent Noise
- Noise increases with distance
- Models propagation effects
- **Parameters:**
  - `coefficient`: Scaling factor (std = coef × √distance)
  - `min_std_m`: Minimum standard deviation

#### Quantization Noise
- ADC resolution limits
- Uniform quantization error
- **Parameters:**
  - `adc_bits`: ADC resolution
  - `full_scale_range_m`: ADC full scale range

#### Clock Jitter
- Timing uncertainty
- Allan deviation model
- **Parameters:**
  - `allan_deviation_ps`: Allan deviation in picoseconds
  - `integration_time_s`: Integration time

### 2. Channel Effects

#### Multipath/NLOS
- Positive range bias for NLOS
- Increased variance
- **Parameters:**
  - `nlos_probability`: Probability of NLOS condition
  - `bias_range_m`: Range of positive bias
  - `excess_std_factor`: Additional variance factor

#### Shadowing
- Large-scale fading
- Log-normal distribution
- **Parameters:**
  - `std_db`: Standard deviation in dB
  - `correlation_distance_m`: Decorrelation distance

#### Small-Scale Fading
- Rician/Rayleigh fading
- For dynamic environments
- **Parameters:**
  - `k_factor_db`: Rician K-factor
  - `enabled`: Usually disabled for static scenarios

### 3. Hardware Imperfections

#### Frequency Offset
- Crystal accuracy limitations
- Creates distance-proportional bias
- **Parameters:**
  - `max_offset_ppb`: Maximum offset in parts per billion
  - `temperature_coefficient_ppb_per_c`: Temperature dependency

#### Phase Noise
- Oscillator stability
- Affects range accuracy
- **Parameters:**
  - `phase_noise_dbc_per_hz`: Phase noise at 1kHz offset
  - `corner_frequency_hz`: Corner frequency

#### Antenna Delay
- Calibration errors
- Temperature-dependent drift
- **Parameters:**
  - `calibration_error_std_ps`: Calibration error standard deviation
  - `temperature_coefficient_ps_per_c`: Temperature coefficient

### 4. Environmental Factors

#### Node Motion
- Position uncertainty from movement
- Velocity-based model
- **Parameters:**
  - `velocity_std_mps`: Velocity standard deviation
  - `max_velocity_mps`: Maximum velocity

#### Doppler Shift
- Frequency shifts from motion
- Affects ranging accuracy
- **Parameters:**
  - `max_velocity_mps`: Maximum velocity
  - `carrier_freq_ghz`: Carrier frequency

#### Interference
- Other RF sources
- Burst interference model
- **Parameters:**
  - `sinr_db`: Signal-to-interference ratio
  - `burst_probability`: Probability of interference

## Noise Presets

### IDEAL
- **Purpose:** Algorithm development and baseline testing
- **Performance:** ~18.5mm RMSE (30-node system)
- **Configuration:** All noise disabled
- **Use Case:** Verifying algorithm correctness

### CLEAN
- **Purpose:** Lab/controlled environment testing
- **Performance:** ~50mm RMSE
- **Configuration:**
  - High SNR (30dB)
  - Minimal multipath (disabled)
  - Low frequency offset (2ppb)
  - High-resolution ADC (14-bit)
- **Use Case:** Ideal deployment conditions

### REALISTIC
- **Purpose:** Typical UWB deployment
- **Performance:** ~1.7m RMSE
- **Configuration:**
  - Moderate SNR (20dB)
  - 10% NLOS probability
  - 10ppb frequency offset
  - Standard 12-bit ADC
- **Use Case:** Real-world indoor/outdoor deployment

### HARSH
- **Purpose:** Stress testing
- **Performance:** ~3m RMSE
- **Configuration:**
  - Low SNR (10dB)
  - 30% NLOS probability
  - 50ppb frequency offset
  - Environmental factors enabled
- **Use Case:** Challenging environments, robustness testing

## Configuration

### Python API

```python
from ftl.noise_model import NoiseConfig, create_preset_config

# Use preset
noise_config = create_preset_config("realistic")

# Custom configuration
noise_config = NoiseConfig(
    preset=None,
    enable_noise=True,
    random_seed=42,  # For reproducibility
    thermal=ThermalNoiseConfig(
        enabled=True,
        snr_db=25.0
    ),
    multipath=MultipathConfig(
        enabled=True,
        nlos_probability=0.15
    )
)

# Use with TL system
ftl = Large30NodeFTL(ftl_config, noise_config)
```

### YAML Configuration

```yaml
noise:
  preset: "realistic"
  enable_noise: true
  random_seed: 42

  thermal:
    enabled: true
    snr_db: 20.0
    bandwidth_mhz: 500.0

  multipath:
    enabled: true
    nlos_probability: 0.1
    bias_range_m: [0.0, 0.5]
```

## Performance Impact

### Expected RMSE by Preset (30-node, 5-anchor system)

| Preset | Position RMSE | Time RMSE | Notes |
|--------|--------------|-----------|--------|
| IDEAL | 18.5mm | 4ps | No noise, original performance |
| CLEAN | 20-50mm | 10-50ps | Lab conditions |
| REALISTIC | 1-2m | 1-3ns | Typical deployment |
| HARSH | 2-5m | 3-5ns | Challenging environment |

### Key Noise Source Impacts

1. **Thermal Noise (SNR)**
   - SNR > 30dB: < 50mm impact
   - SNR = 20dB: ~500mm impact
   - SNR = 10dB: > 1m impact

2. **Multipath/NLOS**
   - 10% NLOS: ~200mm additional error
   - 20% NLOS: ~500mm additional error
   - 30% NLOS: > 1m additional error

3. **Frequency Offset**
   - 10ppb: ~100mm at 30m range
   - 50ppb: ~500mm at 30m range
   - 100ppb: > 1m at 30m range

## Analysis Tools

### Noise Validation
```bash
python test_noise_validation.py
```
- Tests individual noise sources
- Verifies statistical properties
- Compares with theoretical bounds
- Generates distribution visualizations

### Impact Analysis
```bash
python analyze_noise_impact.py
```
- Parameter sweeps (SNR, NLOS, frequency)
- Performance vs noise curves
- Combined effects analysis
- Saves results to JSON

### Comparison Testing
```bash
python test_30node_system.py --compare
```
- Tests all presets
- Side-by-side performance comparison
- Convergence analysis

## Best Practices

### For Algorithm Development
1. Start with IDEAL preset to verify correctness
2. Test with CLEAN to ensure basic robustness
3. Validate with REALISTIC for deployment readiness
4. Stress test with HARSH for edge cases

### For Performance Evaluation
1. Use fixed random seeds for reproducibility
2. Run multiple trials with different seeds
3. Report percentiles (50th, 90th, 95th) not just mean
4. Test with gradual noise increase

### For Real Deployment
1. Characterize actual environment noise
2. Create custom preset matching conditions
3. Validate with field measurements
4. Include margin for worst-case conditions

## Theoretical Bounds

The system performance is bounded by the Cramér-Rao Lower Bound (CRLB):

```
CRLB = GDOP × σ_measurement / √(n_anchors)
```

Where:
- GDOP: Geometric Dilution of Precision (~1.4 for good geometry)
- σ_measurement: Measurement standard deviation
- n_anchors: Number of anchor nodes

Example for 5 anchors, 10mm measurement noise:
```
CRLB = 1.4 × 0.01 / √5 = 6.3mm
```

## Extending the Noise Model

### Adding New Noise Sources

1. Add configuration class in `ftl/noise_model.py`:
```python
@dataclass
class NewNoiseConfig:
    enabled: bool = True
    parameter1: float = 1.0
```

2. Add to NoiseConfig:
```python
new_noise: NewNoiseConfig = field(default_factory=NewNoiseConfig)
```

3. Implement in NoiseGenerator.add_measurement_noise():
```python
if self.config.new_noise.enabled:
    noise_contribution = self._compute_new_noise(...)
    noisy_range += noise_contribution
```

### Creating Custom Presets

1. Create YAML file in `configs/`:
```yaml
noise:
  preset: null  # Custom
  thermal:
    snr_db: 25.0
  # ... other settings
```

2. Or in Python:
```python
custom_config = NoiseConfig(
    preset=None,
    # ... custom settings
)
```

## Troubleshooting

### High RMSE with Noise
- Check if noise levels are appropriate for your scenario
- Verify measurement covariance is properly estimated
- Consider increasing iterations or relaxing convergence criteria

### Non-reproducible Results
- Set `random_seed` in NoiseConfig
- Ensure same seed across all tests
- Check for other sources of randomness

### Convergence Issues
- Reduce noise levels temporarily
- Increase `max_iterations`
- Adjust optimizer parameters (lambda, gradient tolerance)

## References

1. IEEE 802.15.4a Channel Model for UWB
2. Cramér-Rao Lower Bound for Localization
3. Allan Variance for Clock Stability
4. Saleh-Valenzuela Multipath Model

## Future Enhancements

- [ ] Dynamic noise adaptation based on SNR estimation
- [ ] Machine learning-based noise parameter estimation
- [ ] Integration with real channel measurements
- [ ] Outlier rejection for NLOS mitigation
- [ ] Advanced multipath resolution algorithms