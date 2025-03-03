# Adaptive Shrinkage Deep Kernel Gaussian Process (AS-DKGP)

A Python package for implementing Deep Kernel Gaussian Process regression with adaptive shrinkage estimation for personalized trajectory prediction.

## Installation

```bash
pip install adaptive-shrinkage-dkgp
```

## Components

1. Population Deep Kernel GP (p-DKGP)
   - Trains on population data
   - Stores deep network parameters and GP weights

2. Subject-Specific Deep Kernel GP (ss-DKGP)
   - Personalizes predictions for individual subjects
   - Initialized with p-DKGP parameters

3. Adaptive Shrinkage Estimator
   - Calculates oracle shrinkage through optimization
   - Uses XGBoost for regression
   - Combines p-DKGP and ss-DKGP predictions

## Usage

### Training the Models

```python
from adaptive_shrinkage_dkgp.models import PopulationDKGP, SubjectSpecificDKGP, AdaptiveShrinkage

# Train population model
p_dkgp = PopulationDKGP()
p_dkgp.fit(X_train, y_train)

# Train adaptive shrinkage
shrinkage = AdaptiveShrinkage()
shrinkage.fit(X_val, y_val)

# Personalization for a specific subject
ss_dkgp = SubjectSpecificDKGP()
ss_dkgp.fit(X_subject, y_subject)
```

See the `examples/demo.py` for a complete demonstration.

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@article{your-paper,
    title={Your Paper Title},
    author={Your Name},
    journal={Journal Name},
    year={2024}
}
```
