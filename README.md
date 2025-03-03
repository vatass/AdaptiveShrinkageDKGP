# Adaptive Shrinkage Deep Kernel Gaussian Process (AS-DKGP)

This repository contains the official implementation of the paper:

**"Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories"**  
*Accepted at International Conference on Learning Representations (ICLR) 2025*

## Overview

This project implements an adaptive shrinkage estimation framework for personalized deep kernel regression, specifically designed for modeling individual brain trajectories. The method combines population-level deep kernel Gaussian processes with subject-specific models using  Adaptive Shrinkage Estimation.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AdaptiveShrinkageDKGP.git
cd AdaptiveShrinkageDKGP
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The pipeline expects a CSV file with the following structure:

```
PTID,X,Y
subject1,[feature_vector_1],[target_value_1]
subject1,[feature_vector_2],[target_value_2]
subject2,[feature_vector_1],[target_value_1]
...
```

Where:
- `PTID`: Subject identifier (string)
- `X`: Input features (comma-separated values representing the feature vector)
- `Y`: Target biomarker value (float)

### Data Format Requirements:
1. Each row represents one observation for a subject
2. Multiple rows per subject are allowed (longitudinal data)
3. Features should be preprocessed and normalized
4. Missing values should be handled before creating the CSV

Example of creating the dataset:
```python
import pandas as pd
import numpy as np

# Example data structure
data = {
    'PTID': ['sub1', 'sub1', 'sub2', 'sub2', 'sub2'],
    'X': [
        [1.2, 0.5, -0.3],  # feature vector for sub1, timepoint 1
        [1.3, 0.6, -0.2],  # feature vector for sub1, timepoint 2
        [0.8, 0.4, -0.5],  # feature vector for sub2, timepoint 1
        [0.9, 0.5, -0.4],  # feature vector for sub2, timepoint 2
        [1.0, 0.6, -0.3],  # feature vector for sub2, timepoint 3
    ],
    'Y': [0.5, 0.6, 0.3, 0.4, 0.5]  # biomarker values
}

df = pd.DataFrame(data)
df.to_csv('data/biomarker_data.csv', index=False)
```

## Running the Pipeline

1. Prepare your data file as described above and place it in the `data` directory.

2. Modify the parameters in `examples/run_pipeline.py` if needed:
```python
# In main():
data_path = "data/biomarker_data.csv"  # Path to your data file
model_dir = "models"                    # Directory to save trained models
results_dir = "results"                 # Directory to save results
```

3. Run the pipeline:
```bash
python examples/run_pipeline.py
```

The pipeline will:
1. Load and preprocess the data
2. Train a population-level DKGP model
3. Train the adaptive shrinkage estimator
4. Evaluate personalization on test subjects
5. Save results and trained models

## Results

The pipeline generates:
1. Trained models in the `models/` directory:
   - `population_dkgp.pt`: Population-level DKGP model
   - `adaptive_shrinkage.json`: Adaptive shrinkage estimator

2. Evaluation results in `results/personalization_results.csv` containing:
   - Subject-wise performance metrics
   - MSE for population, subject-specific, and combined predictions
   - Performance analysis by number of observations

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{
    title={Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories},
    author={[Author Names]},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
```

## License

[Add your license information here]

## Contact

For questions about the code or paper, please open an issue or contact [your contact information].
