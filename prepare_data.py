import pandas as pd
import numpy as np

# Read the data
print("Loading data...")
data = pd.read_csv('data/biomarker_data.csv')

# Separate features
roi_cols = [f'ROI_{i}' for i in range(1, 146)]
covariate_cols = [f'Covariate{i}' for i in range(1, 6)]

# Create feature vector X by combining ROIs, covariates, and time
print("Creating feature vectors...")
data['X'] = data.apply(
    lambda row: np.concatenate([
        row[roi_cols].values,           # ROI features
        row[covariate_cols].values,     # Clinical covariates
        [row['Time']]                   # Time point
    ]),
    axis=1
)

# Save processed data
print("Saving processed data...")
processed_data = data[['PTID', 'X']].copy()
processed_data.to_csv('data/biomarker_data_processed.csv', index=False)
print("Done! Data saved to data/biomarker_data_processed.csv") 