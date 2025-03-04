import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pathlib import Path
import torch
from models.adaptive_dkgp import AdaptiveDKGP

def plot_trajectory_with_uncertainty(model, patient_data, time_points, save_path=None):
    """
    Plot trajectory with uncertainty bands for a single patient.
    
    Args:
        model: Trained AdaptiveDKGP model
        patient_data: DataFrame containing patient data
        time_points: Array of time points to predict
        save_path: Path to save the plot (optional)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Prepare input data
    X = torch.tensor(patient_data['X'].values[0], dtype=torch.float32)
    X_repeated = X.repeat(len(time_points), 1)
    
    # Update time points
    time_tensor = torch.tensor(time_points, dtype=torch.float32).reshape(-1, 1)
    X_repeated[:, -1] = time_tensor.squeeze()
    
    # Get predictions with uncertainty
    with torch.no_grad():
        predictions = model(X_repeated)
        mean = predictions.mean.numpy()
        lower, upper = predictions.confidence_region()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, mean, 'b-', label='Predicted trajectory')
    plt.fill_between(time_points, lower, upper, color='b', alpha=0.2, label='95% CI')
    
    # Add actual observations if available
    actual_times = patient_data['Time'].values
    actual_values = patient_data['Y'].values
    plt.scatter(actual_times, actual_values, c='r', label='Actual observations')
    
    plt.xlabel('Time (months)')
    plt.ylabel('Biomarker Value')
    plt.title(f'Patient Trajectory (ID: {patient_data["PTID"].iloc[0]})')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_trajectory_animation(model, patient_data, time_points, save_path):
    """
    Create an animated GIF of trajectory adaptation over time.
    
    Args:
        model: Trained AdaptiveDKGP model
        patient_data: DataFrame containing patient data
        time_points: Array of time points to predict
        save_path: Path to save the GIF
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        current_time = time_points[frame]
        
        # Get predictions up to current time
        current_times = time_points[:frame+1]
        X = torch.tensor(patient_data['X'].values[0], dtype=torch.float32)
        X_repeated = X.repeat(len(current_times), 1)
        time_tensor = torch.tensor(current_times, dtype=torch.float32).reshape(-1, 1)
        X_repeated[:, -1] = time_tensor.squeeze()
        
        with torch.no_grad():
            predictions = model(X_repeated)
            mean = predictions.mean.numpy()
            lower, upper = predictions.confidence_region()
        
        # Plot trajectory and uncertainty
        ax.plot(current_times, mean, 'b-', label='Predicted trajectory')
        ax.fill_between(current_times, lower, upper, color='b', alpha=0.2, label='95% CI')
        
        # Plot actual observations up to current time
        mask = patient_data['Time'] <= current_time
        actual_times = patient_data.loc[mask, 'Time'].values
        actual_values = patient_data.loc[mask, 'Y'].values
        ax.scatter(actual_times, actual_values, c='r', label='Actual observations')
        
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Biomarker Value')
        ax.set_title(f'Patient Trajectory Over Time\nTime: {current_time:.1f} months')
        ax.legend()
        
        # Set consistent axis limits
        ax.set_xlim(time_points.min(), time_points.max())
        y_min = min(patient_data['Y'].min(), lower.min())
        y_max = max(patient_data['Y'].max(), upper.max())
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    
    anim = FuncAnimation(fig, update, frames=len(time_points), 
                        interval=200, repeat=False)
    anim.save(save_path, writer='pillow')
    plt.close()

def visualize_all_patients(model_path, data_path, output_dir='results/visualizations'):
    """
    Generate visualizations for all patients in the dataset.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the processed data
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model = torch.load(model_path)
    data = pd.read_csv(data_path)
    
    # Generate time points for prediction
    time_points = np.linspace(0, 60, 100)  # 5 years with 100 points
    
    # Generate visualizations for each patient
    for patient_id in data['PTID'].unique():
        patient_data = data[data['PTID'] == patient_id]
        
        # Static plot
        plot_path = output_dir / f'patient_{patient_id}_trajectory.png'
        plot_trajectory_with_uncertainty(model, patient_data, time_points, plot_path)
        
        # Animated GIF
        gif_path = output_dir / f'patient_{patient_id}_animation.gif'
        create_trajectory_animation(model, patient_data, time_points, gif_path)

if __name__ == '__main__':
    # Example usage
    model_path = 'models/trained_model.pt'
    data_path = 'data/biomarker_data_processed.csv'
    visualize_all_patients(model_path, data_path) 