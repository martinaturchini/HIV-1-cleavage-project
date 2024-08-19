import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from source.plot_metrics import plot_metrics 
import os

def test_plot_metrics():
    # Define test data
    history = {
        'train_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
        'val_loss': [0.65, 0.55, 0.45, 0.35, 0.25],
        'train_accuracy': [0.6, 0.65, 0.7, 0.75, 0.8],
        'val_accuracy': [0.55, 0.6, 0.65, 0.7, 0.75]
    }
    
    df_results = pd.DataFrame({
        'history': [history, history, history],
        'dense_nodes': [[64, 32], [64, 64], [32, 32]],
        'dropout': [0.2, 0.3, 0.4]
    })
    
    global file_dir
    file_dir = os.getcwd()  # Use current working directory for testing

    # Define test parameters
    metric_name = 'loss'
    y_label = 'Loss'
    file_name = 'Test_Loss_Plot.png'

    # Call the function with test data
    file_path = plot_metrics(df_results, metric_name, y_label, file_name)

    # Verify that the plot was saved
    assert os.path.exists(file_path), f"File {file_path} was not created."
    print(f"File {file_path} created successfully!")

    # Clean up: Remove the file after verification
    os.remove(file_path)
    print(f"File {file_path} has been removed after the test.")
