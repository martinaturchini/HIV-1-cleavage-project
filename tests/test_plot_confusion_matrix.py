import numpy as np
import os
import shutil
from source.plot_confusion_matrix import plot_confusion_matrix

def test_plot_confusion_matrix():
    # Create a sample confusion matrix
    cm = np.array([[50, 2, 1],
                   [10, 30, 5],
                   [2, 1, 35]])

    # Define class names
    classes = ['Class A', 'Class B', 'Class C']

    # Define method name
    method = 'Test Method'

    # Set the directory for saving the file
    file_dir = "test_output"
    if not os.path.exists(file_dir):
        os.makedirs(f"{file_dir}//HIV-CP Plots")
    
    # Call the function
    plot_confusion_matrix(file_dir=file_dir, cm=cm, classes=classes, method=method)

    # Check if the file was created
    file_path = f"{file_dir}//HIV-CP Plots/Confusion matrix {method}.png"
    assert os.path.exists(file_path), "Confusion matrix plot file was not created."

    print(f"Test passed: Confusion matrix plot saved at {file_path}")

    # Clean up
    shutil.rmtree(file_dir)
