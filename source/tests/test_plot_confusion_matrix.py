import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import os

def test_plot_confusion_matrix():
    # Define test data
    cm = np.array([[10, 2, 1],
                   [3, 7, 0],
                   [1, 2, 8]])
    classes = ['Class A', 'Class B', 'Class C']
    method = 'TestMethod'
    mod_num = '1'

    # Call the function with test data
    file_path = plot_confusion_matrix(cm, classes, method, normalize=False, title='Test Confusion Matrix ', mod_num=mod_num)

    # Verify that the plot was saved
    assert os.path.exists(file_path), f"File {file_path} was not created."
    print(f"File {file_path} created successfully!")

    # Optionally: Add image comparison here if needed

    # Clean up: Remove the file after verification
    os.remove(file_path)
    print(f"File {file_path} has been removed after the test.")

# Run the test
test_plot_confusion_matrix()
