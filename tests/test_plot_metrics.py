import unittest
import os
import pandas as pd
import shutil
from source.plot_metrics import plot_metrics 

class TestPlotMetrics(unittest.TestCase):

    def setUp(self):
        # Setup any necessary variables or state before each test
        self.history = {
            'train_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
            'val_loss': [0.65, 0.55, 0.45, 0.35, 0.25],
            'train_accuracy': [0.6, 0.65, 0.7, 0.75, 0.8],
            'val_accuracy': [0.55, 0.6, 0.65, 0.7, 0.75]
        }

        self.df_results = pd.DataFrame({
            'history': [self.history, self.history, self.history],
            'dense_nodes': [[64, 32], [64, 64], [32, 32]],
            'dropout': [0.2, 0.3, 0.4]
        })

        # Set the directory for saving the file
        self.file_dir = "test_output"
        if not os.path.exists(self.file_dir):
            os.makedirs(f"{self.file_dir}//HIV-CP Plots")

    def tearDown(self):
        # Clean up any files created during the test
        if os.path.exists(self.file_dir):
            shutil.rmtree(self.file_dir)

    def test_plot_metrics(self):
        # Define test parameters
        metric_name = 'loss'
        y_label = 'Loss'
        file_name = 'Test_Loss_Plot.png'

        # Call the function with test data
        file_path = plot_metrics(self.df_results, metric_name, y_label, self.file_dir, file_name)
        
        # Debugging print statements to check the path
        print(f"Expected file path: {file_path}")

        # Verify that the plot was saved
        self.assertTrue(os.path.exists(file_path), f"File {file_path} was not created.")
        print(f"File {file_path} created successfully!")

if __name__ == '__main__':
    unittest.main()
