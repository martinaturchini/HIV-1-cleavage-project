import unittest
import os
import pandas as pd
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

        self.file_dir = os.getcwd()  # Use current working directory for testing

    def tearDown(self):
        # Clean up any files created during the test
        file_path = os.path.join(self.file_dir, 'Test_Loss_Plot.png')
        if os.path.exists(file_path):
            os.remove(file_path)

    def test_plot_metrics(self):
        # Define test parameters
        metric_name = 'loss'
        y_label = 'Loss'
        file_name = 'Test_Loss_Plot.png'

        # Call the function with test data
        file_path = plot_metrics(self.df_results, metric_name, y_label, file_dir, file_name)

        # Verify that the plot was saved
        self.assertTrue(os.path.exists(file_path), f"File {file_path} was not created.")
        print(f"File {file_path} created successfully!")

if __name__ == '__main__':
    unittest.main()
