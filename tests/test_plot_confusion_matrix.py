import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
from source.plot_confusion_matrix import plot_confusion_matrix 

class TestPlotConfusionMatrix(unittest.TestCase):

    def setUp(self):
        # Initialize test data and directory for saving files
        self.cm = np.array([[10, 2, 1],
                            [3, 7, 0],
                            [1, 2, 8]])
        self.classes = ['Class A', 'Class B', 'Class C']
        self.method = 'TestMethod'
        self.mod_num = '1'
        self.file_name = 'Test_Confusion_Matrix.png'
        self.file_path = os.path.join(os.getcwd(), self.file_name)  # Path to save plot

    def tearDown(self):
        # Clean up any files created during the test
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_plot_confusion_matrix(self):
        # Call the function with test data
        file_path = plot_confusion_matrix(self.cm, self.classes, self.method, normalize=False, title='Test Confusion Matrix', mod_num=self.mod_num)

        # Verify that the plot was saved
        self.assertTrue(os.path.exists(file_path), f"File {file_path} was not created.")
        print(f"File {file_path} created successfully!")

if __name__ == '__main__':
    unittest.main()

