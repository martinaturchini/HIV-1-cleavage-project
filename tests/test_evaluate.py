import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../source')))

import unittest
from unittest.mock import patch, MagicMock
from source.separate import separate
from source.evaluate import evaluate
from source.plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

class TestEvaluateFunction(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('source.plot_confusion_matrix.plot_confusion_matrix')  # Mocking custom plot_confusion_matrix function
    @patch('source.separate.separate')  # Mocking the separate function in the evaluate module
    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.ConfusionMatrixDisplay.from_predictions')
    @patch('sklearn.metrics.auc')
    @patch('sklearn.metrics.confusion_matrix')
    @patch('sklearn.metrics.classification_report')
    def test_evaluate(self, mock_classification_report, mock_confusion_matrix, mock_auc,
                      mock_disp, mock_roc_curve, mock_separate, mock_plot_conf_matrix,
                      mock_savefig, mock_show):

        # Mocking inputs and outputs
        x_test = np.array([[0, 1], [1, 0]])  # Example test data
        Y_test = np.array([0, 1])  # Ground truth labels
        Y_pred = np.array([0.1, 0.9])  # Predicted probabilities
        Y_cls = np.array([0, 1])  # Predicted class labels

        model = MagicMock()
        model.evaluate.return_value = (0.2, 0.9)  # Mock loss and accuracy
        model.predict.return_value = Y_pred

        # Mocking return values
        mock_separate.return_value = Y_cls  # Mock class separation
        mock_classification_report.return_value = {
            'weighted avg': {'f1-score': 0.85}
        }
        mock_confusion_matrix.return_value = np.array([[1, 0], [0, 1]])  
        mock_auc.return_value = 0.95  # Mock AUC score
        mock_roc_curve.return_value = ([0.0, 0.5, 1.0], [0.0, 0.75, 1.0], None)

        file_dir = "test_directory"
        mod_num = 1
                          
        evaluate(x_test, Y_test, model, mod_num, file_dir)

        # Check if the model's evaluate and predict functions are called
        model.evaluate.assert_called_once_with(x_test, Y_test, verbose=0)
        model.predict.assert_called_once_with(x_test)

        # Verify if the plots are displayed
        mock_show.assert_called()

if __name__ == '__main__':
    unittest.main()

