import unittest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from source.build_model import build_model

class TestBuildModel(unittest.TestCase):

    def setUp(self):
        self.input_shape = (10,)
        self.params = {
            'dense_nodes': [64, 32],
            'dropout': 0.5
        }
        self.best_params = {
            'alpha': 0.001,
            'learning_rate_init': 0.01
        }

    def test_model_creation(self):
        """Test if the model is created without errors."""
        model = build_model(self.input_shape, self.params, self.best_params)
        self.assertIsInstance(model, Sequential)

    def test_model_structure(self):
        """Test if the model has the correct number of layers."""
        model = build_model(self.input_shape, self.params, self.best_params)
        # Input layer + 2 Dense layers + 2 Dropout layers + Output layer
        expected_layers = len(self.params['dense_nodes']) * 2 + 1
        self.assertEqual(len(model.layers), expected_layers)

    def test_layer_configurations(self):
        """Test if the layers are configured correctly."""
        model = build_model(self.input_shape, self.params, self.best_params)
        for i, nodes in enumerate(self.params['dense_nodes']):
            dense_layer = model.layers[i * 2]
            dropout_layer = model.layers[i * 2 + 1]

            self.assertIsInstance(dense_layer, Dense)
            self.assertEqual(dense_layer.units, nodes)
            self.assertEqual(dense_layer.activation.__name__, 'relu')
            self.assertEqual(dense_layer.kernel_regularizer.l2, self.best_params['alpha'])

            self.assertIsInstance(dropout_layer, Dropout)
            self.assertEqual(dropout_layer.rate, self.params['dropout'])

        # Check the final output layer
        output_layer = model.layers[-1]
        self.assertIsInstance(output_layer, Dense)
        self.assertEqual(output_layer.units, 1)
        self.assertEqual(output_layer.activation.__name__, 'sigmoid')

    def test_optimizer_and_compilation(self):
        """Test if the model is compiled with the correct optimizer and loss."""
        model = build_model(self.input_shape, self.params, self.best_params)
        self.assertIsInstance(model.optimizer, Adam)
        self.assertEqual(model.loss, 'binary_crossentropy')

if __name__ == '__main__':
    unittest.main()
