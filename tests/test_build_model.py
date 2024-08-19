import unittest
import numpy as np
import tensorflow as tf
from source.build_model import build_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Assume build_model is defined in the same module or imported from another module
# from your_module import build_model

class TestBuildModel(unittest.TestCase):

    def setUp(self):
        # Initialize test parameters
        self.input_shape = (10,)
        self.params = {
            'dense_nodes': [32, 16],
            'dropout': 0.5
        }
        self.best_params = {
            'alpha': 0.01,
            'learning_rate_init': 0.001
        }

    def test_build_model(self):
        # Build the model
        model = build_model(self.input_shape, self.params, self.best_params)

        # Verify the model is a Sequential model
        self.assertIsInstance(model, Sequential, "Model is not a Sequential instance")

        # Verify the number of layers
        expected_layers = len(self.params['dense_nodes']) + 2  # Dense layers + input + output
        self.assertEqual(len(model.layers), expected_layers, f"Expected {expected_layers} layers, got {len(model.layers)}")

        # Verify first layer is an InputLayer
        self.assertIsInstance(model.layers[0], InputLayer, "First layer is not an InputLayer")

        # Verify Dense layers configuration
        for i, nodes in enumerate(self.params['dense_nodes']):
            layer = model.layers[i + 1]  # InputLayer is first
            self.assertIsInstance(layer, Dense, f"Layer {i+1} is not Dense")
            self.assertEqual(layer.units, nodes, f"Dense layer {i+1} units expected {nodes}, got {layer.units}")
            self.assertEqual(layer.activation.__name__, 'relu', f"Dense layer {i+1} activation expected 'relu', got {layer.activation.__name__}")

        # Verify Dropout layers
        for i, dropout_layer in enumerate([model.layers[j] for j in range(2, len(model.layers) - 1)]):
            self.assertIsInstance(dropout_layer, Dropout, f"Layer {i+1} is not Dropout")
            self.assertEqual(dropout_layer.rate, self.params['dropout'], f"Dropout rate expected {self.params['dropout']}, got {dropout_layer.rate}")

        # Verify final layer
        output_layer = model.layers[-1]
        self.assertIsInstance(output_layer, Dense, "Output layer is not Dense")
        self.assertEqual(output_layer.units, 1, f"Output layer units expected 1, got {output_layer.units}")
        self.assertEqual(output_layer.activation.__name__, 'sigmoid', f"Output layer activation expected 'sigmoid', got {output_layer.activation.__name__}")

        # Check optimizer
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam', "Optimizer is not Adam")
        self.assertAlmostEqual(model.optimizer.learning_rate.numpy(), self.best_params['learning_rate_init'],
                               msg=f"Optimizer learning rate expected {self.best_params['learning_rate_init']}, got {model.optimizer.learning_rate.numpy()}")

if __name__ == '__main__':
    unittest.main()
