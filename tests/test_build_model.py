import unittest
from tensorflow.keras import Model
from source.build_model import build_model

class TestBuildModel(unittest.TestCase):

    def setUp(self):
        # This will run before each test
        self.input_shape = (10,)
        self.params = {
            'dense_nodes': [64, 32],
            'dropout': 0.5
        }
        self.best_params = {
            'alpha': 0.01,
            'learning_rate_init': 0.001
        }

    def test_model_structure(self):
        model = build_model(self.input_shape, self.params, self.best_params)
        
        # Check if the model is an instance of keras Model
        self.assertIsInstance(model, Model)
        
        # Check the number of layers (2 Dense layers + 2 Dropout layers + 1 output layer)
        self.assertEqual(len(model.layers), len(self.params['dense_nodes']) * 2 + 1)
        
        # Check the first Dense layer has correct number of nodes
        self.assertEqual(model.layers[1].units, self.params['dense_nodes'][0])
        
        # Check the activation function of the first Dense layer
        self.assertEqual(model.layers[1].activation.__name__, 'relu')
        
        # Check that dropout is correctly applied
        self.assertEqual(model.layers[2].rate, self.params['dropout'])
        
        # Check the final layer has one unit and sigmoid activation
        self.assertEqual(model.layers[-1].units, 1)
        self.assertEqual(model.layers[-1].activation.__name__, 'sigmoid')

    def test_model_compilation(self):
        model = build_model(self.input_shape, self.params, self.best_params)
        
        # Check if the model is compiled with the right loss function
        self.assertEqual(model.loss, 'binary_crossentropy')
        
        # Check if the optimizer is an instance of Adam and has the correct learning rate
        self.assertIsInstance(model.optimizer, Adam)
        self.assertEqual(model.optimizer.learning_rate.numpy(), self.best_params['learning_rate_init'])
        
        # Check if 'accuracy' is in the metrics
        self.assertIn('accuracy', model.metrics_names)

    def test_edge_case_empty_dense_nodes(self):
        # Test case where dense_nodes list is empty
        params = {'dense_nodes': [], 'dropout': 0.5}
        model = build_model(self.input_shape, params, self.best_params)
        
        # The model should only contain the input layer and the final output layer
        self.assertEqual(len(model.layers), 1)

    def test_invalid_input_shape(self):
        with self.assertRaises(ValueError):
            build_model(None, self.params, self.best_params)

    def test_invalid_params(self):
        invalid_params = {'dense_nodes': 'invalid_type', 'dropout': 'invalid_type'}
        with self.assertRaises(TypeError):
            build_model(self.input_shape, invalid_params, self.best_params)

    def test_invalid_best_params(self):
        invalid_best_params = {'alpha': 'invalid_type', 'learning_rate_init': 'invalid_type'}
        with self.assertRaises(TypeError):
            build_model(self.input_shape, self.params, invalid_best_params)

if __name__ == '__main__':
    unittest.main()
