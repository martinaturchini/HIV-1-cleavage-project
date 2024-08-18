import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Test build_model function
def test_build_model():
    # Define test input shape and parameters
    input_shape = (10,)
    params = {
        'dense_nodes': [32, 16],
        'dropout': 0.5
    }
    best_params = {
        'alpha': 0.01,
        'learning_rate_init': 0.001
    }

    # Build model
    model = build_model(input_shape, params, best_params)

    # Verify the model is a Sequential model
    assert isinstance(model, Sequential), "Model is not a Sequential instance"

    # Verify the number of layers
    expected_layers = len(params['dense_nodes']) + 2  # Dense layers + input + output
    assert len(model.layers) == expected_layers, f"Expected {expected_layers} layers, got {len(model.layers)}"

    # Verify first layer is an InputLayer
    assert isinstance(model.layers[0], InputLayer), "First layer is not an InputLayer"

    # Verify Dense layers configuration
    for i, nodes in enumerate(params['dense_nodes']):
        layer = model.layers[i + 1]  # InputLayer is first
        assert isinstance(layer, Dense), f"Layer {i+1} is not Dense"
        assert layer.units == nodes, f"Dense layer {i+1} units expected {nodes}, got {layer.units}"
        assert layer.activation.__name__ == 'relu', f"Dense layer {i+1} activation expected 'relu', got {layer.activation.__name__}"

    # Verify Dropout layer
    for i, dropout_layer in enumerate([model.layers[j] for j in range(2, len(model.layers)-1)]):
        assert isinstance(dropout_layer, Dropout), f"Layer {i+1} is not Dropout"
        assert dropout_layer.rate == params['dropout'], f"Dropout rate expected {params['dropout']}, got {dropout_layer.rate}"

    # Verify final layer
    output_layer = model.layers[-1]
    assert isinstance(output_layer, Dense), "Output layer is not Dense"
    assert output_layer.units == 1, f"Output layer units expected 1, got {output_layer.units}"
    assert output_layer.activation.__name__ == 'sigmoid', f"Output layer activation expected 'sigmoid', got {output_layer.activation.__name__}"

    # Check optimizer
    assert model.optimizer.__class__.__name__ == 'Adam', "Optimizer is not Adam"
    assert model.optimizer.learning_rate == best_params['learning_rate_init'], \
        f"Optimizer learning rate expected {best_params['learning_rate_init']}, got {model.optimizer.learning_rate.numpy()}"

    # If no assertion errors, print success message
    print("All tests passed!")

# Run the test
test_build_model()
