import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Optimizer, Adam
from keras.layers import Dense, InputLayer, Dropout
from keras.regularizers import l2

def build_model(input_shape, params, best_params):
    """
      Create a model with the best parameters.
    """
    model = Sequential()
    model.add(InputLayer(input_shape))

    for nodes in params['dense_nodes']:
        model.add(Dense(nodes, activation='relu', kernel_regularizer=l2(best_params['alpha'])))
        model.add(Dropout(params['dropout']))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=best_params['learning_rate_init'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
