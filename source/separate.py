import numpy as np

def separate(Y_pred):
    Y_cls = np.where(Y_pred <= 0.5, 0, 1)
    return Y_cls
