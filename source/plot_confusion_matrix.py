import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from itertools import product

def plot_confusion_matrix(cm, classes, method,
                          normalize=False,
                          title='Confusion matrix ',
                          cmap=plt.cm.Blues, mod_num=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + method + mod_num)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(f"{file_dir}//HIV-CP Plots/Confusion matrix "+
                method + mod_num + ".png")
    plt.show()
