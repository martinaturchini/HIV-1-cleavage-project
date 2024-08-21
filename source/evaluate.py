import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from separate import separate

def evaluate(x_test, Y_test, model, mod_num):

    """
    This function evaluate loss and metric, computes the confusion matrix,
    ROC curve and plots them
    """

    #Evaluate loss and metrics
    loss, accuracy = model.evaluate(x_test, Y_test, verbose=0)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    # Predict the values from the test dataset
    Y_pred = model.predict(x_test)

    # Classify predictions in 0 (for values <= 0.5)or 1
    Y_cls = separate(Y_pred)

    # Generate classification report
    cr = classification_report(Y_test, Y_cls, output_dict=True)
    print('Classification Report:\n', classification_report(Y_test,Y_cls))

    # Compute confusion matrix
    confusion_mtx = confusion_matrix(Y_test, Y_cls)
    tn, fp, fn, tp = confusion_mtx.ravel()

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f1 = cr['weighted avg']['f1-score']

    # Compute ROC curve and AUC. The "thresholds" value (3rd) is not being used,
    # so it can be substitued by an underscore "_" to improve readability and 
    # prevent warnings.
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)

    # Print key metrics
    print('AUC:', auc(fpr,tpr))
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    print('Average of performance metrics:',
          (accuracy+sensitivity+specificity+auc(fpr,tpr)+f1)/5)

    # Plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = (0,1), method="MLP ",
                          mod_num=mod_num)

    # Display the normalized confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(Y_test,Y_cls,
                                                   display_labels=(0,1),
                                                   cmap=plt.cm.Blues,
                                                   normalize='true')

    disp.ax_.set_title(f"Normalized confusion matrix MLP {mod_num}")

    plt.savefig(f"{file_dir}//HIV-CP Plots/Normalized confusion matrix MLP_{mod_num}.png")
    plt.show()

    # Plot the ROC curve
    plt.plot(fpr,tpr)
    plt.title(f'ROC Curve - MLP {mod_num}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
