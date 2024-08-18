import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def compare_roccurves():
  """
  To compare the different results obtained with the 3 methods.
  """
  plt.figure()
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')

  y_pred_MLP = model3.predict(X_test)
  fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_MLP)

  plt.plot(fpr1,tpr1)
  plt.plot(fpr2,tpr2)
  plt.plot(fpr3,tpr3)

  plt.legend(['MLP','kNN','LR'], frameon=False)
  plt.savefig(f"{file_dir}//HIV-CP Plots/ROC.png")

  plt.show()
