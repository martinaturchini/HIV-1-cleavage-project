import matplotlib.pyplot as plt

def plot_history(network_history, mod_num, file_dir):
 """
    Plotting loss and accuracy values obtained for each epoch.
 """
 plt.figure()
 plt.xlabel('Epochs')
 plt.ylabel('Loss')
 plt.plot(network_history.history['loss'], '--')
 plt.plot(network_history.history['val_loss'], '-')
 plt.legend(['Training', 'Validation'],
              frameon=False)
 plt.savefig(f"{file_dir}//HIV-CP Plots/Loss MLP "+mod_num+".png")

 plt.figure()
 plt.xlabel('Epochs')
 plt.ylabel('Accuracy')
 plt.plot(network_history.history['accuracy'], '--')
 plt.plot(network_history.history['val_accuracy'], '-')
 plt.legend(['Training', 'Validation'],
             loc='lower right', frameon=False)
 plt.savefig(f"{file_dir}//HIV-CP Plots/Accuracy MLP "+mod_num+".png")

 plt.show()
