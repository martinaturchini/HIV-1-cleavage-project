import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(df_results, metric_name, y_label, file_dir, file_name):
    # Initialize the plot
    plt.figure()

    # Plot the loss or accuracy curves of the three best performing models
    for idx, (row_index, row_series) in enumerate(df_results.head(3).iterrows()):
        x = np.arange(1, len(row_series['history'][f'train_{metric_name}']) + 1)
        parameter_combination_string = f"dense nodes = {row_series['dense_nodes']}; dropout={row_series['dropout']};"
        
        # Plot training and validation curves
        plt.plot(x, row_series['history'][f'train_{metric_name}'], '--', color=f'C{idx}', label=f"Training - {parameter_combination_string}")
        plt.plot(x, row_series['history'][f'val_{metric_name}'], '-', color=f'C{idx}', label=f"Validation - {parameter_combination_string}")

    # Configure plot settings
    plt.xlabel('Epochs')
    plt.xticks(np.arange(0, 32, step=5))
    plt.ylabel(y_label)
    plt.legend(frameon=False)

    # Construct the file path
    file_path = os.path.join(file_dir, "HIV-CP Plots", file_name)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the plot
    plt.savefig(file_path)
    plt.show()

    # Return the file path
    return file_path
