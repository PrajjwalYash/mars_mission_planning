import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error

def evaluate_and_plot(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y):
    """Train the model, evaluate it, and plot the results."""
    # Train the model
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    best_model = model(X_train=X_train, y_train=y_train)
    
    # Predict and evaluate on training data
    train_predict = best_model.predict(X_train)
    train_mae = mean_absolute_error(y_true=y_train, y_pred=train_predict)
    train_r2 = r2_score(y_true=y_train, y_pred=train_predict)
    print(f"{model_name} Training MAE:", train_mae)
    print(f"{model_name} Training R2:", train_r2)
    print('*' * 20)
    
    # Predict and evaluate on validation data
    val_predict = best_model.predict(X_val)
    val_mae = mean_absolute_error(y_true=y_val, y_pred=val_predict)
    val_r2 = r2_score(y_true=y_val, y_pred=val_predict)
    print(f"{model_name} Validation MAE:", val_mae)
    print(f"{model_name} Validation R2:", val_r2)
    print('*' * 20)
    
    # Predict and evaluate on test data
    test_predict = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_true=y_test, y_pred=test_predict)
    test_r2 = r2_score(y_true=y_test, y_pred=test_predict)
    print(f"{model_name} Test MAE:", test_mae)
    print(f"{model_name} Test R2:", test_r2)
    
    # Unscale the test data for visualization
    y_test_unscaled = (scaler_y.inverse_transform(y_test.reshape(-1, 1)))
    test_predict_unscaled = (scaler_y.inverse_transform(test_predict.reshape(-1, 1)))
    
    # Plot predicted vs actual values
    plt.figure(figsize=(20, 12))
    plt.scatter(y_test_unscaled, test_predict_unscaled, label=f'Predicted vs Observed ({model_name})')
    max_val = max(y_test_unscaled.max(), test_predict_unscaled.max())
    min_val = min(y_test_unscaled.min(), test_predict_unscaled.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label=f'R2_score = {np.round(test_r2,2)}')
    plt.xlabel('Log of Observed dust deposition rate (in kgm$^{-2}$s$^{-1}$)', fontsize = 20)
    plt.ylabel('Log of Predicted dust deposition rate (in kgm$^{-2}$s$^{-1}$)', fontsize = 20)
    plt.title(f'Predicted vs Actual Values ({model_name})', fontsize = 30)
    plt.legend(fontsize = 20)
    plt.grid()
    plt.tight_layout()
    output_path = os.path.join(parent_directory, 'outputs', model_name + '_prediction_vs_observation_DD_rate.png')
    plt.savefig(output_path)
    
    # Return the evaluation metrics
    return {
        'Model': model_name,
        'Train MAE': train_mae,
        'Train R2': train_r2,
        'Validation MAE': val_mae,
        'Validation R2': val_r2,
        'Test MAE': test_mae,
        'Test R2': test_r2
    }


def save_performance_and_plot(df_performance):
    """
    Save the model performance metrics to a CSV file and generate bar plots for R2 and MAE.
    
    Args:
    - df_performance (pd.DataFrame): DataFrame containing the performance metrics for each model.
    """
    # Ensure the output directory exists
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory = os.path.join(parent_directory, 'outputs')
    
    # Save the performance metrics to a CSV file
    performance_output_path = os.path.join(output_directory, 'model_performance_comparison.csv')
    df_performance.to_csv(performance_output_path, index=False)
    
    # Define bar plot parameters
    metrics = ['R2', 'MAE']
    stages = ['Train', 'Validation', 'Test']
    colors = ['skyblue', 'lightgreen', 'salmon']
    bar_width = 0.25
    bar_positions = np.arange(len(df_performance))

    # Plot comparison of R2 and MAE across models using subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))

    for i, metric in enumerate(metrics):
        # Define the column names for the metric
        metric_columns = [f'{stage} {metric}' for stage in stages]
        
        # Plot the bars
        for j, (stage, color) in enumerate(zip(stages, colors)):
            axs[i].bar(bar_positions + j * bar_width,
                       df_performance[metric_columns[j]],
                       width=bar_width,
                       label=f'{stage} {metric}',
                       color=color)

        # Set the titles, labels, and legends
        axs[i].set_title(f'{metric} Score Comparison')
        axs[i].set_xlabel('Model')
        axs[i].set_ylabel(metric)
        axs[i].set_xticks(bar_positions + bar_width)
        axs[i].set_xticklabels(df_performance['Model'])
        axs[i].legend()
        axs[i].grid(True)

    # Set the main title for the entire plot
    fig.suptitle('Model Performance Comparison (R2 and MAE)', fontsize=16)
    fig.tight_layout()
    # Save the figure
    comparison_plot_path = os.path.join(output_directory, 'model_performance_comparison.png')
    plt.savefig(comparison_plot_path)