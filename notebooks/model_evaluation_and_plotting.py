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
    print(f"{model_name} Training MAE:", mean_absolute_error(y_true=y_train, y_pred=train_predict))
    print(f"{model_name} Training R2:", r2_score(y_true=y_train, y_pred=train_predict))
    print('*' * 20)
    
    # Predict and evaluate on validation data
    val_predict = best_model.predict(X_val)
    print(f"{model_name} Validation MAE:", mean_absolute_error(y_true=y_val, y_pred=val_predict))
    print(f"{model_name} Validation R2:", r2_score(y_true=y_val, y_pred=val_predict))
    print('*' * 20)
    
    # Predict and evaluate on test data
    test_predict = best_model.predict(X_test)
    print(f"{model_name} Test MAE:", mean_absolute_error(y_true=y_test, y_pred=test_predict))
    print(f"{model_name} Test R2:", r2_score(y_true=y_test, y_pred=test_predict))
    
    # Unscale the test data for visualization
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    test_predict_unscaled = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
    
    # Plot predicted vs actual values
    plt.figure(figsize=(20, 12))
    plt.scatter(y_test_unscaled, test_predict_unscaled, label=f'Predicted vs Actual ({model_name})')
    max_val = max(y_test_unscaled.max(), test_predict_unscaled.max())
    min_val = min(y_test_unscaled.min(), test_predict_unscaled.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='y = x line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values ({model_name})')
    plt.legend()
    output_path = os.path.join(parent_directory, 'outputs', model_name+'_prediction_vs_observation_DD_rate.png')
    plt.savefig(output_path)