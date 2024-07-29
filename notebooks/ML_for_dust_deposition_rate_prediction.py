import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error
from ML_data_creation import *
from ensembled_tree_models import *
from model_evaluation_and_plotting import *
# Define the sites for fetching data
sites = [
    {'site_name': 'elysium', 'lat': 3, 'full_name': 'Elysium Planitia 3 N, 136 E '},
    {'site_name': 'oxia', 'lat': 18.75, 'full_name': 'Oxia Planum 18 N, 325 E '},
    {'site_name': 'mawrth_vallis', 'lat': 22.3, 'full_name': 'Mawrth Vallis 22 N, 343 E '},
    {'site_name': 'vernal', 'lat': 6, 'full_name': 'Vernal Crater 6 N, 355 E '},
    {'site_name': 'valles', 'lat': -13.9, 'full_name': 'Valles Marineres 14 S, 300 E '},
    {'site_name': 'aram', 'lat': 2.6, 'full_name': 'Aram Chaos 2 N, 339 E '},
    {'site_name': 'meridiani', 'lat': -1.95, 'full_name': 'Meridiani Planum 2 S, 354 E '},
    {'site_name': 'eberswalde', 'lat': -24, 'full_name': 'Eberswalde Crater 24 S, 327 E '}
]

# Fetch, preprocess, and rescale the data
df_merged = fetch_data(sites=sites)
df_preprocessed = data_preprocess(df=df_merged)
scaler_x, scaler_y, df_X, df_y, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = data_rescaling_and_train_test_creation(df_preprocessed)

# def evaluate_and_plot(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y):
#     """Train the model, evaluate it, and plot the results."""
#     # Train the model
#     best_model = model(X_train=X_train, y_train=y_train)
    
#     # Predict and evaluate on training data
#     train_predict = best_model.predict(X_train)
#     print(f"{model_name} Training MAE:", mean_absolute_error(y_true=y_train, y_pred=train_predict))
#     print(f"{model_name} Training R2:", r2_score(y_true=y_train, y_pred=train_predict))
#     print('*' * 20)
    
#     # Predict and evaluate on validation data
#     val_predict = best_model.predict(X_val)
#     print(f"{model_name} Validation MAE:", mean_absolute_error(y_true=y_val, y_pred=val_predict))
#     print(f"{model_name} Validation R2:", r2_score(y_true=y_val, y_pred=val_predict))
#     print('*' * 20)
    
#     # Predict and evaluate on test data
#     test_predict = best_model.predict(X_test)
#     print(f"{model_name} Test MAE:", mean_absolute_error(y_true=y_test, y_pred=test_predict))
#     print(f"{model_name} Test R2:", r2_score(y_true=y_test, y_pred=test_predict))
    
#     # Unscale the test data for visualization
#     y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
#     test_predict_unscaled = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
    
#     # Plot predicted vs actual values
#     plt.figure(figsize=(20, 12))
#     plt.scatter(y_test_unscaled, test_predict_unscaled, label=f'Predicted vs Actual ({model_name})')
#     max_val = max(y_test_unscaled.max(), test_predict_unscaled.max())
#     min_val = min(y_test_unscaled.min(), test_predict_unscaled.min())
#     plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='y = x line')
#     plt.xlabel('Actual Values')
#     plt.ylabel('Predicted Values')
#     plt.title(f'Predicted vs Actual Values ({model_name})')
#     plt.legend()
#     plt.show()

# Evaluate and plot for LightGBM
evaluate_and_plot(lgb_optimal_model, 'LightGBM', X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_y)

# Evaluate and plot for XGBoost
evaluate_and_plot(xgb_optimal_model, 'XGBoost', X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_y)
