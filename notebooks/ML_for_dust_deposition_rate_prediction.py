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
    {'site_name': 'valles', 'lat': -13.9, 'full_name': 'Valles Marineres 14 S, 300 E '},
    {'site_name': 'mawrth_vallis', 'lat': 22.3, 'full_name': 'Mawrth Vallis 22 N, 343 E '},
    {'site_name': 'eberswalde', 'lat': -24, 'full_name': 'Eberswalde Crater 24 S, 327 E '},
    {'site_name': 'aram', 'lat': 2.6, 'full_name': 'Aram Chaos 2 N, 339 E '},
    {'site_name': 'vernal', 'lat': 6, 'full_name': 'Vernal Crater 6 N, 355 E '},
    {'site_name': 'meridiani', 'lat': -1.95, 'full_name': 'Meridiani Planum 2 S, 354 E '}
    
]

# Fetch, preprocess, and rescale the data
df_merged = fetch_data(sites=sites)
df_preprocessed = data_preprocess(df=df_merged)
scaler_x, scaler_y, df_X, df_y, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = data_rescaling_and_train_test_creation(df_preprocessed)


performance_metrics = []
# Evaluate and plot for LightGBM
performance_metrics.append(evaluate_and_plot(lgb_optimal_model, "LightGBM", X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_y))
# Evaluate and plot for XGBoost
performance_metrics.append(evaluate_and_plot(xgb_optimal_model, "XGBoost", X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_y))
# Evaluate and plot for RF
performance_metrics.append(evaluate_and_plot(rf_optimal_model, "RandomForest", X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_y))

df_performance = pd.DataFrame(performance_metrics)
save_performance_and_plot(df_performance=df_performance)