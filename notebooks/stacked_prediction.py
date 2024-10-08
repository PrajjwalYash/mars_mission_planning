from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_sample_weight

def load_model(model_name):
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    model_path = os.path.join(parent_directory, 'models', f'{model_name}.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    return model


def stacking_optimal_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y):
    # Load the pre-trained base models
    lgb_model = load_model('lightgbm')
    xgb_model = load_model('xgb')
    rf_model = load_model('randomforest')
    svr_model = load_model('svr')
    knn_model = load_model('knn')
    # Resample the training data for diversity
    # weights = compute_sample_weight(class_weight='balanced', y=y_train)
    # Define the meta-model
    meta_model = LGBMRegressor(boosting_type=  'goss', extra_trees=  True, lambda_l1 =  0, lambda_l2 =   0.2, learning_rate=  0.1, linear_tree =  True, max_bin =  10, max_depth = 
8, min_split_gain = 0, n_estimators =  150,  num_leaves =  31, verbose=  -1)
    # meta_model = RandomForestQuantileRegressor(n_estimators=400, max_depth=15, random_state=42)

    # Create the stacking regressor with pre-trained base models
    stacking_model = StackingCVRegressor(
        regressors=[lgb_model, xgb_model, rf_model, svr_model, knn_model],
        meta_regressor=meta_model,
        cv=3,
        use_features_in_secondary=True
    )

    # Train the stacking regressor
    stacking_model.fit(X_train, y_train)
    
    # Evaluate on training data
    train_predict = stacking_model.predict(X_train)
    train_mae = mean_absolute_error(y_true=y_train, y_pred=train_predict)
    train_r2 = r2_score(y_true=y_train, y_pred=train_predict)
    print('*' * 20)
    print(f"{model_name} Training MAE:", train_mae)
    print(f"{model_name} Training R2:", train_r2)
    print('*' * 20)
    
    # Evaluate on validation data
    val_predict = stacking_model.predict(X_val)
    val_mae = mean_absolute_error(y_true=y_val, y_pred=val_predict)
    val_r2 = r2_score(y_true=y_val, y_pred=val_predict)
    print(f"{model_name} Validation MAE:", val_mae)
    print(f"{model_name} Validation R2:", val_r2)
    print('*' * 20)
    
    # Evaluate on test data
    test_predict = stacking_model.predict(X_test)
    test_mae = mean_absolute_error(y_true=y_test, y_pred=test_predict)
    test_r2 = r2_score(y_true=y_test, y_pred=test_predict)
    print(f"{model_name} Test MAE:", test_mae)
    print(f"{model_name} Test R2:", test_r2)

    # Unscale the test data for visualization
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    test_predict_unscaled = scaler_y.inverse_transform(test_predict.reshape(-1, 1))
    
    # Plot predicted vs actual values
    plt.figure(figsize=(20, 12))
    plt.scatter(y_test_unscaled, test_predict_unscaled, label=f'Predicted vs Observed ({model_name})')
    max_val = max(y_test_unscaled.max(), test_predict_unscaled.max())
    min_val = min(y_test_unscaled.min(), test_predict_unscaled.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label=f'R2_score = {np.round(test_r2,2)}')
    plt.xlabel('Log of Observed dust deposition rate (in kgm$^{-2}$s$^{-1}$)', fontsize=20)
    plt.ylabel('Log of Predicted dust deposition rate (in kgm$^{-2}$s$^{-1}$)', fontsize=20)
    plt.title(f'Predicted vs Actual Values ({model_name})', fontsize=30)
    plt.legend(fontsize=20)
    plt.grid()
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.getcwd()), 'outputs', model_name + '_prediction_vs_observation_DD_rate.png')
    plt.savefig(output_path)

    return {
        'Model': model_name,
        'Train MAE': train_mae,
        'Train R2': train_r2,
        'Validation MAE': val_mae,
        'Validation R2': val_r2,
        'Test MAE': test_mae,
        'Test R2': test_r2
    }