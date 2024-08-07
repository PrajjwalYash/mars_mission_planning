import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
import pickle
import os

# Function to calculate Mean Absolute Error (MAE) as a custom scoring function
def mae_score(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Function to find the optimal LightGBM model
def lgb_optimal_model(X_train, y_train):
    # Define the MAE scorer for GridSearchCV
    mae_scorer = make_scorer(mae_score, greater_is_better=False)

    # Define the parameter grid for LightGBM
    param_dist = {
        'num_leaves': [31, 50],
        'max_depth': [8, 15],
        'max_bin': [10,20],
        'extra_trees': [True],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [150, 200],
        'boosting_type': ['goss'],
        'lambda_l1': [0, 0.2],
        'linear_tree': [True],
        'lambda_l2': [0, 0.2],
        'min_split_gain': [0, 0.1],
        'verbose':[-1]
    }

    # Set up cross-validation with 3 folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize the LightGBM regressor
    lgb_model = lgb.LGBMRegressor()

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_dist, scoring=mae_scorer, cv=kf, 
                               verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and score from the grid search
    best_params = grid_search.best_params_
    best_scores = grid_search.best_score_
    print(f"Best parameters found for LightGBM: ", best_params)
    print(f"Best MAE score for LightGBM: ", -best_scores)

    # Train the model with the best hyperparameters
    lgb_model = lgb.LGBMRegressor(**best_params)
    lgb_model.fit(X_train, y_train)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory = os.path.join(parent_directory, 'models', 'lightgbm.pkl')
    with open(output_directory, 'wb') as file:
        pickle.dump(lgb_model, file)
    return lgb_model

# Function to find the optimal XGBoost model
def xgb_optimal_model(X_train, y_train):
    # Define the MAE scorer for GridSearchCV
    mae_scorer = make_scorer(mae_score, greater_is_better=False)

    # Define the parameter grid for XGBoost
    param_dist = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [200, 350, 400],
        'gamma': [1.5, 3],
        'max_bin': [10, 20],
        'reg_alpha': [0.5, 1, 2],  # L1 regularization term on weights
        'reg_lambda': [0.5, 1, 2],  # L2 regularization term on weights
        'min_child_weight': [3, 5, 10],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.8, 1]
    }

    # Set up cross-validation with 3 folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize the XGBoost regressor
    xgb_model = xgb.XGBRegressor()

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_dist, scoring=mae_scorer, cv=kf, 
                               verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and score from the grid search
    best_params = grid_search.best_params_
    best_scores = grid_search.best_score_
    print(f"Best parameters found for XGBoost: ", best_params)
    print(f"Best MAE score for XGBoost: ", -best_scores)

    # Train the model with the best hyperparameters
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_train, y_train)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory = os.path.join(parent_directory, 'models', 'xgb.pkl')
    with open(output_directory, 'wb') as file:
        pickle.dump(xgb_model, file)
    return xgb_model


def rf_optimal_model(X_train, y_train):
    # Define the MAE scorer for GridSearchCV
    mae_scorer = make_scorer(mae_score, greater_is_better=False)

    # Define the parameter grid for Random Forest
    param_dist = {
        'n_estimators': [350, 450, 500],
        'max_depth': [10, 15],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [2, 4],
        'bootstrap': [True, False],
        'ccp_alpha': [0.01, 0.05, 0.1] 
    }

    # Set up cross-validation with 3 folds
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize the Random Forest regressor
    rf_model = RandomForestRegressor()

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_dist, scoring=mae_scorer, cv=kf, 
                               verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and score from the grid search
    best_params = grid_search.best_params_
    best_scores = grid_search.best_score_
    print(f"Best parameters found for Random Forest: ", best_params)
    print(f"Best MAE score for Random Forest: ", -best_scores)

    # Train the model with the best hyperparameters
    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(X_train, y_train)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory = os.path.join(parent_directory, 'models', 'randomforest.pkl')
    with open(output_directory, 'wb') as file:
        pickle.dump(rf_model, file)
    return rf_model
