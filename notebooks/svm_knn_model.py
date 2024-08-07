from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
import pickle
import os
from sklearn.utils.class_weight import compute_sample_weight


def svm_optimal_model(X_train, y_train):
    # Define the MAE scorer for GridSearchCV
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Define the parameter grid for SVM
    param_dist = {
        'kernel': ['linear', 'rbf'],   # Linear and RBF kernels
        'C': [0.1, 1, 10],             # Regularization parameter
        'gamma': ['scale', 'auto'],    # Kernel coefficient for 'rbf'
        'epsilon': [0.05, 0.1, 0.2],   # Epsilon in the epsilon-SVR model
    }

    # Set up cross-validation with 3 folds
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # Initialize the SVM regressor
    svm_model = SVR()

   # Resample the training data for diversity
    # weights = compute_sample_weight(class_weight='balanced', y=y_train)
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_dist, scoring=mae_scorer, cv=kf, 
                               verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and score from the grid search
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters found for SVM: {best_params}")
    print(f"Best MAE score for SVM: {-best_score}")

    # Train the model with the best hyperparameters
    svm_model = SVR(**best_params)
    svm_model.fit(X_train, y_train)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory = os.path.join(parent_directory, 'models', 'svr.pkl')
    with open(output_directory, 'wb') as file:
        pickle.dump(svm_model, file)
    return svm_model

def knn_optimal_model(X_train, y_train):
    # Define the MAE scorer for GridSearchCV
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Define the parameter grid for KNN
    param_dist = {
        'n_neighbors': [3, 5, 7, 9],  # Number of neighbors
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
        'p': [1, 2]  # Power parameter for the Minkowski metric
    }

    # Set up cross-validation with 3 folds
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # Initialize the KNN regressor
    knn_model = KNeighborsRegressor()

   # Resample the training data for diversity
    # weights = compute_sample_weight(class_weight='balanced', y=y_train)
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=knn_model, param_grid=param_dist, scoring=mae_scorer, cv=kf, 
                               verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and score from the grid search
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters found for KNN: {best_params}")
    print(f"Best MAE score for KNN: {-best_score}")

    # Train the model with the best hyperparameters
    knn_model = KNeighborsRegressor(**best_params)
    knn_model.fit(X_train, y_train)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory = os.path.join(parent_directory, 'models', 'knn.pkl')
    with open(output_directory, 'wb') as file:
        pickle.dump(knn_model, file)
    return knn_model