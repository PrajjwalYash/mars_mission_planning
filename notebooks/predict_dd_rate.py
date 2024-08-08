from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

def load_model(model_name):
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    model_path = os.path.join(parent_directory, 'models', f'{model_name}.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

def lgb_prediction(test_data):
    lgb_model = load_model('lightgbm')
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    model_path = os.path.join(parent_directory, 'models', 'scaler_y.pkl')
    with open(model_path, 'rb') as file:
        scaler_y = pickle.load(file)
    prediction = lgb_model.predict(test_data)
    prediction = scaler_y.inverse_transform(prediction)
    prediction = np.exp(prediction)

    return prediction


def svr_prediction(test_data):
    lgb_model = load_model('svr')
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    model_path = os.path.join(parent_directory, 'models', 'scaler_y.pkl')
    with open(model_path, 'rb') as file:
        scaler_y = pickle.load(file)
    prediction = lgb_model.predict(test_data)
    prediction = scaler_y.inverse_transform(prediction)
    prediction = np.exp(prediction)

    return prediction

