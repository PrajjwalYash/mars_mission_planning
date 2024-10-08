import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Function to fetch and merge data from multiple sites
def fetch_data(sites):
    df_complete = pd.DataFrame(columns=['Ls', 'TOA', 'Eff', 'Direct', 'Diffuse', 'DD'])
    for site in sites:
        # Define paths to input files based on the current directory structure
        current_working_directory = os.getcwd()
        parent_directory = os.path.dirname(current_working_directory)
        input_path1 = os.path.join(parent_directory, 'outputs', site['site_name'] + '_sur_irr_.csv')
        input_path2 = os.path.join(parent_directory, 'data', f"{site['site_name']}_dd_mean.csv")

        # Read the CSV files into DataFrames
        df1 = pd.read_csv(input_path1)
        df2 = pd.read_csv(input_path2)

        # Merge the DataFrames on the 'Ls' column and interpolate missing 'DD' values
        df1 = pd.merge(df1, df2, on='Ls', how='left', suffixes=('', '_1'))
        df1['DD'] = df1['DD'].interpolate(method='linear')

        # Append the merged DataFrame to the complete DataFrame
        df_complete = df_complete.append(df1)
    
    # Define the output path and save the complete DataFrame to a CSV file
    output_path = os.path.join(parent_directory, 'outputs', 'all_sites_merged.csv')
    df_complete.to_csv(output_path, index=False)
    return df_complete

# Function to preprocess the data
def data_preprocess(df):
    # Select relevant columns and create new feature columns 'B/E' and 'D/E'
    df = df[['Ls', 'TOA', 'Direct', 'Diffuse', 'DD']]
    df['B/E'] = df['Direct'] / df['TOA']
    df['D/E'] = df['Diffuse'] / df['TOA']

    # Drop unnecessary columns
    df = df.drop(columns=['Direct', 'Diffuse'])

    # Save the preprocessed DataFrame to a CSV file
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'preprocessed_data.csv')
    df.to_csv(output_path, index=False)
    return df

# Function to rescale the data and create train, validation, and test sets
def data_rescaling_and_train_test_creation(df):
    # Split the DataFrame into features (X) and target (y)
    df_X = df.drop(columns=['DD'])
    # df_y = df[['DD']]
    df_y = np.log(df[['DD']])

    # Split data into train+validation and test sets
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(df_X, df_y, random_state=100, shuffle=True, test_size=0.2)

    # Further split the train+validation set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tr_val, y_tr_val, random_state=100, shuffle=True, test_size=0.4)

    # Initialize scalers for features and target
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform the training data, and transform the validation and test data
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    X_val_scaled = scaler_x.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val).ravel()
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test).ravel()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_directory1 = os.path.join(parent_directory, 'models', 'scaler_x.pkl')
    output_directory2 = os.path.join(parent_directory, 'models', 'scaler_y.pkl')
    with open(output_directory1, 'wb') as file:
        pickle.dump(scaler_x, file)
    with open(output_directory2, 'wb') as file:
        pickle.dump(scaler_y, file)
    return scaler_x, scaler_y, df_X, df_y, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled


def data_preprocess_test_api(Ls, E_t, B_t, D_t):
    df_test = pd.DataFrame(columns=['Ls', 'TOA', 'Direct', 'Diffuse'])
    df_test['Ls'] = Ls
    df_test['TOA'] = E_t
    df_test['Direct'] = B_t
    df_test['Diffuse'] = D_t
    df_test['B/E'] = df_test['Direct'] / df_test['TOA']
    df_test['D/E'] = df_test['Diffuse'] / df_test['TOA']

    # Drop unnecessary columns
    df_test = df_test.drop(columns=['Direct', 'Diffuse'])

    # Load the standard scaler
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    model_path = os.path.join(parent_directory, 'models', 'scaler_x.pkl')
    with open(model_path, 'rb') as file:
        scaler_x = pickle.load(file)
    
    test_data = scaler_x.transform(df_test)

    return test_data