import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def fetch_data(sites):
    df_complete = pd.DataFrame(columns=['Ls', 'TOA', 'Eff', 'Direct', 'Diffuse', 'DD'])
    for site in sites:
        current_working_directory = os.getcwd()
        parent_directory = os.path.dirname(current_working_directory)
        input_path1 = os.path.join(parent_directory, 'outputs', site['site_name'] + '_sur_irr_.csv')
        input_path2 = os.path.join(parent_directory, 'data', f"{site['site_name']}_dd_mean.csv")
        df1 = pd.read_csv(input_path1)
        df2 = pd.read_csv(input_path2)
        df1 = pd.merge(df1, df2, on='Ls', how='left', suffixes=('', '_1'))
        df1['DD'] = df1['DD'].interpolate(method='linear')
        df_complete = df_complete.append(df1)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'all_sites_merged.csv')
    df_complete.to_csv(output_path, index=False)
    return df_complete

def data_preprocess(df):
    df = df[['Ls', 'TOA', 'Direct', 'Diffuse', 'DD']]
    df['B/E'] = df['Direct']/df['TOA']
    df['D/E'] = df['Diffuse']/df['TOA']
    df = df.drop(columns = ['Ls', 'Direct', 'Diffuse'])
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'preprocessed_data.csv')
    df.to_csv(output_path, index=False)
    return df

def data_rescaling_and_train_test_creation(df):
    df_X = df.drop(columns = ['DD'])
    df_y = df[['DD']]
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(df_X, df_y, random_state=100, shuffle=True, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_tr_val, y_tr_val)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled= scaler_x.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    return scaler_x,scaler_y, df_X, df_y, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled



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
df_merged = fetch_data(sites=sites)
df_preprocessed = data_preprocess(df=df_merged)
scaler_x,scaler_y, df_X, df_y, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = data_rescaling_and_train_test_creation(df_preprocessed)
print(y_test_scaled)