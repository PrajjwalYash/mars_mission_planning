import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os

def fetch_data(sites):
    df_complete = pd.DataFrame(columns=['Ls', 'TOA', 'Eff', 'Direct', 'Diffuse'])
    for site in sites:
        current_working_directory = os.getcwd()
        parent_directory = os.path.dirname(current_working_directory)
        input_path = os.path.join(parent_directory, 'outputs', site['site_name'] + '_sur_irr_.csv')
        df = pd.read_csv(input_path)
        df_complete = df_complete.append(df)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'all_sites_merged.csv')
    df_complete.to_csv(output_path)
    return df_complete

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
fetch_data(sites=sites)