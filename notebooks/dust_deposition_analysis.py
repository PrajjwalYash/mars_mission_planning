import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter

# Define the range of solar longitude (Ls)
Ls = np.linspace(0, 355, 72)

# Function to get dust deposition reference
def get_dd_ref():
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    path = os.path.join(parent_directory, 'data', 'elysium_dd_mean.csv')
    ely_dd = pd.read_csv(path)

    ely_ddrate = ely_dd['DD'].values
    dd_int = interpolate.interp1d(ely_dd['Ls'].values, ely_ddrate)

    dd_rate = np.array([dd_int(Ls[i]) for i in range(len(Ls))])
    ely_dd_mean = dd_rate.mean()

    return ely_dd_mean

# Function to calculate dust deposition factor for 2028
def dd_fac_2028(site, ref_dd_mean, Ef_p):
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    path = os.path.join(parent_directory, 'data', f"{site['site_name']}_dd_mean.csv")
    df_dd = pd.read_csv(path)

    ddrate = df_dd['DD'].values
    dd_int = interpolate.interp1d(df_dd['Ls'].values, ddrate)

    dd_rate = np.array([dd_int(Ls[i]) for i in range(len(Ls))])
    Ls_dd = 1 - 0.002 / ref_dd_mean * dd_rate

    sol_dd, sol_pow = compute_sol_dd_and_pow(Ls_dd, Ef_p, offsets_2028)
    
    new = compute_new(sol_dd)
    
    return new, sol_pow

# Function to calculate available energy for 2028
def avail_energy_2028(sol_pow, new, site_name):
    return avail_energy(sol_pow, new, site_name, 2028)

# Function to calculate dust deposition factor for 2031
def dd_fac_2031(site, ref_dd_mean, Ef_p):
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    path = os.path.join(parent_directory, 'data', f"{site['site_name']}_dd_mean.csv")
    df_dd = pd.read_csv(path)

    ddrate = df_dd['DD'].values
    dd_int = interpolate.interp1d(df_dd['Ls'].values, ddrate)

    dd_rate = np.array([dd_int(Ls[i]) for i in range(len(Ls))])
    Ls_dd = 1 - 0.002 / ref_dd_mean * dd_rate

    sol_dd, sol_pow = compute_sol_dd_and_pow(Ls_dd, Ef_p, offsets_2031)
    
    new = compute_new(sol_dd)
        
    return new, sol_pow

# Function to calculate available energy for 2031
def avail_energy_2031(sol_pow, new, site_name):
    return avail_energy(sol_pow, new, site_name, 2031)

# Helper function to compute sol_dd and sol_pow
def compute_sol_dd_and_pow(Ls_dd, Ef_p, offsets):
    sol_dd, sol_pow = [], []
    for offset in offsets:
        sol_dd.append(Ls_dd[int(offset)])
        sol_pow.append(Ef_p[int(offset)])
    return sol_dd, sol_pow

# Helper function to compute new array
def compute_new(sol_dd):
    new = np.ones_like(sol_dd)
    for i in range(len(sol_dd)):
        new[i] = np.prod(sol_dd[:i + 1])
    return new

# Helper function to calculate available energy and plot
def avail_energy(sol_pow, new, site_name, year):
    sol_pow_dd = [sol_pow[i] * new[i] for i in range(len(new))]
    sol_pow_smooth = 0.27778 * savgol_filter(sol_pow_dd, 5, 1) # window size 51, polynomial order 3
    
    f = plt.figure()
    plt.rcParams.update({'font.size': 16})
    f.set_figwidth(10)
    f.set_figheight(7)
    plt.plot(sol_pow_smooth)
    plt.xlabel('Sols')
    plt.ylabel('kWh/m$^{2}$')
    plt.title(f'Available energy at {site_name} \n since time of landing for {year} window')
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path1 = os.path.join(parent_directory, 'outputs', site_name + f'Available_energy_with_dust_deposition_{year}_{site_name}.png')
    output_path2 = os.path.join(parent_directory, 'outputs', site_name + f'{site_name}_{year}.txt')
    plt.savefig(output_path1)
    np.savetxt(output_path2, sol_pow_smooth)
    
    return sol_pow_smooth

# Offsets for 2028 and 2031 calculations
offsets_2028 = [(32 + i / 9) for i in range(54)] + [(32 + i / 8) for i in range(54, 142)] + \
               [(32 + i / 7) for i in range(142, 163)] + [(32 + i / 8) for i in range(163, 180)]

offsets_2031 = [(39 + i / 9) for i in range(54)] + [(39 + i / 9) for i in range(54, 142)] + \
               [(32 + i / 9) for i in range(142, 163)] + [(32 + i / 8) for i in range(163, 180)]
