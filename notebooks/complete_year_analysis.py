import numpy as np
import scipy.stats
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import interpolate

def plot_daily_maxima(site_name, Ls, Ew1, Ef_w1, Bw1, Dw1):
    """
    Plots the annual variation of daily maxima solar irradiance for a given site.

    Parameters:
    site_name (str): Name of the site.
    Ls (array-like): Array of solar longitude values.
    Ew1 (array-like): Top of atmosphere irradiation values.
    Ef_w1 (array-like): Effective normal irradiation values.
    Bw1 (array-like): Direct irradiation values.
    Dw1 (array-like): Diffuse irradiation values.
    """
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    
    # Plotting various irradiation components
    plt.plot(Ls, Ew1, label='Top of atmosphere irradiation')
    plt.plot(Ls, Ef_w1, label='Effective normal irradiation')
    plt.plot(Ls, Bw1, label='Direct irradiation')
    plt.plot(Ls, Dw1, label='Diffuse irradiation')
    
    # Marking specific events with vertical lines
    plt.axvline(32*5, ls='--', color='cyan', label='Landing time: 2028 window')
    plt.axvline(39*5, ls='--', color='goldenrod', label='Landing time: 2031 window')
    plt.axvline(251, ls='--', color='red', label='Perihelion')
    
    # Adding legend, title, and labels
    plt.legend()
    plt.title('Annual variation of daily maxima solar irradiance\n at {}'.format(site_name), fontsize=20)
    plt.xlabel('Time of the year', fontsize=20)
    plt.ylabel('W/m$^2$', fontsize=20)
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', site_name + '_daily_maxima.png')
    # Saving the plot
    plt.savefig(output_path)

def plot_available_energy_wo_dust(site_name, Ls, Ef_p):
    """
    Plots the effective available energy at a site without considering dust deposition.

    Parameters:
    site_name (str): Name of the site.
    Ls (array-like): Array of solar longitude values.
    Ef_p (array-like): Effective available energy values.
    """   
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(7)
    # Plotting the total flux available
    plt.plot(Ls, Ef_p, label='Total flux available')
    plt.legend(loc='upper right')
    plt.suptitle('Effective available energy at {} \n without dust deposition'.format(site_name))
    plt.xlabel('Time of the year')
    plt.ylabel('MJ/m$^2$')
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', site_name + '_available_energy_before_dust_deposition.png')
    # Saving the plot
    plt.savefig(output_path)

def export_complete_year_irradiance(site_name, Ls, Ew1, Ef_w1, Bw1, Dw1):
    """
    Exports the complete year irradiance data to a CSV file.

    Parameters:
    site_name (str): Name of the site.
    Ls (array-like): Array of solar longitude values.
    Ew1 (array-like): Top of atmosphere irradiation values.
    Ef_w1 (array-like): Effective normal irradiation values.
    Bw1 (array-like): Direct irradiation values.
    Dw1 (array-like): Diffuse irradiation values.
    """
    # Creating a DataFrame with the irradiance data
    sur_irr = pd.DataFrame(Ls, columns=['Ls'])
    sur_irr['TOA'] = Ew1
    sur_irr['Eff'] = Ef_w1
    sur_irr['Direct'] = Bw1
    sur_irr['Diffuse'] = Dw1
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', site_name + '_sur_irr_' + site_name + '.csv')
    # Saving the DataFrame to a CSV file
    sur_irr.to_csv(output_path)
