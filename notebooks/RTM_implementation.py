import numpy as np
from mars_environment import *
from complete_year_irr import *
from complete_year_analysis import *
from dust_deposition_analysis import *
from load_support import *
import pandas as pd
import matplotlib.pyplot as plt

# Constants and parameters (assuming these are defined elsewhere)
d = np.pi / 180
P = 88775
O = 25.2 * d
e = 0.0934
Lsp = 251 * d
Ls = np.linspace(0, 355, 72)
lamda = np.linspace(200, 1790, 160)
At = np.linspace(-88775 / 2, 88775 / 2, 1000)
B_l = np.zeros((np.shape(lamda)[0], np.shape(At)[0]))
T_l = np.zeros((np.shape(lamda)[0], np.shape(At)[0]))
E_l = np.zeros((np.shape(lamda)[0], np.shape(At)[0]))
# Mission details
ecl_load=120
sunlit_load=200
payload=150
bat_eff=0.83
panel_size=7
panel_eff=0.28

def main():
    # List of sites with their details
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
    
    # Initialize df_compare only once
    df_compare = pd.DataFrame(columns=['Sols'])
    load_support_duration_2028 = pd.DataFrame(columns=['Sols'])
    load_support_duration_2031 = pd.DataFrame(columns=['Sols'])

    # Iterate over each site and perform calculations
    for site in sites:
        tau_rc, tau_lc, tau_lca, tau_lcs, g_c = water_ice_cloud()
        
        A_l = surface_albedo()
        tau_ga, tau_gs = atm_gas()
        
        g_d, w_d, Q_d = optical_params()
        tau_rd = get_optical_depth(site=site)
        tau_ld, tau_lds, tau_lda = all_wavelength_dust_od(tau_rd, Q_d, w_d)
        tau_total = tau_lc + tau_ga + tau_gs + tau_ld
        w_total = tau_lcs + tau_lds + tau_gs
        w_total = w_total / tau_total
        g_total = (tau_lds * g_d + tau_lcs * g_c) / (tau_lcs + tau_lds + tau_gs)
        w_total[w_total > 1] = 1
        
        E_ml = am_mars_spectrum()
        E_t, T_t, B_t, D_t, Ep, Tp, Bp, Dp, Ew1, Tw1, Bw1, Dw1 = complete_year_irradiance(lat=site['lat'], tau_total=tau_total, w_total=w_total, g_total=g_total, A_l=A_l, E_ml=E_ml)
        Ef_t = B_t + 0.86 * D_t
        Ef_w1 = Bw1 + 0.86 * Dw1
        Ef_p = Bp + 0.86 * Dp
        
        plot_daily_maxima(site_name=site['full_name'], Ls=Ls, Ew1=Ew1, Ef_w1=Ef_w1, Bw1=Bw1, Dw1=Dw1)
        plot_available_energy_wo_dust(site_name=site['full_name'], Ls=Ls, Ef_p=Ef_p)
        export_complete_year_irradiance(site_name=site['site_name'], Ls=Ls, Ew1=Ew1, Ef_w1=Ef_w1, Bw1=Bw1, Dw1=Dw1)
        
        ref_dd_mean = get_dd_ref()
        
        new, sol_pow, ecl_dur_2028 = dd_fac_2028(site=site, ref_dd_mean=ref_dd_mean, Ef_p=Ef_p, Ef_t = Ef_t)
        energy_2028 = avail_energy_2028(sol_pow=sol_pow, new=new, site_name=site['full_name'])
        
        new, sol_pow, ecl_dur_2031 = dd_fac_2031(site=site, ref_dd_mean=ref_dd_mean, Ef_p=Ef_p, Ef_t = Ef_t)
        # print(ecl_dur_2028)
        energy_2031 = avail_energy_2031(sol_pow=sol_pow, new=new, site_name=site['full_name'])
        
        # Update df_compare with energy data for each site and launch window
        if 'Sols' not in df_compare.columns:
            df_compare['Sols'] = np.arange(1, 181, 1)
        
        df_compare[site['site_name'] + str(2028)] = energy_2028
        df_compare[site['site_name'] + str(2031)] = energy_2031

        load_support_duration_2028['Sols'] = np.arange(1,180,1)
        load_support_duration_2028[site['site_name']] = load_support_duration(ecl_dur=np.array(ecl_dur_2028), sol_pow_smooth=energy_2028,ecl_load=ecl_load, sunlit_load=sunlit_load, payload=payload, bat_eff=bat_eff, panel_size=panel_size,panel_eff=panel_eff)
        load_support_duration_2031['Sols'] = np.arange(1,180,1)
        load_support_duration_2031[site['site_name']] = load_support_duration(ecl_dur=np.array(ecl_dur_2031), sol_pow_smooth=energy_2031,ecl_load=ecl_load, sunlit_load=sunlit_load, payload=payload, bat_eff=bat_eff, panel_size=panel_size,panel_eff=panel_eff)
    load_support_duration_2031.set_index('Sols', inplace = True)
    load_support_duration_2028.set_index('Sols', inplace = True)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'comparison_load_support_2028.csv')
    load_support_duration_2028.to_csv(output_path)
    output_path = os.path.join(parent_directory, 'outputs', 'comparison_load_support_2031.csv')
    load_support_duration_2031.to_csv(output_path)
    # Perform energy comparison plots
    df_compare_cumsum = df_compare.cumsum(axis=0)
    df_compare_cumsum = df_compare_cumsum.divide(np.array(df_compare_cumsum.index + 1), axis=0)
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'comparison_energy.csv')
    df_compare_cumsum.to_csv(output_path)
    
    energy_compare_2031 = df_compare_cumsum[df_compare_cumsum.columns[df_compare_cumsum.columns.str.endswith('2031')]]
    energy_compare_2028 = df_compare_cumsum[df_compare_cumsum.columns[df_compare_cumsum.columns.str.endswith('2028')]]    
    
    plt.figure(figsize=(20, 12))
    for col in energy_compare_2028.columns:
        plt.plot(np.arange(1, 181, 1), energy_compare_2028[col].values, lw=3, label=col)
    plt.xlabel('Sols since landing', fontsize=30)
    plt.ylabel('kWh/m$^{2}$', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.title('Available energy comparison for different sites: 2028 launch window', fontsize=30)
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'Energy_comparison_2028.png')
    plt.savefig(output_path)
    
    plt.figure(figsize=(20, 12))
    for col in energy_compare_2031.columns:
        plt.plot(np.arange(1,181,1), energy_compare_2031[col].values, lw=3, label=col)
    plt.xlabel('Sols since landing', fontsize=30)
    plt.ylabel('kWh/m$^{2}$', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.title('Available energy comparison for different sites: 2031 launch window', fontsize=30)
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'Energy_comparison_2031.png')
    plt.savefig(output_path)

    plt.figure(figsize=(20, 12))
    for col in load_support_duration_2028.columns:
        plt.plot(np.arange(1,180,1), load_support_duration_2028[col].values, lw=3, label=col)
    plt.xlabel('Sols since landing', fontsize=30)
    plt.ylabel('Time (in hours)', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.title('Payload operation duration possible since landing for different sites: 2028 launch window', fontsize=30)
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'payload_duration_comparison_2028.png')
    plt.savefig(output_path)


    plt.figure(figsize=(20, 12))
    for col in load_support_duration_2031.columns:
        plt.plot(np.arange(1,180,1), load_support_duration_2031[col].values, lw=3, label=col)
    plt.xlabel('Sols since landing', fontsize=30)
    plt.ylabel('Time (in hours)', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=20)
    plt.title('Payload operation duration possible since landing for different sites: 2031 launch window', fontsize=30)
    plt.grid()
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    output_path = os.path.join(parent_directory, 'outputs', 'payload_duration_comparison_2031.png')
    plt.savefig(output_path)

if __name__ == "__main__":
    main()
