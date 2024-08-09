import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import functions from your modules
from mars_environment import *
from complete_year_irr import *
from complete_year_analysis import *
from dust_deposition_analysis import *
from load_support import *

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

# Site details
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

st.title("Mars Mission Power Analysis")

# Select multiple sites
selected_sites = st.multiselect(
    "Select Landing Sites",
    [site['full_name'] for site in sites],
    default=[site['full_name'] for site in sites[:2]]  # Default to the first two sites
)

# Slider for daytime and nighttime loads
ecl_load = st.slider("Select Nighttime Load (W)", min_value=100, max_value=250, value=120, step=10)
sunlit_load = st.slider("Select Daytime Load (W)", min_value=100, max_value=250, value=200, step=10)

# Slider for payload
payload = st.slider("Select Payload (kg)", min_value=140, max_value=180, value=150, step=5)

# Slider for panel size
panel_size = st.slider("Select Panel Size (m²)", min_value=5.0, max_value=7.0, value=7.0, step=0.5)
    # Fixed parameters
bat_eff = 0.83
panel_eff = 0.28

# Initialize dataframes for comparison
df_compare = pd.DataFrame(columns=['Sols'])
load_support_duration_2028 = pd.DataFrame(columns=['Sols'])
load_support_duration_2031 = pd.DataFrame(columns=['Sols'])
# Iterate over selected sites and perform calculations
for site in sites:
    if site['full_name'] not in selected_sites:
        continue

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
    E_t, T_t, B_t, D_t, Ep, Tp, Bp, Dp, Ew1, Tw1, Bw1, Dw1 = complete_year_irradiance(
        lat=site['lat'], tau_total=tau_total, w_total=w_total, g_total=g_total, A_l=A_l, E_ml=E_ml
    )
    Ef_t = B_t + 0.86 * D_t
    Ef_w1 = Bw1 + 0.86 * Dw1
    Ef_p = Bp + 0.86 * Dp

    plot_daily_maxima(site_name=site['full_name'], Ls=Ls, Ew1=Ew1, Ef_w1=Ef_w1, Bw1=Bw1, Dw1=Dw1)
    plot_available_energy_wo_dust(site_name=site['full_name'], Ls=Ls, Ef_p=Ef_p)
    export_complete_year_irradiance(site_name=site['site_name'], Ls=Ls, Ew1=Ew1, Ef_w1=Ef_w1, Bw1=Bw1, Dw1=Dw1)

    ref_dd_mean = get_dd_ref()

    new, sol_pow, ecl_dur_2028 = dd_fac_2028(site=site, ref_dd_mean=ref_dd_mean, Ef_p=Ef_p, Ef_t=Ef_t)
    energy_2028 = avail_energy_2028(sol_pow=sol_pow, new=new, site_name=site['full_name'])

    new, sol_pow, ecl_dur_2031 = dd_fac_2031(site=site, ref_dd_mean=ref_dd_mean, Ef_p=Ef_p, Ef_t=Ef_t)
    energy_2031 = avail_energy_2031(sol_pow=sol_pow, new=new, site_name=site['full_name'])

    if 'Sols' not in df_compare.columns:
        df_compare['Sols'] = np.arange(1, 181, 1)

    df_compare[site['site_name'] + str(2028)] = energy_2028
    df_compare[site['site_name'] + str(2031)] = energy_2031

    load_support_duration_2028['Sols'] = np.arange(1, 180, 1)
    load_support_duration_2028[site['site_name']] = load_support_duration(
        ecl_dur=np.array(ecl_dur_2028), sol_pow_smooth=energy_2028,
        ecl_load=ecl_load, sunlit_load=sunlit_load, payload=payload,
        bat_eff=bat_eff, panel_size=panel_size, panel_eff=panel_eff
    )

    load_support_duration_2031['Sols'] = np.arange(1, 180, 1)
    load_support_duration_2031[site['site_name']] = load_support_duration(
        ecl_dur=np.array(ecl_dur_2031), sol_pow_smooth=energy_2031,
        ecl_load=ecl_load, sunlit_load=sunlit_load, payload=payload,
        bat_eff=bat_eff, panel_size=panel_size, panel_eff=panel_eff
    )

df_compare_cumsum = df_compare.cumsum(axis=0)
df_compare_cumsum = df_compare_cumsum.divide(np.array(df_compare_cumsum.index + 1), axis=0)

energy_compare_2031 = df_compare_cumsum[df_compare_cumsum.columns[df_compare_cumsum.columns.str.endswith('2031')]]
energy_compare_2028 = df_compare_cumsum[df_compare_cumsum.columns[df_compare_cumsum.columns.str.endswith('2028')]]
st.write(energy_compare_2028.head())
plt.figure(figsize=(20, 12))
for col in energy_compare_2028.columns:
    plt.plot(np.arange(1, 181, 1), energy_compare_2028[col], label=col)
for col in energy_compare_2031.columns:
    plt.plot(np.arange(1, 181, 1), energy_compare_2031[col], label=col)

plt.xlabel("Sols")
plt.ylabel("Cumulative Mean Energy (kWh)")
plt.title("Cumulative Mean Energy Comparison for Selected Sites")
plt.legend()
plt.grid(True)
st.pyplot(plt)
