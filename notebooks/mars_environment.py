import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import os

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


def water_ice_cloud():
    print("Starting water_ice_cloud function")
    tau_rc = np.zeros_like(Ls)
    tau_lc = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))
    tau_lca = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))
    tau_lcs = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))
    g_c = 0.85 * np.ones_like(lamda)
    w_c = np.ones_like(lamda)
    Q_c = np.zeros_like(lamda)

    for k in range(0, 31):
        Q_c[k] = 2.1
    for k in range(31, 160):
        Q_c[k] = 2.2

    for k in range(0, 11):
        w_c[k] = 0.67

    for i in range(0, 6):
        tau_rc[i] = 0.01
    for i in range(6, 31):
        tau_rc[i] = 0.16 - ((0.15) / (75 ** 2)) * ((Ls[i] - 90) ** 2)
    for i in range(30, 42):
        tau_rc[i] = 0.02
    for i in range(42, 56):
        tau_rc[i] = 0.01
    for i in range(56, 62):
        tau_rc[i] = 0.02
    for i in range(62, 67):
        tau_rc[i] = 0.01
    for i in range(67, 72):
        tau_rc[i] = 0.02
    for i in range(0, 72):
        for k in range(0, 160):
            tau_lc[i][k] = (Q_c[k] * tau_rc[i]) / (1.45 * (1 - 0.4))
            tau_lcs[i][k] = w_c[k] * tau_lc[i][k]
            tau_lca[i][k] = (1 - w_c[k]) * tau_lc[i][k]
    print("Completed water_ice_cloud function")
    return tau_rc, tau_lc, tau_lca, tau_lcs, g_c


def surface_albedo():
    print("Starting surface_albedo function")
    mf = np.ones_like(Ls)
    A = 0.4 * np.ones_like(lamda)
    Ap = np.zeros_like(Ls)
    A_l = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))

    for k in range(0, 21):
        A[k] = 0.05
    for k in range(21, 51):
        A[k] = (3.888 * 10 ** -6) * ((lamda[k] - 400) ** 2) + 0.05

    for i in range(0, 36):
        mf[i] = 0.87 - ((9.4 * 10 ** -4) * Ls[i])
    for i in range(36, 54):
        mf[i] = 0.0044 * (Ls[i] - 180) + 0.7
    for i in range(54, 72):
        mf[i] = 1.1 - 0.00255 * (Ls[i] - 270)
    for i in range(0, 72):
        A_l[i][:] = mf[i] * A
    for k in range(0, len(Ls)):
        Ap[k] = A_l[k][145]
    print("Completed surface_albedo function")
    return A_l


def atm_gas():
    print("Starting atm_gas function")
    tau_ga = np.zeros_like(lamda)
    tau_gs = np.zeros_like(lamda)
    for k in range(0, 41):
        tau_gs[k] = 0.18 - (0.18 / 20) * (np.sqrt(lamda[k] - 200))
    for k in range(0, 6):
        tau_ga[k] = 0.000012 * ((lamda[k] - 200) ** 2)
    for k in range(6, 11):
        tau_ga[k] = 0.03 - 0.000012 * ((lamda[k] - 250) ** 2)
    print("Completed atm_gas function")
    return tau_ga, tau_gs


def optical_params():
    print("Starting optical_params function")
    g_d = 0.7 * np.ones_like(lamda)
    w_d = 0.98 * np.ones_like(lamda)
    Q_d = np.zeros_like(lamda)

    for k in range(0, 31):
        Q_d[k] = 2.5
    for k in range(31, 160):
        Q_d[k] = 2.5 + 0.002044 * (lamda[k] - 500)
    for k in range(0, 51):
        w_d[k] = 0.65 + ((9.16667 * 10 ** -7) * ((lamda[k] - 100) ** 2))
        g_d[k] = 0.85 - 0.00025 * (lamda[k] - 100)
    print("Completed optical_params function")
    return g_d, w_d, Q_d


def get_optical_depth(site):
    print("Starting get_optical_depth function")
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    path = os.path.join(parent_directory, 'data', str(site['site_name']) + '_tau.csv')
    print(f"Reading file from path: {path}")
    tau = pd.read_csv(path)
    tau_int = interpolate.interp1d(tau['Ls'].values, tau['tau'].values)
    tau_int = tau_int(Ls)
    tau_rd = tau_int * 2.67 / 2.6
    plt.plot(Ls, tau_rd)
    print("Completed get_optical_depth function")
    return tau_rd


def all_wavelength_dust_od(tau_rd, Q_d, w_d):
    print("Starting all_wavelength_dust_od function")
    tau_ld = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))
    tau_lda = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))
    tau_lds = np.zeros((np.shape(Ls)[0], np.shape(lamda)[0]))
    for i in range(0, 72):
        for k in range(0, 160):
            tau_ld[i][k] = (Q_d[k] * tau_rd[i]) / (Q_d[68])
            tau_lds[i][k] = w_d[k] * tau_ld[i][k]
            tau_lda[i][k] = (1 - w_d[k]) * tau_ld[i][k]
    print("Completed all_wavelength_dust_od function")
    return tau_ld, tau_lds, tau_lda


def am_mars_spectrum():
    print("Starting am_mars_spectrum function")
    current_working_directory = os.getcwd()
    parent_directory = os.path.dirname(current_working_directory)
    path1 = os.path.join(parent_directory, 'data', 'am02.txt')
    path2 = os.path.join(parent_directory, 'data', 'lamda2.txt')
    print(f"Reading files from paths: {path1} and {path2}")
    Em0 = np.loadtxt(path1)
    lamda2 = np.loadtxt(path2)
    Emr = Em0 / ((1.52) ** 2)
    Eminter = interpolate.interp1d(lamda2, Emr)
    E_ml = np.zeros_like(lamda)
    for k in range(1, len(lamda)):
        E1 = Eminter(lamda[k])
        E_ml[k] = E1
    E_ml[0] = Emr[0]
    print("Completed am_mars_spectrum function")
    return E_ml
