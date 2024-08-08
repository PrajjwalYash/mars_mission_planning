import numpy as np

# Constants
d = np.pi / 180   # Conversion factor from degrees to radians
P = 88775         # Period of Mars' rotation in seconds (one sol)
O = 25.2 * d      # Obliquity of Mars' axis in radians
e = 0.0934        # Orbital eccentricity of Mars
Lsp = 251 * d     # Solar longitude of perihelion in radians

# Arrays
Ls = np.linspace(0, 355, 72)             # Solar longitude array in degrees
lamda = np.linspace(200, 1790, 160)      # Wavelength array in nanometers
At = np.linspace(-P / 2, P / 2, 1000)    # Time array over one sol

# Initialize arrays for irradiance calculations
B_l = np.zeros((len(lamda), len(At)))
T_l = np.zeros((len(lamda), len(At)))
E_l = np.zeros((len(lamda), len(At)))

def complete_year_irradiance(lat, tau_total, w_total, g_total, A_l, E_ml):
    """
    Calculate the irradiance over a complete Martian year for a given latitude.

    Parameters:
    lat (float): Latitude in degrees.
    tau_total (list): List of tau values.
    w_total (list): List of single scattering albedo values.
    g_total (list): List of asymmetry factor values.
    A_l (list): List of surface albedo values.
    E_ml (list): List of extraterrestrial solar irradiance values.

    Returns:
    tuple: Arrays of total, beam, transmitted, and diffused irradiance for the Martian year.
    """
    num_Ls = len(Ls)
    num_At = len(At)
    num_lamda = len(lamda)

    # Initialize arrays for daily and annual irradiance values
    Ep = np.zeros(num_Ls)
    Tp = np.zeros(num_Ls)
    Dp = np.zeros(num_Ls)
    Bp = np.zeros(num_Ls)
    Ew1 = np.zeros(num_Ls)
    Tw1 = np.zeros(num_Ls)
    Dw1 = np.zeros(num_Ls)
    Bw1 = np.zeros(num_Ls)

    E_t = np.zeros((num_Ls, num_At))
    B_t = np.zeros((num_Ls, num_At))
    T_t = np.zeros((num_Ls, num_At))
    D_t = np.zeros((num_Ls, num_At))

    Phi = lat * d  # Latitude in radians

    # Convert inputs to arrays
    Lst_array = Ls[:, None] * d
    tau_total_array = np.array(tau_total)
    w_total_array = np.array(w_total)
    g_total_array = np.array(g_total)
    A_l_array = np.array(A_l)
    E_ml_array = np.array(E_ml)
    t_array = At

    for q in range(num_Ls):
        Lst = Lst_array[q]
        tau = tau_total_array[q]
        w = w_total_array[q]
        g = g_total_array[q]
        A = A_l_array[q]

        # Calculate the cosine of the solar zenith angle
        mu = (np.sin(O) * np.sin(Lst) * np.sin(Phi) +
              np.sqrt(1 - np.sin(O)**2 * np.sin(Lst)**2) * np.cos(Phi) * np.cos((2 * np.pi * t_array) / P))
        
        # Distance factor
        r2 = ((1 + e * np.cos(Lst - Lsp)) / (1 - e**2))**2
        
        # Extraterrestrial irradiance
        E = E_ml_array[:, None] * mu * r2
        E[E < 0] = 0
        
        # Direct beam irradiance
        B = E * np.exp(-tau[:, None] / mu)

        # Calculate scattering parameters
        k_l = np.sqrt(3 * (1 - w) * (1 - g * w))
        P_l = (2 / 3) * np.sqrt(3 * (1 - w) / (1 - g * w))
        alph = (3 / 4) * mu * w[:, None] * (1 + g[:, None] * (1 - w[:, None])) / (1 - (mu * k_l[:, None])**2)
        bet = (1 / 2) * mu * w[:, None] * (1 / mu + 3 * mu * g[:, None] * (1 - w[:, None])) / (1 - (mu * k_l[:, None])**2)

        # Calculate coefficients for multiple scattering
        C3 = A[:, None] + (1 - A[:, None]) * alph - (1 + A[:, None]) * bet
        C4 = (1 - A[:, None]) + P_l[:, None] * (1 + A[:, None])
        C5 = (1 - A[:, None]) - P_l[:, None] * (1 + A[:, None])

        # Compute the scattering coefficients
        C2 = ((1 + P_l[:, None]) * C3 * np.exp(-tau[:, None] / mu) - (alph + bet) * C5 * np.exp(-k_l[:, None] * tau[:, None])) / \
             ((1 + P_l[:, None]) * C4 * np.exp(k_l[:, None] * tau[:, None]) - (1 - P_l[:, None]) * C5 * np.exp(-k_l[:, None] * tau[:, None]))
        C1 = (-(1 - P_l[:, None]) * C3 * np.exp(-tau[:, None] / mu) + (alph + bet) * C4 * np.exp(k_l[:, None] * tau[:, None])) / \
             ((1 + P_l[:, None]) * C4 * np.exp(k_l[:, None] * tau[:, None]) - (1 - P_l[:, None]) * C5 * np.exp(-k_l[:, None] * tau[:, None]))

        # Transmitted irradiance
        T = E * (C1 * np.exp(-k_l[:, None] * tau[:, None]) * (1 + P_l[:, None]) + 
                 C2 * np.exp(k_l[:, None] * tau[:, None]) * (1 - P_l[:, None]) -
                 (alph + bet - 1) * np.exp(-tau[:, None] / mu))
        T[T < 0] = 0
        T[np.isnan(T)] = 0

        # Assign results to arrays
        E_l = E
        B_l = B
        T_l = T
        B_l[np.isnan(B_l)] = 0

        # Integrate over wavelength
        temp4 = np.trapz(E_l, dx=10, axis=0)
        temp5 = np.trapz(B_l, dx=10, axis=0)
        temp6 = np.trapz(T_l, dx=10, axis=0)

        E_t[q] = temp4
        B_t[q] = temp5
        T_t[q] = temp6
        B_t[np.isnan(B_t)] = 0
        T_t[np.isnan(T_t)] = 0
        D_t = T_t - B_t

        # Integrate over time
        Ep[q] = np.trapz(E_t[q], dx=P / 1000) / (10**6)
        Bp[q] = np.trapz(B_t[q], dx=P / 1000) / (10**6)
        Tp[q] = np.trapz(T_t[q], dx=P / 1000) / (10**6)
        Dp[q] = np.trapz(D_t[q], dx=P / 1000) / (10**6)
        Ew1[q] = np.max(E_t[q])
        Bw1[q] = np.max(B_t[q])
        Tw1[q] = np.max(T_t[q])
        Dw1[q] = np.max(D_t[q])

    return E_t, T_t, B_t, D_t, Ep, Tp, Bp, Dp, Ew1, Tw1, Bw1, Dw1


def rtm_api(Ls, tau, lat, w_total, g_total, A_l, E_ml):
     w_total_array = np.array(w_total)
     g_total_array = np.array(g_total)
     A_l_array = np.array(A_l)
     E_ml_array = np.array(E_ml)
     t_array = At
     Phi = lat * d  # Latitude in radians
     Lst = Ls
     t_array = At
     w = w_total_array[int(Ls/5)]
     g = g_total_array[int(Ls/5)]
     A = A_l_array[int(Ls/5)]
     # Calculate the cosine of the solar zenith angle
     mu = (np.sin(O) * np.sin(Lst) * np.sin(Phi) +
          np.sqrt(1 - np.sin(O)**2 * np.sin(Lst)**2) * np.cos(Phi) * np.cos((2 * np.pi * t_array) / P))
     # Distance factor
     r2 = ((1 + e * np.cos(Lst - Lsp)) / (1 - e**2))**2
     
     # Extraterrestrial irradiance
     E = E_ml_array[:, None] * mu * r2
     E[E < 0] = 0
     
     # Direct beam irradiance
     B = E * np.exp(-tau[:, None] / mu)
     k_l = np.sqrt(3 * (1 - w) * (1 - g * w))
     P_l = (2 / 3) * np.sqrt(3 * (1 - w) / (1 - g * w))
     alph = (3 / 4) * mu * w[:, None] * (1 + g[:, None] * (1 - w[:, None])) / (1 - (mu * k_l[:, None])**2)
     bet = (1 / 2) * mu * w[:, None] * (1 / mu + 3 * mu * g[:, None] * (1 - w[:, None])) / (1 - (mu * k_l[:, None])**2)
     # Calculate coefficients for multiple scattering
     C3 = A[:, None] + (1 - A[:, None]) * alph - (1 + A[:, None]) * bet
     C4 = (1 - A[:, None]) + P_l[:, None] * (1 + A[:, None])
     C5 = (1 - A[:, None]) - P_l[:, None] * (1 + A[:, None])

     # Compute the scattering coefficients
     C2 = ((1 + P_l[:, None]) * C3 * np.exp(-tau[:, None] / mu) - (alph + bet) * C5 * np.exp(-k_l[:, None] * tau[:, None])) / \
          ((1 + P_l[:, None]) * C4 * np.exp(k_l[:, None] * tau[:, None]) - (1 - P_l[:, None]) * C5 * np.exp(-k_l[:, None] * tau[:, None]))
     C1 = (-(1 - P_l[:, None]) * C3 * np.exp(-tau[:, None] / mu) + (alph + bet) * C4 * np.exp(k_l[:, None] * tau[:, None])) / \
          ((1 + P_l[:, None]) * C4 * np.exp(k_l[:, None] * tau[:, None]) - (1 - P_l[:, None]) * C5 * np.exp(-k_l[:, None] * tau[:, None]))

     # Transmitted irradiance
     T = E * (C1 * np.exp(-k_l[:, None] * tau[:, None]) * (1 + P_l[:, None]) + 
               C2 * np.exp(k_l[:, None] * tau[:, None]) * (1 - P_l[:, None]) -
               (alph + bet - 1) * np.exp(-tau[:, None] / mu))
     # Assign results to arrays
     E_l = E
     B_l = B
     T_l = T
     B_l[np.isnan(B_l)] = 0

     # Integrate over wavelength
     temp4 = np.trapz(E_l, dx=10, axis=0)
     temp5 = np.trapz(B_l, dx=10, axis=0)
     temp6 = np.trapz(T_l, dx=10, axis=0)

     E_t = temp4
     B_t = temp5
     T_t = temp6
     B_t[np.isnan(B_t)] = 0
     T_t[np.isnan(T_t)] = 0
     D_t = T_t - B_t
     return Ls, E_t, B_t, D_t