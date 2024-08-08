import numpy as np
from mars_environment import *
from complete_year_irr import *
from complete_year_analysis import *
from dust_deposition_analysis import *
from load_support import *
from ML_data_creation import *
from predict_dd_rate import *
import pandas as pd
import matplotlib.pyplot as plt

A_l = surface_albedo()
g_total, w_total, _ = optical_params()
E_ml = am_mars_spectrum()

#inputs
Ls = 200
lat = 4
tau = 0.8

#Implementation
Ls, E_t, B_t, D_t = rtm_api(Ls=Ls, tau=tau, lat = lat, w_total=w_total, g_total=g_total, A_l=A_l, E_ml=E_ml)
test_data = data_preprocess_test_api(Ls=Ls, E_t= E_t, B_t=B_t, D_t=D_t)

dd_rate_prediction_lgb = lgb_prediction(test_data=test_data)
print('Predicted dust deposition rate = {} kgm$^{-2}$s$^{-1}$'.format(dd_rate_prediction_lgb))
print('*'*20)
dd_rate_prediction_svr = svr_prediction(test_data=test_data)
print('Predicted dust deposition rate = {} kgm$^{-2}$s$^{-1}$'.format(dd_rate_prediction_svr))
print('*'*20)