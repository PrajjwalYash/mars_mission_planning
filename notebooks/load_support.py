import numpy as np
def load_support_duration(ecl_dur, sol_pow_smooth,ecl_load, sunlit_load, payload, bat_size, disch_vlt, bat_eff, panel_size, panel_eff):
    bat_drain = ecl_dur[:-1]*ecl_load
    bat_dod = ((bat_drain/disch_vlt)/bat_size)*100
    bat_dod[bat_dod>100]=100
    sunlit_req = sunlit_load*(24.66-ecl_dur[1:])
    gen = 1000*sol_pow_smooth[1:]*panel_size*panel_eff
    excess_gen = gen - sunlit_req-(1/bat_eff)*bat_drain
    payload_duration = excess_gen/payload
    payload_duration = np.array(payload_duration)
    payload_duration[payload_duration>24.66] = 24.66
    payload_duration[payload_duration<0] = 0
    return payload_duration, bat_dod