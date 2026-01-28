import numpy as np
import pickle
from process_statsmodels import process_array_slope_per_ice
from Utils_functions import read_data, utils, global_vars

ftype = np.float64
# computes the trends of emissions per unit of sea ice
if __name__ == '__main__':
    season = global_vars.season_to_analise
    season_dict = global_vars.seasons_info[season]
    months = season_dict['months']
    one_month = season_dict['one_month']

    variables_info = {
        'AER_F_POL': {'lim': 0.01, 'unit': 'ng m${^{-2}}$ s${^{-1}}$'},
        'AER_F_PRO': {'lim': 0.1, 'unit': 'ng m${^{-2}}$ s${^{-1}}$'},
        'AER_F_LIP': {'lim': 2, 'unit': 'ng m${^{-2}}$ s${^{-1}}$'},
        'AER_F_SS': {'lim': 4, 'unit': 'ng m${^{-2}}$ s${^{-1}}$'},
    }

    C_emi = []
    var_ids = ['emi_POL', 'emi_PRO', 'emi_LIP', 'emi_SS']
    for c_elem in range(len(var_ids)):
        emi, _ = read_data.read_each_aerosol_data(months,
                                                  var_ids[c_elem],
                                                      'emi',
                                                  1e12,
                                                  two_dim=True)
        C_emi.append(emi)

    C_tot_emi = C_emi[0] + C_emi[1] + C_emi[2]

    seaice_aer, _ = read_data.read_each_aerosol_data(months,
                                                     'seaice',
                                                     'echam',
                                                     1,
                                                     two_dim=True)  # * 100

    list_variables = [
        C_emi[0], C_emi[1], C_emi[2], C_emi[3],
    ]

    for idx, var_na in enumerate(list(variables_info.keys())):
        variables_info[var_na]['data'] = list_variables[idx]

    for var_na, dict_var in variables_info.items():
        print('Computing ' + var_na + ' trend')

        if var_na == 'Sea_ice_1m' or var_na == 'Sea_ice_area_px_1m' or var_na == 'AER_SIC_1m':
            months_list = one_month
        else:
            months_list = months

        aer_conc = False
        if var_na[:3] == 'AER':
            aer_conc = True
            data_month_ice_reg, _, _ = utils.pick_month_var_reg(seaice_aer,
                                                                months_list,
                                                                aer_conc=aer_conc)
            trends_per_regions = True

        data_month_reg, lat, lon = utils.pick_month_var_reg(dict_var['data'],
                                                            months_list,
                                                            aer_conc=aer_conc)

        variables_info[var_na]['data_season_reg'] = data_month_reg
        variables_info[var_na]['lat'] = lat
        variables_info[var_na]['lon'] = lon

        X_aux = np.ma.masked_where(data_month_ice_reg < 0.2,
                                   data_month_ice_reg)
        X_aux = X_aux.filled(np.nan)
        X = X_aux.astype(ftype)

        Y_aux = np.ma.masked_where(data_month_ice_reg < 0.2,
                                   data_month_reg)
        Y_aux = Y_aux.filled(np.nan)
        Y = Y_aux.astype(ftype)

        x_lat, y_lon = Y.shape[1:]

        slope = np.empty((x_lat, y_lon),
                         dtype=ftype)
        p_value = np.empty((x_lat, y_lon),
                           dtype=ftype)
        intercept = np.empty((x_lat, y_lon),
                             dtype=ftype)

        process_array_slope_per_ice(Y,
                                    X,
                                    slope,
                                    p_value,
                                    intercept)

        variables_info[var_na]['slope'] = slope
        variables_info[var_na]['pval'] = p_value
        variables_info[var_na]['intercept'] = intercept


    with open(f"TrendsDict_per_ice_{season}.pkl", "wb") as myFile:
        pickle.dump(variables_info, myFile)