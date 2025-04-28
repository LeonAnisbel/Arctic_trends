import gc
import numpy as np
import statsmodels.api as sm
import pickle

import xarray as xr
import Trend_all_arctic
import global_vars
from process_statsmodels import process_array_slope
import read_data, utils
import plots
from utils import calculate_anomaly

ftype = np.float64


def initialize_array():
    return np.empty((x_lat, y_lon), dtype=ftype)


if __name__ == '__main__':
    season = global_vars.season_to_analise
    season_dict = global_vars.seasons_info[season]
    months = season_dict['months']
    one_month = season_dict['one_month']


# Read in aerosol concentration
    C_conc = []
    C_conc_anomaly = []
    var_ids = ['POL_AS', 'PRO_AS', 'LIP_AS', 'SS_AS']
    for c_elem in range(len(var_ids)):
        conc, _ = read_data.read_each_aerosol_data(months,
                                                   var_ids[c_elem],
                                                   'tracer',
                                                   1e12,
                                                   two_dim=False)

        C_conc.append(conc)
        C_conc_anomaly.append(calculate_anomaly(conc))

    C_tot_conc = C_conc[0] + C_conc[1] + C_conc[2]
    C_conc_ssa = C_tot_conc + C_conc[3]

    C_tot_conc_anomaly = calculate_anomaly(C_tot_conc)
    C_conc_ssa_anomaly = calculate_anomaly(C_conc_ssa)

# Read in aerosol emission mass flux and flux per month (_m)
    C_emi = []
    C_emi_m = []
    C_emi_anomaly = []
    var_ids = ['emi_POL', 'emi_PRO', 'emi_LIP', 'emi_SS']
    for c_elem in range(len(var_ids)):
        emi, emi_m = read_data.read_each_aerosol_data(months,
                                                      var_ids[c_elem],
                                                      'emi',
                                                      1e12,
                                                      two_dim=True)
        C_emi.append(emi)
        C_emi_m.append(emi_m)
        C_emi_anomaly.append(calculate_anomaly(emi))

    C_tot_emi = C_emi[0] + C_emi[1] + C_emi[2]
    C_emi_ssa = C_tot_emi +  C_emi[3]

    C_tot_emi_anomaly = calculate_anomaly(C_tot_emi)
    C_emi_ssa_anomaly = calculate_anomaly(C_emi_ssa)

    C_tot_emi_m = C_emi_m[0] + C_emi_m[1] + C_emi_m[2]
    C_ssa_emi_m =  C_tot_emi_m +  C_emi_m[3]

    print('Finished reading aerosol emission data')

# Read in emission drivers
    sst_aer, _ = read_data.read_each_aerosol_data(months,
                                                  'tsw',
                                                  'echam',
                                                  1,
                                                  two_dim=True)
    sst_aer_K = sst_aer - 273.16

    u10, _ = read_data.read_each_aerosol_data(months,
                                              'velo10m',
                                              'vphysc',
                                              1,
                                              two_dim=True)
    seaice_aer, _ = read_data.read_each_aerosol_data(months,
                                                     'seaice',
                                                     'echam',
                                                     1,
                                                     two_dim=True)
    gbox_area, _ = read_data.read_each_aerosol_data(months,
                                                    'gboxarea',
                                                    'emi',
                                                    1,
                                                    two_dim=True)
    C_ice_aer_area_px = seaice_aer * gbox_area * 1.e-6  # from m2 to km2
    print('Finished reading aerosol emission drivers')

# Read in aerosol organic mass fraction (OMF)
    data_omf = read_data.read_omf_data() * 100
    tot_omf = (data_omf['OMF_POL'] +
               data_omf['OMF_LIP'] +
               data_omf['OMF_PRO'])
    print('Finished reading OMF  data')

# Read in biomolecule ocean concentration and other biogeochemical indicators from the BGC model (FESOM-REcoM)
    C_pcho, C_dcaa, C_pl, C_ice, C_temp, C_NPP, C_DIN = read_data.read_ocean_data()
    C_ice_area_px = utils.compute_seaice_area_px(C_ice) # sea ice from FESOM-RECOM
    tot_biom_conc = C_pcho + C_dcaa + C_pl
    C_NPP_anomaly = calculate_anomaly(C_NPP)
    print('Finished reading biomolecule concentration and SIC from FESOm-REcoM data')

    list_variables = [
        C_emi[0], C_emi[1], C_emi[2], C_tot_emi, C_emi[3], C_emi_ssa,
        C_emi_anomaly[0], C_emi_anomaly[1], C_emi_anomaly[2], C_tot_emi_anomaly, C_emi_anomaly[3], C_emi_ssa_anomaly,
        C_emi_m[0], C_emi_m[1], C_emi_m[2], C_tot_emi_m, C_emi_m[3], C_ssa_emi_m,
        u10, sst_aer_K, seaice_aer * 100, C_ice_aer_area_px, seaice_aer * 100,
        C_conc[0], C_conc[1], C_conc[2], C_tot_conc, C_conc[3], C_conc_ssa,
        C_conc_anomaly[0], C_conc_anomaly[1], C_conc_anomaly[2], C_tot_conc_anomaly, C_conc_anomaly[3], C_conc_ssa_anomaly,
        data_omf['OMF_POL'], data_omf['OMF_PRO'], data_omf['OMF_LIP'], tot_omf,
        C_pcho, C_dcaa, C_pl, tot_biom_conc,
        C_ice * 100, C_ice * 100, C_ice_area_px,  C_ice_area_px,
        C_temp, C_NPP, C_NPP_anomaly, C_DIN,
        ]
    variables_info = utils.create_var_info_dict()

    da_type = global_vars.data_type
    file_name = da_type
    print(da_type)
    for idx, var_na in enumerate(list(variables_info.keys())):
        variables_info[var_na]['orig_data'] = list_variables[idx]

    for var_na, dict_var in variables_info.items():
        # var_na = list(variables_info['var_names'].keys())[i]
        print('Computing ' + var_na + ' trend')

        aer_conc = False
        if var_na[:3] == 'AER':
            aer_conc = True
            if da_type == 'log_data':
                da_tmp = variables_info[var_na]['orig_data'].where(variables_info[var_na]['orig_data'] > 0,
                                                   other=np.nan)
                variables_info[var_na][da_type] = np.log(da_tmp)
        else:
            da_type = 'orig_data'

        if var_na == 'Sea_ice_1m' or var_na == 'Sea_ice_area_px_1m' :#or var_na == 'AER_SIC_1m':
            data_reg = dict_var[da_type].where(dict_var[da_type].lat > 60,
                                               drop=True)
            da_months = data_reg.where((data_reg.time.dt.month >= months[0]) &
                                       (data_reg.time.dt.month <= months[-1]),
                                       drop=True)
            variables_info[var_na]['data_season_reg'] = da_months
        else:
            data_month_arctic, lat, lon = utils.pick_month_var_reg(dict_var[da_type],
                                                                months,
                                                                aer_conc=aer_conc)

            variables_info[var_na]['data_season_reg'] = data_month_arctic
            variables_info[var_na]['data_time_mean'] = data_month_arctic.mean('time',
                                                                           skipna=True)
            da_compute = data_month_arctic.compute()
            variables_info[var_na]['data_time_median'] = da_compute.median('time',
                                                                           skipna=True)
            del da_compute

            variables_info[var_na]['lat'] = lat
            variables_info[var_na]['lon'] = lon

            X = data_month_arctic.time.values.astype(ftype)  # dt.year.
            X = sm.add_constant(X)
            Y = data_month_arctic.values.astype(ftype)

            x_lat, y_lon = Y.shape[1:]

            slope = initialize_array()
            p_value = initialize_array()
            intercept = initialize_array()
            trend = initialize_array()
            tau = initialize_array()
            significance = initialize_array()

            process_array_slope(Y,
                                X,
                                slope,
                                p_value,
                                intercept,
                                trend,
                                tau,
                                significance)

            variables_info[var_na]['slope'] = slope
            variables_info[var_na]['pval'] = p_value
            variables_info[var_na]['intercept'] = intercept
            variables_info[var_na]['trend'] = trend
            variables_info[var_na]['tau'] = tau
            variables_info[var_na]['significance'] = significance

            # plots.plot_trend(slope,
            #                  p_value,
            #                  lat,
            #                  lon,
            #                  'Trend_' + var_na + '.png',
            #                  dict_var['lim'],
            #                  dict_var['unit'])

            Trend_all_arctic.trend_aver_per_reg(variables_info,
                                                var_na,
                                                data_month_arctic,
                                                data_month_arctic,
                                                var_na[:3],
                                                per_unit_sic=False)

            gc.collect()

    with open(f"TrendsDict_{season}_{file_name}.pkl", "wb") as myFile:
        pickle.dump(variables_info, myFile)
