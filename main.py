import numpy as np
import statsmodels.api as sm
import pickle
import Trend_all_arctic
from process_statsmodels import process_array_slope
import read_data, utils
import plots

ftype = np.float64
if __name__ == '__main__':
    months = [7, 8, 9]
    season = 'JAS'
    one_month = [9]

    # months = [6, 7, 8]
    # season='JJA'
    # one_month = [8]

    # months = [4, 5, 6]
    # season='AMJ'
    # one_month = [6]

    # months = [1, 2, 3]
    # season='JFM'
    # one_month = [1]

    variables_info = {
        'Sea_ice': {'lim': 1.5, 'unit': '% '},
        'Sea_ice_1m': {'lim': 1.5, 'unit': '% '},
        'Sea_ice_area_px': {'lim': 1.5, 'unit': '% '},
        # 'Sea_ice_area_px_1m': {'lim': 1.5, 'unit': '% '},
        'SST': {'lim': 0.1, 'unit': '$^{o}C$ '},
        'NPP': {'lim': 1, 'unit': '$mmol\ C$ ${m^{-2}}$ ${d^{-1}}$ '},
        'DIN': {'lim': 0.01, 'unit': '$mmol\ C$ ${m^{-2}}$ ${d^{-1}}$ '},
        'AER_F_POL': {'lim': 0.01, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_PRO': {'lim': 0.1, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_LIP': {'lim': 2, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_tot': {'lim': 2, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_SS': {'lim': 4, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_POL_yr': {'lim': 0.01, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_PRO_yr': {'lim': 0.1, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_LIP_yr': {'lim': 2, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_tot_yr': {'lim': 2, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_SS_yr': {'lim': 4, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},

        'AER_U10': {'lim': 4, 'unit': 'm ${s^{-1}}$'},
        'AER_SST': {'lim': 4, 'unit': '$^{o}C$'},
        'AER_SIC': {'lim': 4, 'unit': '%'},
        'AER_SIC_area_px': {'lim': 4, 'unit': '%'},
        'AER_SIC_1m': {'lim': 4, 'unit': '%'},
        # 'AER_POL': {'lim': 0.01, 'unit': 'ng ${m^{-3}}$ '},
        # 'AER_PRO': {'lim': 0.1, 'unit': 'ng ${m^{-3}}$ '},
        # 'AER_LIP': {'lim': 2, 'unit': 'ng ${m^{-3}}$ '},
        # 'AER_SS': {'lim': 4, 'unit': 'ng ${m^{-3}}$ '},
        'OMF_POL': {'lim': 0.003, 'unit': '% '},
        'OMF_PRO': {'lim': 0.02, 'unit': '% '},
        'OMF_LIP': {'lim': 0.5, 'unit': '% '},
        'OMF_tot': {'lim': 0.8, 'unit': '% '},
        'PCHO': {'lim': 0.05, 'unit': '$mmol\ C$ ${m^{-3}}$ '},
        'DCAA': {'lim': 0.02, 'unit': '$mmol\ C$ ${m^{-3}}$ '},
        'PL': {'lim': 0.01, 'unit': '$mmol\ C$ ${m^{-3}}$ '},
        'Biom_tot': {'lim': 0.1, 'unit': '$mmol\ C$ ${m^{-3}}$ '}
    }

    # C_pol, C_pro, C_lip, C_ss = read_data.read_aerosol_data(months)

    # C_ss = read_data.read_each_aerosol_data(months, 'SS_AS', 'SS_AS_t63', 1e12)
    # C_pol = read_data.read_each_aerosol_data(months, 'POL_AS', 'POL_AS_t63', 1e12)
    # C_pro = read_data.read_each_aerosol_data(months, 'PRO_AS', 'PRO_AS_t63', 1e12)
    # C_lip = read_data.read_each_aerosol_data(months, 'LIP_AS', 'LIP_AS_t63', 1e12)

    C_ss_emi = read_data.read_each_aerosol_data(months,
                                                'emi_SS',
                                                'emi',
                                                1e12,
                                                two_dim=True)
    C_pol_emi = read_data.read_each_aerosol_data(months,
                                                 'emi_POL',
                                                 'emi',
                                                 1e12,
                                                 two_dim=True)
    C_pro_emi = read_data.read_each_aerosol_data(months,
                                                 'emi_PRO',
                                                 'emi',
                                                 1e12,
                                                 two_dim=True)
    C_lip_emi = read_data.read_each_aerosol_data(months,
                                                 'emi_LIP',
                                                 'emi',
                                                 1e12,
                                                 two_dim=True)
    C_tot_emi = C_pol_emi + C_pro_emi + C_lip_emi

    gbox_area = read_data.read_each_aerosol_data(months,
                                                 'gboxarea',
                                                 'emi',
                                                 1,
                                                 two_dim=True)

    fac_sec_to_yr = 31557600
    fac_ng_to_tg = 1e-12  # to Kg, 1e-21 to Tg
    unit_factor = gbox_area * fac_ng_to_tg  # kg/yr
    # (ng/s) * fac_ng_to_tg #(Tg/s) * fac_sec_to_yr # Tg/yr

    C_ss_emi_yr = C_ss_emi * unit_factor
    C_pol_emi_yr = C_pol_emi * unit_factor
    C_pro_emi_yr = C_pro_emi * unit_factor
    C_lip_emi_yr = C_lip_emi * unit_factor
    C_tot_emi_yr = C_tot_emi * unit_factor

    print('Finished reading aerosol emission data')

    sst_aer = read_data.read_each_aerosol_data(months,
                                               'tsw',
                                               'echam',
                                               1,
                                               two_dim=True) - 273.16
    u10 = read_data.read_each_aerosol_data(months,
                                           'velo10m',
                                           'vphysc',
                                           1,
                                           two_dim=True)
    seaice_aer = read_data.read_each_aerosol_data(months,
                                                  'seaice',
                                                  'echam',
                                                  1,
                                                  two_dim=True)

    # seaice_aer = read_data.read_each_aerosol_data(months, 'seaice', 'echam_regular_grid', 1, two_dim=True)
    C_ice_aer_area_px = seaice_aer * gbox_area * 1.e-6  # from m2 to km2

    print('Finished reading SST, SIC and  wind data', C_ice_aer_area_px.max().values, C_ice_aer_area_px.mean().values)

    data_omf = read_data.read_omf_data() * 100
    tot_omf = (data_omf['OMF_POL'] +
               data_omf['OMF_LIP'] +
               data_omf['OMF_PRO'])
    print('Finished reading OMF  data')

    C_pcho, C_dcaa, C_pl, C_ice, C_temp, C_NPP, C_DIN = read_data.read_ocean_data()
    C_ice_area_px = utils.compute_seaice_area_px(C_ice)
    print('C_ice_area_px', C_ice_area_px.max().values, C_ice_area_px.mean().values)
    print('Finished reading biomolecule concentration and SIC from FESOm-REcoM data')

    tot_biom_oc = C_pcho + C_dcaa + C_pl
    list_variables = [
        C_ice * 100, C_ice * 100,
        C_ice_area_px,  # C_ice_area_px,
        C_temp, C_NPP,
        C_DIN,
        C_pol_emi, C_pro_emi,
        C_lip_emi,
        C_tot_emi, C_ss_emi,
        C_pol_emi_yr, C_pro_emi_yr, C_lip_emi_yr, C_tot_emi_yr, C_ss_emi_yr,
        u10, sst_aer,
        seaice_aer * 100, C_ice_aer_area_px, seaice_aer * 100,
        # C_pol, C_pro, C_lip, C_ss,
        data_omf['OMF_POL'],
        data_omf['OMF_PRO'],
        data_omf['OMF_LIP'],
        tot_omf,
        C_pcho,
        C_dcaa,
        C_pl,
        tot_biom_oc,

    ]

    for idx, var_na in enumerate(list(variables_info.keys())):
        variables_info[var_na]['data'] = list_variables[idx]

    for var_na, dict_var in variables_info.items():
        # var_na = list(variables_info['var_names'].keys())[i]
        print('Computing ' + var_na + ' trend')

        aer_conc = False
        if var_na[:3] == 'AER':
            aer_conc = True

        if var_na == 'Sea_ice_1m' or var_na == 'Sea_ice_area_px_1m' or var_na == 'AER_SIC_1m':
            months_list = one_month
        else:
            months_list = months

        data_month_reg, lat, lon = utils.pick_month_var_reg(dict_var['data'],
                                                            months_list,
                                                            aer_conc=aer_conc)
        variables_info[var_na]['data_season_reg'] = data_month_reg

        variables_info[var_na]['lat'] = lat
        variables_info[var_na]['lon'] = lon

        X = data_month_reg.time.values.astype(ftype)  # dt.year.
        X = sm.add_constant(X)
        Y = data_month_reg.values.astype(ftype)

        x_lat, y_lon = Y.shape[1:]

        slope = np.empty((x_lat, y_lon),
                         dtype=ftype)
        p_value = np.empty((x_lat, y_lon),
                           dtype=ftype)
        intercept = np.empty((x_lat, y_lon),
                             dtype=ftype)

        process_array_slope(Y,
                            X,
                            slope,
                            p_value,
                            intercept)

        variables_info[var_na]['slope'] = slope
        variables_info[var_na]['pval'] = p_value
        variables_info[var_na]['intercept'] = intercept
        # plots.plot_trend(slope,
        #                  p_value,
        #                  lat,
        #                  lon,
        #                  'Trend_' + var_na + '.png',
        #                  dict_var['lim'],
        #                  dict_var['unit'])

        Trend_all_arctic.trend_aver_per_reg(variables_info,
                                            var_na,
                                            data_month_reg,
                                            data_month_reg,
                                            var_na[:3],
                                            per_unit_sic=False)

    with open(f"TrendsDict_{season}.pkl", "wb") as myFile:
        pickle.dump(variables_info, myFile)

