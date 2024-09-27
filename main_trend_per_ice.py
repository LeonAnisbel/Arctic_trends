import numpy as np
import statsmodels.api as sm
import pickle
import Trend_all_arctic
from process_statsmodels import process_array_slope_per_ice
import read_data, utils
import plots

ftype = np.float64
if __name__ == '__main__':
    # months = [7, 8, 9]
    # season='JAS'
    # one_month = [9]

    months = [4, 5, 6]
    season='AMJ'
    one_month = [6]



    variables_info = {
        # 'Sea_ice': {'lim': 1.5, 'unit': '% '},
        # 'Sea_ice_1m': {'lim': 1.5, 'unit': '% '},
        # 'Sea_ice_area_px': {'lim': 1.5, 'unit': '% '},
        # 'Sea_ice_area_px_1m': {'lim': 1.5, 'unit': '% '},
        # 'SST': {'lim': 0.1, 'unit': '$^{o}C$ '},
        # 'NPP': {'lim': 1, 'unit': '$mmol\ C$ ${m^{-2}}$ ${d^{-1}}$ '},
        # 'DIN': {'lim': 0.01, 'unit': '$mmol\ C$ ${m^{-2}}$ ${d^{-1}}$ '},
        'AER_F_POL': {'lim': 0.01, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_PRO': {'lim': 0.1, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_LIP': {'lim': 2, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        'AER_F_SS': {'lim': 4, 'unit': 'ng ${m^{-2}}$ ${s^{-1}}$'},
        # 'AER_U10': {'lim': 4, 'unit': 'm ${s^{-1}}$'},
        # 'AER_SST': {'lim': 4, 'unit': '$^{o}C$'},
        # 'AER_SIC': {'lim': 4, 'unit': '%'},
        # 'AER_SIC_1m': {'lim': 4, 'unit': '%'},
        'AER_POL': {'lim': 0.01, 'unit': 'ng ${m^{-3}}$ '},
        'AER_PRO': {'lim': 0.1, 'unit': 'ng ${m^{-3}}$ '},
        'AER_LIP': {'lim': 2, 'unit': 'ng ${m^{-3}}$ '},
        'AER_SS': {'lim': 4, 'unit': 'ng ${m^{-3}}$ '},
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

    C_ss = read_data.read_each_aerosol_data(months, 'SS_AS', 'SS_AS_t63', 1e12)
    C_pol = read_data.read_each_aerosol_data(months, 'POL_AS', 'POL_AS_t63', 1e12)
    C_pro = read_data.read_each_aerosol_data(months, 'PRO_AS', 'PRO_AS_t63', 1e12)
    C_lip = read_data.read_each_aerosol_data(months, 'LIP_AS', 'LIP_AS_t63', 1e12)

    C_ss_emi = read_data.read_each_aerosol_data(months, 'emi_SS', 'emi', 1e12, two_dim=True)
    C_pol_emi = read_data.read_each_aerosol_data(months, 'emi_POL', 'emi', 1e12, two_dim=True)
    C_pro_emi = read_data.read_each_aerosol_data(months, 'emi_PRO', 'emi', 1e12, two_dim=True)
    C_lip_emi = read_data.read_each_aerosol_data(months, 'emi_LIP', 'emi', 1e12, two_dim=True)

    # sst_aer = read_data.read_each_aerosol_data(months, 'tsw', 'echam', 1, two_dim=True) - 273.16
    # u10 = read_data.read_each_aerosol_data(months, 'wind10', 'echam', 1, two_dim=True)
    seaice_aer = read_data.read_each_aerosol_data(months, 'seaice', 'echam', 1, two_dim=True)  # * 100

    data_omf = read_data.read_omf_data()  # * 100
    tot_omf = (data_omf['OMF_POL'] +
               data_omf['OMF_LIP'] +
               data_omf['OMF_PRO'])

    C_pcho, C_dcaa, C_pl, C_ice, C_temp, C_NPP, C_DIN = read_data.read_ocean_data()
    bx_size = abs(C_ice.lat.values[1] - C_ice.lat.values[0])
    grid_bx_area = (bx_size * 110.574) * (
            bx_size * 111.320 * np.cos(np.deg2rad(C_ice.lat)))  # from % sea ice of grid to km2
    C_ice_area_px = C_ice * grid_bx_area

    tot_biom_oc = C_pcho + C_dcaa + C_pl
    list_variables = [
        # C_ice * 100, C_ice * 100,
        # C_ice_area_px, C_ice_area_px,
        # C_temp, C_NPP,
        # C_DIN,
        C_pol_emi,
        C_pro_emi, C_lip_emi, C_ss_emi,
        # u10, sst_aer, seaice_aer, seaice_aer,
        C_pol, C_pro, C_lip, C_ss,
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
        else:
            data_month_ice_reg, _, _ = utils.pick_month_var_reg(C_ice,  # * 100,
                                                                months_list,
                                                                aer_conc=aer_conc)

        data_month_reg, lat, lon = utils.pick_month_var_reg(dict_var['data'],
                                                            months_list,
                                                            aer_conc=aer_conc)

        variables_info[var_na]['data_season_reg'] = data_month_reg
        variables_info[var_na]['lat'] = lat
        variables_info[var_na]['lon'] = lon

        X_aux = np.ma.masked_where(data_month_ice_reg < 0.2, data_month_ice_reg)
        X_aux = X_aux.filled(np.nan)

        # import cartopy.crs as ccrs
        # import matplotlib.pyplot as plt
        # from matplotlib import ticker as mticker
        # fig, ax = plt.subplots(nrows=1,
        #                        ncols=1,
        #                        sharex=True,
        #                        subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
        # ax.set_extent([-180, 180, 60, 90],
        #               ccrs.PlateCarree())
        # cmap = plt.get_cmap('Blues', 15)
        # cb = ax.pcolormesh(lon,
        #                    lat,
        #                    X_aux,
        #                    cmap=cmap,
        #                    transform=ccrs.PlateCarree())
        # ax.coastlines(color='darkgray')
        # gl = ax.gridlines(draw_labels=True, )
        # gl.ylocator = mticker.FixedLocator([65, 75, 85])
        # plt.savefig(f'TEST.png')

        # X  = X.fillna(0)
        X = X_aux.astype(ftype)

        Y_aux = np.ma.masked_where(data_month_ice_reg < 0.2, data_month_reg)
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

#        Trend_all_arctic.trend_aver_per_reg(variables_info, var_na, data_month_reg, data_month_ice_reg, var_na[:3], per_unit_sic=True)

    with open(f"TrendsDict_per_ice_{season}.pkl", "wb") as myFile:
        pickle.dump(variables_info, myFile)

    # with open("TrendsDict_seaice.txt", "wb") as myFile:
    #     pickle.dump(variables_info, myFile)
