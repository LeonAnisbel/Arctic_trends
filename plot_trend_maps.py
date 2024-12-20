import pickle
import plots, utils
import numpy as np
import plot_aerosol_trend

if __name__ == '__main__':

    season = 'JAS'
    # season='AMJ'
    # season = 'JJA'

    with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)
    seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice')

    seaice_lin = variables_info_yr['AER_SIC_area_px']
    biomol = variables_info_yr['AER_F_tot_m']  # AER_LIP #AER_F_tot_yr

    region = ['Chukchi Sea', 'Greenland & Norwegian Sea', 'East-Siberian Sea', 'Kara Sea']
    for reg in region:

        sst = variables_info_yr['AER_SST'][reg]
        wind = variables_info_yr['AER_U10'][reg]
        decades = ['1990-2004', '2005-2019']

        for idx, dec in enumerate(decades):
            print(reg, dec, 'mean SST', sst[dec]['data_aver_reg'].mean().values, )
            print(reg, dec, 'mean wind', wind[dec]['data_aver_reg'].mean().values, )

    with open(f"TrendsDict_per_ice_{season}.pkl", "rb") as myFile:
        variables_info_seaice = pickle.load(myFile)

    decade = '1990-2019'

    plot_aerosol_trend.plot_trend_aer_concentration(variables_info_yr, seaice, season)
    plot_aerosol_trend.plot_trend_emission(variables_info_seaice, variables_info_yr, seaice, season, decade)
