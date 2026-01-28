import pickle

from Utils_functions import utils, global_vars
from Trend_maps import plot_biomolecule_trend, plot_aerosol_trend

if __name__ == '__main__':
    season = global_vars.season_to_analise

    with open(f"TrendsDict_{season}_orig_data.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)
    seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice', get_min_area=True)

    with open(f"TrendsDict_per_ice_{season}.pkl", "rb") as myFile:
        variables_info_seaice = pickle.load(myFile)

    decade = '1990-2019'

    plot_aerosol_trend.plot_dcaa_spring_summer()
    plot_biomolecule_trend.plot_dcaa_spring_summer()
    plot_biomolecule_trend.plot_trend(variables_info_yr, seaice, season)
    plot_aerosol_trend.plot_trend_emission(variables_info_seaice, variables_info_yr, seaice, season, decade)
    plot_aerosol_trend.plot_trend_aer_concentration(variables_info_yr, seaice, season)
    plot_aerosol_trend.plot_inp_burden(variables_info_yr, seaice, season)
