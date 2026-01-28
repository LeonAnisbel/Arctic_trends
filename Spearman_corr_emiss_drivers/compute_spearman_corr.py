import pickle
from Utils_functions import read_data, utils, global_vars
import numpy as np
from Utils_functions.utils import calculate_anomaly
import xarray as xr
from scipy import stats

ftype = np.float64


def compute_corr_matrix(emi_var, sic_aer, sst_aer, u10_aer, C_tot_omf_day):
    """Computes the correlation of the emission and emission drivers over a 30-yr period for each grid point
    :return None"""

    lat_vals = sic_aer.lat.values
    lon_vals = sic_aer.lon.values
    lat_len = len(lat_vals)
    lon_len = len(lon_vals)
    matrix_sst = np.zeros((lat_len, lon_len))
    matrix_sic = np.zeros((lat_len, lon_len))
    matrix_omf = np.zeros((lat_len, lon_len))
    matrix_u10 = np.zeros((lat_len, lon_len))
    print(lat_len, lon_len, matrix_sic.shape)

    for loid, lo in enumerate(lon_vals):
        for laid, la in enumerate(lat_vals):
            i_val_emi = emi_var.sel(lat=la, lon=lo)
            i_val_u10 = u10_aer.sel(lat=la, lon=lo)
            i_val_sic = sic_aer.sel(lat=la, lon=lo)
            i_val_sst = sst_aer.sel(lat=la, lon=lo)
            i_val_omf = C_tot_omf_day.sel(lat=la, lon=lo)

            i_val_u10 = apply_sic_mask(i_val_sic, i_val_u10)
            i_val_sst = apply_sic_mask(i_val_sic, i_val_sst)
            i_val_omf = apply_sic_mask(i_val_sic, i_val_omf)
            i_val_sic = apply_sic_mask(i_val_sic, i_val_sic)

            matrix_sic = compute_corr_fill_matrix(i_val_emi,i_val_sic, laid, loid, matrix_sic)
            matrix_sst = compute_corr_fill_matrix(i_val_emi,i_val_sst, laid, loid, matrix_sst)
            matrix_omf = compute_corr_fill_matrix(i_val_emi,i_val_omf, laid, loid, matrix_omf)
            matrix_u10 = compute_corr_fill_matrix(i_val_emi,i_val_u10, laid, loid, matrix_u10)

    data_ds = xr.Dataset(
        data_vars=dict(
            sic=(["lat", "lon"], matrix_sic),
            sst=(["lat", "lon"], matrix_sst),
            u10=(["lat", "lon"], matrix_u10),
            omf=(["lat", "lon"], matrix_omf),

        ),
        coords=dict(
            lon=("lon", lon_vals),
            lat=("lat", lat_vals),
        ),
    )

    with open(f"Spearman_corr_emiss_drivers_{season}.pkl", "wb") as myFile:
        pickle.dump(data_ds, myFile)

def get_reg_da(v, idx):
    """Extracts a subgrid given a predefine region definition """
    conditions = utils.get_conds(v.lat, v.lon)
    reg_sel_vals_driver = utils.get_var_reg(v, conditions[idx])
    return reg_sel_vals_driver

def get_spearman(emi, driver):
    """Computes Spearman correlation and returns pvalue"""
    model = stats.spearmanr(emi, driver)
    pval = model.pvalue
    rsval = round(model.statistic, 2)
    if pval > 0.05:
        rsval = np.nan
    return rsval

def compute_corr_per_reg(variables, var_names, emi):
    """Extract grid data for predefined Arctic regions and computes the correlation of the spatially averaged values of
     emission and the emission drivers over a 30-yr period."""
    oof_aer, sic_aer, sst_aer, u10_aer, omf_aer = variables[0], variables[1], variables[2], variables[3], variables[4]
    dict_regions = utils.regions()
    dict_var_regions = {}
    for na in var_names:
        dict_var_regions[na] = {}

    for idx, reg_na in enumerate(list(dict_regions.keys())):
        print(reg_na)
        reg_sel_vals_emi = get_reg_da(emi, idx)
        reg_sel_vals_sic = get_reg_da(sic_aer, idx)
        reg_sel_vals_sst = get_reg_da(sst_aer, idx)
        reg_sel_vals_u10 = get_reg_da(u10_aer, idx)
        reg_sel_vals_omf = get_reg_da(omf_aer, idx)
        reg_sel_vals_oof = get_reg_da(oof_aer, idx)

        reg_sel_vals_sic = reg_sel_vals_sic.mean(dim=['lat','lon'], skipna=True).values
        reg_sel_vals_sst = reg_sel_vals_sst.mean(dim=['lat','lon'], skipna=True).values
        reg_sel_vals_u10 = reg_sel_vals_u10.mean(dim=['lat','lon'], skipna=True).values
        reg_sel_vals_omf = reg_sel_vals_omf.mean(dim=['lat','lon'], skipna=True).values
        reg_sel_vals_emi = reg_sel_vals_emi.sum(dim=['lat','lon'], skipna=True).values
        reg_sel_vals_oof = reg_sel_vals_oof.mean(dim=['lat','lon'], skipna=True).values

        dict_var_regions['SST'][reg_na] = get_spearman(reg_sel_vals_emi, reg_sel_vals_sst)
        dict_var_regions['SIC'][reg_na] = get_spearman(reg_sel_vals_emi, reg_sel_vals_sic)
        dict_var_regions['OMF'][reg_na] = get_spearman(reg_sel_vals_emi, reg_sel_vals_omf)
        dict_var_regions['u10'][reg_na] = get_spearman(reg_sel_vals_emi, reg_sel_vals_u10)
        dict_var_regions['Open Ocean fraction'][reg_na] = get_spearman(reg_sel_vals_emi, reg_sel_vals_oof)

    with open(f"Spearman_corr_emiss_drivers_reg_{season}.pkl", "wb") as myFile:
        pickle.dump(dict_var_regions, myFile)


def compute_corr_fill_matrix(i_val_x, i_val_y, laid, loid, matrix):
    """
    Compute spearman correlation for each grid cell over time
    """
    model = stats.spearmanr(i_val_x, i_val_y)
    pval = model.pvalue
    rsval = round(model.statistic, 2)
    if pval < 0.05:
        matrix[laid, loid] = rsval
    else:
        matrix[laid, loid] = np.nan

    return matrix

def apply_sic_mask(i_val_sic, panel_var):
    # Exclude regions potentially fully covered by ice
    panel_var = np.ma.masked_where(i_val_sic.values > 0.99,
                                   panel_var.values)
    panel_var_nan = panel_var.filled(np.nan)
    return panel_var_nan

if __name__ == '__main__':
    season = global_vars.season_to_analise
    season_dict = global_vars.seasons_info[season]
    months = season_dict['months']
    one_month = season_dict['one_month']

# Read in aerosol emission mass flux and flux as daily values
    C_emi_day = []
    C_emi_anomaly = []
    var_ids_emi = ['emi_POL', 'emi_PRO', 'emi_LIP', 'emi_SS']
    for c_elem in range(len(var_ids_emi)):
        emi_day_reg = read_data.read_daily_data(months,
                                                var_ids_emi[c_elem],
                                                      'emi',
                                                1e12,
                                                emi_gridbox=True,
                                                two_dim=True)

        C_emi_day.append(emi_day_reg)

        C_emi_dict_anom = calculate_anomaly(emi_day_reg)
        C_emi_anomaly.append(C_emi_dict_anom)


    print('Finished reading aerosol emission data')

# Read in emission drivers
    sst_aer = read_data.read_daily_data(months,
                                                  'tsw',
                                                  'vphysc',
                                        1,
                                        two_dim=True)
    u10_aer = read_data.read_daily_data(months,
                                              'velo10m',
                                              'vphysc',
                                        1,
                                        two_dim=True)
    sic_aer = read_data.read_daily_data(months,
                                                     'seaice',
                                                     'echam',
                                        1,
                                        two_dim=True)

    C_omf_day = []
    var_ids_omf = ['OMF_POL', 'OMF_PRO', 'OMF_LIP']
    for c_elem in range(len(var_ids_omf)):
        omf_day_reg = read_data.read_daily_data(months,
                                                var_ids_omf[c_elem],
                                                      'ham',
                                                1,
                                                two_dim=True)
        C_omf_day.append(omf_day_reg)


    C_tot_emi_day = C_emi_day[0] + C_emi_day[1] + C_emi_day[2]
    C_emi_tot_dict_anom = calculate_anomaly(C_tot_emi_day)
    C_tot_omf_day = C_omf_day[0] + C_omf_day[1] + C_omf_day[2]
    openocean_aer = 1 - sic_aer

    # computes the correlation of the spatially averaged values
    compute_corr_per_reg([openocean_aer, sic_aer, sst_aer, u10_aer, C_tot_omf_day],
                         ['Open Ocean fraction', 'SIC','SST', 'u10', 'OMF'],
                         C_tot_emi_day, )

    # computes the correlation over the whole grid
    compute_corr_matrix(C_tot_emi_day, sic_aer, sst_aer, u10_aer, C_tot_omf_day)
