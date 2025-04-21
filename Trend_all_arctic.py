import numpy as np
import statsmodels.api as sm
import read_data, utils
import xarray as xr
import pickle
import warnings
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in double_scalars",
    module="statsmodels.regression.linear_model"
)
ftype = np.float64
def trend_aver_per_reg(variables_info, var_na, data_month_reg, data_month_ice_reg, var_type, per_unit_sic=False):
    lon_360 = data_month_reg.lon.data
    if var_type == 'AER':
        lon = ((lon_360 + 180) % 360) - 180
    else:
        lon = lon_360

    decades = [[1990,1999], [2000, 2009], [2010, 2019], [1990, 2019]]
    decades_idx = [[0,9], [10,19], [20,29], [0,29]]
    decades_na = ['1990-1999', '2000-2009', '2010-2019', '1990-2019']

    decades = [[1990,2004], [2005, 2019], [1990, 2019]]
    decades_idx = [[0,14], [15,29], [0,29]]
    decades_na = ['1990-2004', '2005-2019', '1990-2019']

    reg_data = utils.regions()
    for reg_na in list(reg_data.keys()):
        variables_info[var_na][reg_na] = {}
        for dec_na in decades_na:
            variables_info[var_na][reg_na][dec_na] = {}

    for idx, reg_na in enumerate(list(reg_data.keys())):
        data_ds = utils.create_ds(data_month_reg, lon)
        conditions = utils.get_conds(data_ds.lat, data_ds.lon)
        reg_sel_vals_whole = utils.get_var_reg(data_ds, conditions[idx])
        for dec_na, dec in enumerate(decades):
            if var_type == 'AER':

                reg_sel_vals = reg_sel_vals_whole.where((reg_sel_vals_whole.time >= decades_idx[dec_na][0])&
                                                  (reg_sel_vals_whole.time <= decades_idx[dec_na][1]),
                                                  drop=True)
            else:
                reg_sel_vals = reg_sel_vals_whole.where((reg_sel_vals_whole.time >= dec[0])&
                                                  (reg_sel_vals_whole.time <= dec[1]),
                                                  drop=True)

            if var_na == 'Sea_ice_area_px' or var_na=='AER_SIC_area_px':
                data_month = reg_sel_vals['data_region'].sum(dim=['lat', 'lon'],
                                                             skipna=True) * 1e-6 # from km2 to millions of km2
                variables_info[var_na][reg_na][decades_na[dec_na]]['data_aver_reg'] = data_month

                data_time_mean = data_month.mean(dim='time', skipna=True)

            else:
                data_month = reg_sel_vals['data_region']#
            # print('Computing ' + var_na + ' trend', data_month.max().values)
                data_latlon_mean = data_month.mean(dim=('lat', 'lon'), skipna=True)
                variables_info[var_na][reg_na][decades_na[dec_na]]['data_aver_reg'] = data_latlon_mean

                data_time_mean = data_latlon_mean.mean(dim='time', skipna=True)
            variables_info[var_na][reg_na][decades_na[dec_na]]['data_time_mean'] = data_time_mean


            if per_unit_sic:
                # X_aux = np.ma.masked_where(data_month_ice_reg.mean(dim = 'time') < 0.5, data_month_ice_reg.mean(dim = 'time'))
                data_ds = utils.create_ds(data_month_ice_reg, lon)
                conditions = utils.get_conds(data_ds.lat, data_ds.lon)
                reg_sel_vals_ice = utils.get_var_reg(data_ds, conditions[idx])
                X_aux = np.ma.masked_where(reg_sel_vals_ice['data_region'] < 0.2,
                                           reg_sel_vals_ice['data_region'])
                X_aux = X_aux.filled(np.nan)
                #print('shape X_Aux',X_aux.shape)
                #print(reg_sel_vals_ice['data_region'])
                data_month_ice = utils.create_ds2(X_aux, reg_sel_vals_ice['data_region'])
                data_month_ice = data_month_ice.mean(dim=('lat', 'lon'),
                                                     skipna=True)
                X = data_month_ice.data

                Y_aux = np.ma.masked_where(reg_sel_vals_ice['data_region'] < 0.2, data_month)
                Y_aux = Y_aux.filled(np.nan)
                data_month_var = utils.create_ds2(Y_aux, reg_sel_vals_ice['data_region'])
                Y = data_month_var.data

            else:
                X = data_month.time.values
                Y = variables_info[var_na][reg_na][decades_na[dec_na]]['data_aver_reg'].values

            X = X.astype(ftype)
            X = sm.add_constant(X)
            Y = Y.astype(ftype)

            y_arr = np.array(Y, dtype=float)
            x_arr = np.array(X, dtype=float)
            mask = ~np.isnan(y_arr)
            n = mask.sum()
            y_clean = y_arr[mask]
            x_clean = x_arr[mask]

            if n >= 2 and not np.allclose(y_clean, y_clean[0]):
                result = sm.OLS(y_clean, x_clean).fit()
                intercept = result.params[0]
                slope = result.params[1]
                pval = result.pvalues[1]

                if pval > 0.05:
                    pval = np.nan
                    slope = np.nan
                    intercept = np.nan
                p_value = pval
            else:
                slope = np.nan
                p_value = np.nan
                intercept = np.nan

            variables_info[var_na][reg_na][decades_na[dec_na]]['slope_aver_reg'] = slope
            variables_info[var_na][reg_na][decades_na[dec_na]]['pval_aver_reg'] = p_value
            variables_info[var_na][reg_na][decades_na[dec_na]]['intercept_aver_reg'] = intercept

        # with open("TrendsDictWholeArctic.txt", "wb") as myFile:
        #     pickle.dump(variables_info, myFile)

