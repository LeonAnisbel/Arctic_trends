import numpy as np
import statsmodels.api as sm
import pymannkendall as mk
import read_data, utils
import xarray as xr
import pickle
import warnings


ftype = np.float64
def trend_aver_per_reg(variables_info, var_na, data_month_reg, data_month_ice_reg, var_type, gboxarea, per_unit_sic=False, aer_conc=False):
    lon_360 = data_month_reg.lon.data
    if var_type == 'AER':
        lon = ((lon_360 + 180) % 360) - 180
    else:
        lon = lon_360

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
            if var_type == 'AER': # exclude variables for accumulated emission flux

                reg_sel_vals = reg_sel_vals_whole.where((reg_sel_vals_whole.time >= decades_idx[dec_na][0])&
                                                  (reg_sel_vals_whole.time <= decades_idx[dec_na][1]),
                                                  drop=True)

                gboxarea_ds = utils.create_ds(gboxarea, lon)
                conditions = utils.get_conds(gboxarea_ds.lat, gboxarea_ds.lon)
                reg_sel_gboxarea = utils.get_var_reg(gboxarea_ds, conditions[idx])
                reg_sel_vals_gbx = reg_sel_gboxarea.where((reg_sel_gboxarea.time >= decades_idx[dec_na][0])&
                                                  (reg_sel_gboxarea.time <= decades_idx[dec_na][1]),
                                                  drop=True)
            else:
                reg_sel_vals = reg_sel_vals_whole.where((reg_sel_vals_whole.time >= dec[0])&
                                                  (reg_sel_vals_whole.time <= dec[1]),
                                                  drop=True)

            if var_na == 'Sea_ice_area_px' or var_na == 'AER_SIC_area_px':
                ff = 1e-6 # from km2 to millions of km2
            elif var_na[-2:] == '_m' or var_na[:10] == 'AER_burden':
                ff = 1
            print('REGION', reg_na)

            if (var_na == 'Sea_ice_area_px' or var_na=='AER_SIC_area_px'
                    or var_na[-2:] == '_m' or var_na[:10] == 'AER_burden'): # exclude variables for accumulated emission flux:
                if var_na[:10] == 'AER_burden':
                    # convert from mg/m2 to mg
                    reg_sel_vals['data_region'] = reg_sel_vals['data_region'] * reg_sel_vals_gbx['data_region']
                data_month = reg_sel_vals['data_region'].sum(dim=['lat', 'lon'],
                                                             skipna=True) * ff
                data_type_mean_or_sum = 'data_sum_reg'
                variables_info[var_na][reg_na][decades_na[dec_na]][data_type_mean_or_sum] = data_month


                data_latlon_mean = data_month
            # if var_na != 'Sea_ice_area_px' and var_na !='AER_SIC_area_px':
            else:
                data_type_mean_or_sum = 'data_aver_reg'
                data_month = reg_sel_vals['data_region']
                if aer_conc:
                    data_latlon_mean = utils.get_weighted_mean(reg_sel_vals_gbx['data_region'], data_month, aer_conc=aer_conc)
                else:
                    data_latlon_mean = utils.get_weighted_mean(None, data_month, aer_conc=aer_conc)

                variables_info[var_na][reg_na][decades_na[dec_na]][data_type_mean_or_sum] = data_latlon_mean

            data_time_mean = data_latlon_mean.mean(dim='time', skipna=True)
            da_compute = data_latlon_mean.compute()
            data_time_median = da_compute.median(dim='time', skipna=True)

            del da_compute
            variables_info[var_na][reg_na][decades_na[dec_na]]['data_time_mean'] = data_time_mean
            variables_info[var_na][reg_na][decades_na[dec_na]]['data_time_median'] = data_time_median


            if per_unit_sic:
                data_ds = utils.create_ds(data_month_ice_reg, lon)
                conditions = utils.get_conds(data_ds.lat, data_ds.lon)
                reg_sel_vals_ice = utils.get_var_reg(data_ds, conditions[idx])
                X_aux = np.ma.masked_where(reg_sel_vals_ice['data_region'] < 0.2,
                                           reg_sel_vals_ice['data_region'])
                X_aux = X_aux.filled(np.nan)

                data_month_ice = utils.create_ds2(X_aux, reg_sel_vals_ice['data_region'])


                data_month_ice_latlon_mean = utils.get_weighted_mean(reg_sel_vals_gbx, data_month_ice)
                X = data_month_ice_latlon_mean.data

                Y_aux = np.ma.masked_where(reg_sel_vals_ice['data_region'] < 0.2, data_month)
                Y_aux = Y_aux.filled(np.nan)
                data_month_var = utils.create_ds2(Y_aux, reg_sel_vals_ice['data_region'])
                Y = data_month_var.data

            else:
                X = data_month.time.values
                Y = variables_info[var_na][reg_na][decades_na[dec_na]][data_type_mean_or_sum].values

            X = X.astype(ftype)
            X = sm.add_constant(X)
            Y = Y.astype(ftype)

            y_arr = np.array(Y, dtype=float)
            x_arr = np.array(X, dtype=float)
            mask = ~np.isnan(y_arr)
            n = mask.sum()
            y_clean = y_arr[mask]
            x_clean = x_arr[mask]

            if n >= 10 and not np.allclose(y_clean, y_clean[0]):
                result = mk.original_test(y_clean)

                tau = result.Tau
                p_value = result.p

                h = result.trend
                if h == 'increasing':
                    hh = 1
                if h == 'decreasing':
                    hh = -1
                if h == 'no trend':
                    hh = 0
                trend = hh

                if result.h==False:
                    signif = np.nan
                else:
                    signif = 0.0001 # assign arbitrary small amount

                intercept = result.intercept
                slope = result.slope
            else:
                slope = np.nan
                p_value = np.nan
                intercept = np.nan
                trend = np.nan
                tau = np.nan
                signif = np.nan

            variables_info[var_na][reg_na][decades_na[dec_na]]['slope_aver_reg'] = slope
            variables_info[var_na][reg_na][decades_na[dec_na]]['pval_aver_reg'] = p_value
            variables_info[var_na][reg_na][decades_na[dec_na]]['intercept_aver_reg'] = intercept
            variables_info[var_na][reg_na][decades_na[dec_na]]['trend'] = trend
            variables_info[var_na][reg_na][decades_na[dec_na]]['tau'] = tau
            variables_info[var_na][reg_na][decades_na[dec_na]]['significance'] = signif

        # with open("TrendsDictWholeArctic.txt", "wb") as myFile:
        #     pickle.dump(variables_info, myFile)

