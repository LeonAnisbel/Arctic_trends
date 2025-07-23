import xarray as xr
import numpy as np

import global_vars


def create_var_info_dict():
    npp_din_u = 'mmol C m${^{-2}}$ d${^{-1}}$ '
    flux_u = 'ng m${^{-2}}$ s${^{-1}}$'
    flux_mo_u = 'Tg month$^{-1}$'
    burden_u = 'mg m$^{-2}$'
    sic_u = '% '
    conc_u = 'ng m${^{-3}}$ '
    conc_biom_u = 'mmol C m${^{-3}}$ '
    vars_names = [['AER_burden_POL', burden_u], ['AER_burden_PRO', burden_u], ['AER_burden_LIP', burden_u],
                  ['AER_burden_tot', burden_u], ['AER_burden_SS', burden_u], ['AER_burden_SSA', burden_u],
                  ['AER_F_POL', flux_u], ['AER_F_PRO', flux_u], ['AER_F_LIP', flux_u],
                  ['AER_F_tot', flux_u], ['AER_F_SS', flux_u], ['AER_F_SSA', flux_u],
                  ['AER_F_POL_anom', flux_u], ['AER_F_PRO_anom', flux_u], ['AER_F_LIP_anom', flux_u],
                  ['AER_F_tot_anom', flux_u], ['AER_F_SS_anom', flux_u], ['AER_F_SSA_anom', flux_u],
                  ['AER_F_POL_m', flux_mo_u], ['AER_F_PRO_m', flux_mo_u], ['AER_F_LIP_m', flux_mo_u],
                  ['AER_F_tot_m', flux_mo_u], ['AER_F_SS_m', flux_mo_u], ['AER_F_SSA_m', flux_mo_u],
                  ['AER_U10', 'm s${^{-1}}$'], ['AER_SST', '$^{o}C$ '],
                  ['AER_SIC', sic_u], ['AER_SIC_area_px', sic_u], ['AER_SIC_1m', sic_u],
                  ['AER_POL', conc_u], ['AER_PRO', conc_u], ['AER_LIP', conc_u],
                  ['AER_tot', conc_u], ['AER_SS', conc_u], ['AER_SSA', conc_u],
                  ['AER_POL_anom', conc_u], ['AER_PRO_anom', conc_u], ['AER_LIP_anom', conc_u],
                  ['AER_tot_anom', conc_u], ['AER_SS_anom', conc_u], ['AER_SSA_anom', conc_u],
                  ['OMF_POL', sic_u], ['OMF_PRO', sic_u], ['OMF_LIP', sic_u], ['OMF_tot', sic_u],
                  ['PCHO', conc_biom_u], ['DCAA', conc_biom_u], ['PL', conc_biom_u], ['Biom_tot', conc_biom_u],
                  ['Sea_ice', sic_u], ['Sea_ice_1m', sic_u], ['Sea_ice_area_px', sic_u], ['Sea_ice_area_px_1m', sic_u],
                  ['SST', '$^{o}C$ '], ['NPP', npp_din_u], ['NPP_anom', npp_din_u], ['DIN', npp_din_u],
    ]
    variables_info = {}
    for li in vars_names:
        variables_info[li[0]] = {'unit': li[1]}
    return variables_info

def get_month(da,m):
    da_yr = da
    da_t = da_yr.where(da_yr.time.dt.month == m, drop=True)
    return da_t

def tri_month_mean(v_month, months, two_dim=False, file_type=None):
    da_tri_mean = []
    if len(months) > 1:
        for yr in range(len(v_month[0]['time'])):
            da_t_reg_m_list_yr = xr.concat([v_month[0].isel(time=yr),
                                            v_month[1].isel(time=yr),
                                            v_month[2].isel(time=yr)], dim='t')
            if two_dim and file_type == 'emi':
                da_tri_mean.append(da_t_reg_m_list_yr.sum(dim='t', skipna=True)) # Tg/season
            else:
                da_tri_mean.append(da_t_reg_m_list_yr.mean(dim='t', skipna=True))
        da_tri_mean_yrs = xr.concat(da_tri_mean, dim='time')

    else:
        da_tri_mean_yrs = v_month[0]

    #da_tri_mean_yrs = v_month.groupby((v_month.time.dt.year, v_month.time.dt.month)).mean('time', skipna=True)

    return da_tri_mean_yrs

def season_aver(data, months):
    # da_tri_mean_yrs = data.groupby((data.time.dt.year)).mean('time', skipna=True)
    # print(da_tri_mean_yrs.time)
    v_month = []
    for m in months:
        v_ti = get_month(data, m)
        v_ti['time'] = v_ti['time'].dt.year
        v_month.append(v_ti)
#    v_month_da = xr.concat(v_month, dim='time')
    v_tri_mo = tri_month_mean(v_month, months)
    return v_tri_mo

def pick_month_var_reg(data, months, aer_conc=False):
    if aer_conc:
        data_month = data
    else:
        data_month = season_aver(data, months)

    data_month_reg = data_month.where(data_month.lat > 60, drop=True)
    lat = data_month_reg.lat
    lon = data_month_reg.lon

    return data_month_reg, lat, lon


def alloc_metadata(names, variables_info, trends=False, percent_increase=False, mk_method = True):
    """ :returns datset of variables in names, their units and dataset of p-value """
    var_trend, var_pval, var_lim, var_unit = [], [], [], []
    for id in names:
        sl = variables_info[id]['slope']
        if percent_increase:
            sl = sl * 100
        var_trend.append(sl)
        if mk_method:
            p = 'significance'
        else:
            p = 'pval'
        var_pval.append(variables_info[id][p])
        if trends:
            var_unit.append(variables_info[id]['unit'] +  ' yr$^{-1}$')
        else:
            var_unit.append(variables_info[id]['unit'])
    return var_trend, var_pval, var_unit


def find_yr_min_ice(v_1month):
    """ Finds the year of minimum sea ice
    :return year"""
    list_mins, years = [], []
    for i in v_1month.time:
        list_mins.append(v_1month.where((v_1month.time == i), drop=True).values)
        years.append(i)
    min_val = list_mins.index(np.nanmin(list_mins))
    return years[min_val]

def find_max_lim(panel_var):
    vlims = []
    for vl in panel_var:
        vlims.append(vl.max())
    return vlims

def get_seaice_vals(variables_info, var_na, get_min_area=False):
    """ This function contains the function calls to calculate sea ice minimum
    :return datasets of minimum SIC for SIC>10, average seasonal SIC and minimum SIC """
    v_season = variables_info[var_na]['data_season_reg'].compute()
    if get_min_area:
        seaice_min = get_min_seaice(variables_info, var_na)
        seaice_min_10 = seaice_min.where(seaice_min > 10, drop=True)
    else:
        seaice_min = None
        seaice_min_10 = None
    seaice_mean = v_season.mean('time')
    return [seaice_min_10, seaice_mean, seaice_min]

def get_min_seaice(variables_info, var_na):
    """ Calculates the minimum sea ice over the 30 years for the selected season
    :return dataset of minimum of SIC"""
    v_1month_area = variables_info[f'{var_na}_area_px_1m']['data_season_reg'].compute()
    v_1month_area_tot = v_1month_area.sum(dim=['lat', 'lon'],
                                 skipna=True) * 1e-6
    year_min = find_yr_min_ice(v_1month_area_tot)
    print(year_min)
    v_1month_conc = variables_info[f'{var_na}_1m']['data_season_reg'].compute()
    seaice_min = v_1month_conc.where((v_1month_conc.time == year_min), drop=True).isel(time=0)
    return seaice_min

def get_perc_increase(variables_info, panel_names):
    percent_increase_yr, panel_unit = [], []
    unit_percent_increase_yr = '% '
    nan_matrix = np.empty((variables_info[panel_names[0]]['data_time_mean'].shape))
    nan_matrix[:] = np.nan
    for id in panel_names:
        percent_increase = (np.divide(variables_info[id]['slope'],
                                    variables_info[id]['data_time_mean'])
                                    )
        percent_increase_yr.append(percent_increase*100)
        print(percent_increase.min().values)
        panel_unit.append(unit_percent_increase_yr)
    return percent_increase_yr, panel_unit


def get_weighted_mean(gboxarea_reg, data, aer_conc=False):
    if aer_conc:
        weights = gboxarea_reg / gboxarea_reg.sum(dim=('lat', 'lon'))
    else:
        weights = np.cos(np.deg2rad(data.lat))
        weights /= weights.sum()

    ds_weighted = data.weighted(weights)
    ds_weighted_mean = ds_weighted.mean(dim=['lat', 'lon'], skipna=True)
    return ds_weighted_mean


def get_conds(lat,lon):
    """ Define the subregions limits
    Returns list with region limits, latitude and longitude arrays from the model grid"""
    conditions = [[[lat, global_vars.lat_arctic_lim, 90]],
                  [[lat, 66, 82], [lon, 20, 60]],
                  [[lat, 66, 82], [lon, 60, 100]],
                  [[lat, 66, 82], [lon, 100, 140]],
                  [[lat, 66, 82], [lon, 140, 180]],
                  [[lat, 66, 82], [lon, -180, -160]],
                  [[lat, 66, 82], [lon, -160, -120]],
                  [[lat, 66, 82], [lon, -120, -70]],
                  [[lat, 66, 82], [lon, -70, -50]],
                  [[lat, 66, 82], [lon, -30, 20]],
                  [[lat, 82, 90]], ]

    return conditions
def regions():
    """ This function defines a dictionary with Arctic subregions
    Returns dictionary with regions"""
    reg_data = {'Arctic': {},
                'Barents Sea': {},
                'Kara Sea': {},
                'Laptev Sea': {},
                'East-Siberian Sea': {},
                'Chukchi Sea': {},
                'Beaufort Sea': {},
                'Canadian Archipelago': {},
                'Baffin Bay': {},
                'Greenland & Norwegian Sea': {},
                'Central Arctic': {},
                }
    return reg_data


def get_var_reg(v, cond):
    """ This function is used to select and filter certain regions according to the conditions variable (cond).
    v is the variable dataset, cond is the dictionary containing the regions
    Returns the variable dataset with the data filtered to meet each condition"""
    if len(cond) <= 1:
        v = v.where((cond[0][0] > cond[0][1]) &
                    (cond[0][0] < cond[0][2])
                    , drop=True)
    elif len(cond) > 1:
        v = v.where(((cond[0][0] > cond[0][1]) &
                     (cond[0][0] < cond[0][2]) &
                     (cond[1][0] > cond[1][1]) &
                     (cond[1][0] < cond[1][2]))
                    , drop=True)
    return v


def create_ds(data_month_reg, lon):
    data_ds = xr.Dataset(
        data_vars=dict(
            data_region=(["time", "lat", "lon"], data_month_reg.data),
        ),
        coords=dict(
            time=("time", data_month_reg.time.data),
            lon=("lon", lon),
            lat=("lat", data_month_reg.lat.data),
        ),
    )
    return data_ds




def create_ds2(data_array, data_month_reg):
    data_ds = xr.Dataset(
        data_vars=dict(
            data_region=(["time", "lat", "lon"],data_array.data),
        ),
        coords=dict(
            time=("time", data_month_reg.time.data),
            lon=("lon", data_month_reg.lon.data),
            lat=("lat", data_month_reg.lat.data),
        ),
    )
    return data_ds


def compute_seaice_area_px(C_ice):
    bx_size = abs(C_ice.lat.values[1] - C_ice.lat.values[0])
    grid_bx_area = (bx_size * 110.574) * (
            bx_size * 111.320 * np.cos(np.deg2rad(C_ice.lat)))  # from % sea ice of grid to km2
    C_ice_area_px = C_ice * grid_bx_area
    return C_ice_area_px

def calculate_anomaly(conc):
    conc_climatology = conc.mean(dim='time',
                                 skipna=True)
    conc_anomaly = conc - conc_climatology
    return conc_anomaly