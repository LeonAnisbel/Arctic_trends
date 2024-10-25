import xarray as xr
import numpy as np

def get_month(da,m):
    da_yr = da
    da_t = da_yr.where(da_yr.time.dt.month == m, drop=True)
    return da_t

def tri_month_mean(v_month, months):
    da_tri_mean = []
    if len(months) > 1:
        for yr in range(len(v_month[0]['time'])):
            da_t_reg_m_list_yr = xr.concat([v_month[0].isel(time=yr),
                                            v_month[1].isel(time=yr),
                                            v_month[2].isel(time=yr)], dim='t')
            da_tri_mean.append(da_t_reg_m_list_yr.mean(dim='t', skipna=True))
        da_tri_mean_yrs = xr.concat(da_tri_mean, dim='time')

    else:
        da_tri_mean_yrs = v_month[0]

    #print('trimestre', da_tri_mean_yrs)
    return da_tri_mean_yrs

def season_aver(data, months):
    v_month = []
    for m in months:
        v_ti = get_month(data, m)
        v_ti['time'] = v_ti['time'].dt.year
        v_month.append(v_ti)
    v_tri_mo = tri_month_mean(v_month, months)
    return v_tri_mo

def pick_month_var_reg(data, months, aer_conc=False):
    # data_month = data.where((data.time.dt.month == 9), drop=True)

    if aer_conc:
        data_month = data
    else:
        print('entered here')
        data_month = season_aver(data, months)

    print(data_month.shape)
    data_month_reg = data_month.where(data_month.lat > 60, drop=True)
    lat = data_month_reg.lat
    lon = data_month_reg.lon

    return data_month_reg, lat, lon


def alloc_metadata(names, variables_info, trends=False):
    var_trend, var_pval, var_lim, var_unit = [], [], [], []
    for id in names:
        var_trend.append(variables_info[id]['slope'])
        var_pval.append(variables_info[id]['pval'])
        var_lim.append(variables_info[id]['lim'])
        if trends:
            var_unit.append(variables_info[id]['unit'] +  ' $yr^{-1}$')
        else:
            var_unit.append(variables_info[id]['unit'])


    return var_trend, var_pval, var_lim, var_unit


def find_yr_min_ice(v_1month):
    list_mins, years = [], []
    for i in v_1month.time:
        list_mins.append(v_1month.where((v_1month.time == i), drop=True).mean(skipna=True))
        years.append(i)
    min_val = list_mins.index(min(list_mins))
    return years[min_val]

def find_max_lim(panel_var):
    vlims = []
    for vl in panel_var:
        vlims.append(vl.max())
    return vlims

def get_seaice_vals(variables_info, var_na):
    v_season = variables_info[var_na]['data_season_reg'].compute()
    seaice_min = get_min_seaice(variables_info, var_na)
    seaice_min_10 = seaice_min.where(seaice_min > 10, drop=True)
    
    seaice_mean = v_season.mean('time')
    return [seaice_min_10, seaice_mean, seaice_min]

def get_min_seaice(variables_info, var_na):
    v_1month = variables_info[f'{var_na}_1m']['data_season_reg'].compute()
    year_min = find_yr_min_ice(v_1month)
    seaice_min = v_1month.where((v_1month.time == year_min), drop=True).isel(time=0)
    return seaice_min

def get_perc_increase(variables_info, panel_names):
    percent_increase_yr, panel_unit = [], []
    unit_percent_increase_yr = '% '
    nan_matrix = np.empty((variables_info[panel_names[0]]['intercept'].shape))
    nan_matrix[:] = np.nan
    for id in panel_names:
        linear_increase = variables_info[id]['intercept'] - (
                variables_info[id]['intercept'] + (variables_info[id]['slope'] * 30))
        percent_increase = (np.divide(linear_increase, variables_info[id]['intercept'],
                                      out=nan_matrix,
                                      where=(variables_info[id]['intercept'] >= 0)))
        # #variables_info[id]['intercept'] > 0))

        print('max % of increase', percent_increase.max(), 'and min % of increase', percent_increase.min())
        # percent_increase = (linear_increase / variables_info[id]['intercept']) - 1
        percent_increase_yr.append(percent_increase * 100 / 30)
        panel_unit.append(unit_percent_increase_yr)
    return percent_increase_yr, panel_unit




def get_conds(lat,lon):
    conditions = [[[lat, 63, 90]],
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
