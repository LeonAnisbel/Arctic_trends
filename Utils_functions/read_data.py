import xarray as xr
import numpy as np
import os
from Utils_functions import utils, global_vars


def read_files_data(path_dir):
    """
     This function reads (with dask) the data from a certain file type
     :return: dataset
     """
    data = xr.open_mfdataset(path_dir,
                             concat_dim='time',
                             combine='nested')
    return data


def read_ocean_data():
    """
     This function reads the data of FESOM-REcoM interpolated regular grid
     """
    ocean_dir = global_vars.ocean_dir_path

    C_ice = read_files_data(ocean_dir + "a_ice*")['VAR']
    C_temp = read_files_data(ocean_dir + "temperature/sst*.nc")['sst'] - 273.16

    C_tep = read_files_data(ocean_dir + "TEP_files/TEP_regular_grid*.nc")['VAR']

    C_nppd = read_files_data(ocean_dir + 'NPPd*')['VAR']
    C_nppn = read_files_data(ocean_dir + 'NPPn*')['VAR']
    C_din = read_files_data(ocean_dir + 'DIN*')['VAR']

    C_pcho = read_files_data(ocean_dir + 'PCHO_var*')['PCHO']
    C_dcaa = read_files_data(ocean_dir + 'DAA_var*')['DAA']
    C_pl = read_files_data(ocean_dir + "Lipids_var*")['LIPIDS']

    return C_pcho, C_dcaa, C_pl, C_ice, C_temp, C_nppd + C_nppn, C_din


def read_omf_data():
    """
     This function reads the OMF data computed based on biomolecules with the FESOM-REcoM interpolated regular grid
     """
    omf_dir = global_vars.omf_dir_path
    omf = read_files_data(omf_dir)
    return omf


def read_model_spec_data(file):
    """
     This function reads (with dask) specific data
     :return: dataset
     """
    return xr.open_mfdataset(file,
                             concat_dim='time',
                             combine='nested',
                             preprocess=lambda ds:
                             ds[['PRO_AS', 'POL_AS', 'LIP_AS', 'SS_AS']])


def concat_months_selvar(v_yr, unit_factor, var_id, v_month):
    """
     This function concatenates dataarrays and creates new dimension time
     :return list of an specific variable var_id
     """
    v_yr_m = xr.concat(v_yr, dim='time') * unit_factor
    v_month.append(v_yr_m[f'{var_id}'].compute())
    return v_month


def read_each_aerosol_data(months, var_id, file_type, unit_factor, per_month=False, two_dim=False):
    """
     This function reads data of var_id type and returns either the 3-month sum or average value over 30 yr
     :return: dataset
     """
    aer_dir = global_vars.aer_dir_path
    if file_type == 'B24bend_poly_only_inp_marine_concentration':
        aer_dir = global_vars.inp_dir_path
    C_m = []
    v_month = []
    v_m_month = []

    for mo in months:
        if mo < 10:
            mo_str = f'0{mo}'
        else:
            mo_str = f'{mo}'

        v_yr = []
        v_m_yr = []

        for yr in np.arange(1990, 2020):
            files = f'{aer_dir}_{yr}{mo_str}.01_{file_type}.nc'
            if file_type == 'B24bend_poly_only_inp_marine_concentration':
                files = f'{aer_dir}_{yr}{mo_str}_{file_type}.nc'
            if os.path.isfile(files):
                if two_dim:
                    # calculate values per season without conversion from year to month
                    ds = read_files_data(files)
                    v_yr.append(ds.mean(dim='time',
                                        skipna=True))
                    if file_type == 'emi':
                        # calculate values as a conversion from yr to month
                        ds_gboxarea = ds.rename({'gboxarea': f'area_{var_id}'})
                        ds_emi = ds.rename({var_id: f'area_{var_id}'})
                        ds_emi_gboxarea = ds_gboxarea * ds_emi * 86400  # 86400 seconds in a day
                        var_id_new = f'area_{var_id}'
                        v_m_yr.append(ds_emi_gboxarea.sum(dim='time',
                                                          skipna=True))
                else:
                    files_dens = f'{aer_dir}_{yr}{mo_str}.01_vphysc.nc'
                    ds_dens = read_files_data(files_dens).isel(lev=46).rename({'rhoam1': f'{var_id}'})
                    ds_conc = read_files_data(files).isel(lev=46)
                    ds_conc_ng_m3 = ds_dens*ds_conc
                    v_yr.append(ds_conc_ng_m3.mean(dim='time',
                                                   skipna=True))
            else:
                pass
        v_month = concat_months_selvar(v_yr,
                                       unit_factor,
                                       var_id,
                                       v_month)
        if two_dim and file_type == 'emi':
            v_m_month = concat_months_selvar(v_m_yr,
                                             1e-9,
                                             var_id_new,
                                             v_m_month)

    if two_dim and file_type == 'emi':
        C_m = utils.tri_month_mean_sum(v_m_month,
                                       months,
                                       two_dim=two_dim,
                                       file_type=file_type)
    C = utils.tri_month_mean_sum(v_month,
                                 months)

    return C, C_m


def read_daily_data(months, var_id, file_type, unit_factor, emi_gridbox=False, two_dim=False):
    """
     This function reads data as daily values and returns dataarray with daily values
     :return: dataarray
     """
    aer_dir = global_vars.aer_dir_path
    v_m_month = []
    v_m_yr = []

    for mo in months:
        if mo < 10:
            mo_str = f'0{mo}'
        else:
            mo_str = f'{mo}'

        for yr in np.arange(1990, 2020):
            files = f'{aer_dir}_{yr}{mo_str}.01_{file_type}.nc'
            if two_dim:
                ds = read_files_data(files)
                ds_var = ds
                var_id_new = f'{var_id}'
                if emi_gridbox:
                    # calculate values considering the grid cell size
                    ds_gboxarea = ds.rename({'gboxarea': f'area_{var_id}'})
                    ds_emi = ds.rename({var_id: f'area_{var_id}'})
                    ds_emi_gboxarea = ds_gboxarea * ds_emi * 86400  # kg/day, 86400 seconds in a day (model values are given as daily quantities)
                    var_id_new = f'area_{var_id}'
                    ds_var = ds_emi_gboxarea
                ds_daily = ds_var.groupby('time.day').mean(skipna=True)
                v_m_yr.append(ds_daily[var_id_new])

    v_m_month = xr.concat(v_m_yr, dim='day') * unit_factor
    lon_360 = ((v_m_month.lon.values + 180) % 360) - 180

    data_ds = xr.Dataset(
        data_vars=dict(
            data_region=(["time", "lat", "lon"], v_m_month.values),
        ),
        coords=dict(
            time=("time", v_m_month.day.values),
            lon=("lon", lon_360),
            lat=("lat", v_m_month.lat.values),
        ),
    )

    return data_ds['data_region']
