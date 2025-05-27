import xarray as xr
import numpy as np
import os
import global_vars
import utils


def read_files_data(path_dir):
    data = xr.open_mfdataset(path_dir,
                             concat_dim='time',
                             combine='nested')
    return data


def read_ocean_data():
    ocean_dir = global_vars.ocean_dir_path

    C_ice_msk = read_files_data(ocean_dir + "mask_a_ice*")['VAR']

    C_ice = read_files_data(ocean_dir + "a_ice*")['VAR']
    C_temp = read_files_data(ocean_dir + "temperature/sst*.nc")['sst'] - 273.16  # * C_ice_msk-273.16

    C_tep = read_files_data(ocean_dir + "TEP_files/TEP_regular_grid*.nc")['VAR']  # * C_ice_msk

    C_nppd = read_files_data(ocean_dir + 'NPPd*')['VAR']  # * C_ice_msk
    C_nppn = read_files_data(ocean_dir + 'NPPn*')['VAR']  # * C_ice_msk
    C_din = read_files_data(ocean_dir + 'DIN*')['VAR']  # * C_ice_msk

    C_pcho = read_files_data(ocean_dir + 'PCHO_var*')['PCHO']  # * C_ice_msk
    C_dcaa = read_files_data(ocean_dir + 'DAA_var*')['DAA']  # * C_ice_msk
    C_pl = read_files_data(ocean_dir + "Lipids_var*")['LIPIDS']  # * C_ice_msk
    C_conc_tot = C_pcho + C_dcaa + C_pl

    return C_pcho, C_dcaa, C_pl, C_ice, C_temp, C_nppd + C_nppn, C_din


def read_omf_data():
    omf_dir = global_vars.omf_dir_path
    omf = read_files_data(omf_dir)  # omf in %
    return omf


def read_model_spec_data(file):
    return xr.open_mfdataset(file,
                             concat_dim='time',
                             combine='nested',
                             preprocess=lambda ds:
                             ds[['PRO_AS', 'POL_AS', 'LIP_AS', 'SS_AS']])


def read_aerosol_data(months):
    aer_dir = global_vars.aer_dir_path

    v_month_lip = []
    v_month_pol = []
    v_month_pro = []
    v_month_ss = []

    for mo in months:
        if mo < 10:
            mo_str = f'0{mo}'
        else:
            mo_str = f'{mo}'

        v_yr = []
        for yr in np.arange(1990, 2020):
            # print(mo_str)
            files = f'{aer_dir}_{yr}{mo_str}.01_tracer.nc'
            # file_ro = f'{data_dir}{exp}_{yr}{mo_str}.01_vphysc.nc'
            # print(files)
            v_yr.append(read_files_data(files).isel(lev=46).mean(dim='time'))

            # data_ro = read_model_spec_data(file_ro)

            # da_ro, da_ds = [], []
            # for ti in ti_sel:
            #     # da_ro.append(data_ro['rhoam1'].isel(time=ti).isel(lev=46))
            #     da_ds.append(data.isel(time=ti).isel(lev=46))
            # da_m_ro = xr.concat(da_ro, dim='time')
            # da_m_ds = xr.concat(da_ds, dim='time')

        v_yr_m = xr.concat(v_yr, dim='time') * 1e12
        v_month_lip.append(v_yr_m['LIP_AS'].compute())
        v_month_pol.append(v_yr_m['POL_AS'].compute())
        v_month_pro.append(v_yr_m['PRO_AS'].compute())
        v_month_ss.append(v_yr_m['SS_AS'].compute())

    C_lip = utils.tri_month_mean(v_month_lip, months)
    C_pol = utils.tri_month_mean(v_month_pol, months)
    C_pro = utils.tri_month_mean(v_month_pro, months)
    C_ss = utils.tri_month_mean(v_month_ss, months)

    return C_pol, C_pro, C_lip, C_ss


def concat_months_selvar(v_yr, unit_factor, var_id, v_month):
    v_yr_m = xr.concat(v_yr, dim='time') * unit_factor
    v_month.append(v_yr_m[f'{var_id}'].compute())
    return v_month


def read_each_aerosol_data(months, var_id, file_type, unit_factor, per_month=False, two_dim=False):
    aer_dir = global_vars.aer_dir_path
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
            if os.path.isfile(files):
                if two_dim:
                    # calculate values per season without conversion from year to month
                    ds = read_files_data(files)
                    v_yr.append(ds.mean(dim='time', skipna=True))
                    if file_type == 'emi':
                        # calculate values as a conversion from yr to month
                        ds_gboxarea = ds.rename({'gboxarea': f'area_{var_id}'})
                        ds_emi = ds.rename({var_id: f'area_{var_id}'})
                        ds_emi_gboxarea = ds_gboxarea * ds_emi * 86400
                        var_id_new = f'area_{var_id}'
                        v_m_yr.append(ds_emi_gboxarea.sum(dim='time', skipna=True))
                else:
                    files_dens = f'{aer_dir}_{yr}{mo_str}.01_vphysc.nc'
                    print(files_dens)
                    ds_dens = read_files_data(files_dens).isel(lev=46).rename({'rhoam1': f'{var_id}'})
                    ds_conc = read_files_data(files).isel(lev=46)
                    ds_conc_ng_m3 = ds_dens*ds_conc
                    v_yr.append(ds_conc_ng_m3.mean(dim='time', skipna=True))
            else:
                pass
        v_month = concat_months_selvar(v_yr, unit_factor, var_id, v_month)
        if two_dim and file_type == 'emi':
            v_m_month = concat_months_selvar(v_m_yr, 1e-9, var_id_new, v_m_month)

    if two_dim and file_type == 'emi':
        C_m = utils.tri_month_mean(v_m_month, months, two_dim=two_dim, file_type=file_type)
        # C_m_ds = xr.concat(v_m_month, dim='time')
        # C_m = utils.season_aver(C_m_ds, months)

    C = utils.tri_month_mean(v_month, months)
    # C_ds = xr.concat(v_month, dim='time')
    # C = utils.season_aver(C_ds, months)

    return C, C_m
