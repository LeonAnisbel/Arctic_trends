import xarray as xr
import numpy as np

import global_vars
import utils


def read_files_data(path_dir):
    data = xr.open_mfdataset(path_dir,
                             concat_dim='time',
                             combine='nested')
    return data


def read_ocean_data():
    ocean_dir = "../regular_grid_interp/"

    C_ice_msk = read_files_data(ocean_dir + "mask_a_ice*")['VAR']

    C_ice = read_files_data(ocean_dir + "a_ice*")['VAR']
    C_temp = read_files_data(ocean_dir + "temperature/sst*.nc")['sst'] -273.16#* C_ice_msk-273.16

    C_tep = read_files_data(ocean_dir + "TEP_files/TEP_regular_grid*.nc")['VAR']# * C_ice_msk

    C_nppd = read_files_data(ocean_dir + 'NPPd*')['VAR'] #* C_ice_msk
    C_nppn = read_files_data(ocean_dir + 'NPPn*')['VAR'] #* C_ice_msk
    C_din = read_files_data(ocean_dir + 'DIN*')['VAR'] #* C_ice_msk

    C_pcho = read_files_data(ocean_dir + 'PCHO_var*')['PCHO'] #* C_ice_msk
    C_dcaa = read_files_data(ocean_dir + 'DAA_var*')['DAA'] #* C_ice_msk
    C_pl = read_files_data(ocean_dir + "Lipids_var*")['LIPIDS'] #* C_ice_msk
    C_conc_tot = C_pcho + C_dcaa + C_pl

    return C_pcho, C_dcaa, C_pl, C_ice, C_temp, C_nppd+C_nppn, C_din


def read_omf_data():
    omf_dir = global_vars.omf_dir_path
    omf = read_files_data(omf_dir) # omf in %
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
            #print(mo_str)
            files = f'{aer_dir}_{yr}{mo_str}.01_tracer.nc'
            # file_ro = f'{data_dir}{exp}_{yr}{mo_str}.01_vphysc.nc'
            #print(files)
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



def read_each_aerosol_data(months, var_id, file_type, unit_factor, two_dim=False):
    aer_dir = global_vars.aer_dir_path

    v_month = []
    for mo in months:
        if mo < 10:
            mo_str = f'0{mo}'
        else:
            mo_str = f'{mo}'

        v_yr = []
        for yr in np.arange(1990, 2020):
            files = f'{aer_dir}_{yr}{mo_str}.01_{file_type}.nc'
            #print(files)

            if two_dim:
                v_yr.append(read_files_data(files).mean(dim='time'))
            else:
                v_yr.append(read_files_data(files).isel(lev=0).mean(dim='time'))

        v_yr_m = xr.concat(v_yr, dim='time') * unit_factor
        v_month.append(v_yr_m[f'{var_id}'].compute())

    C = utils.tri_month_mean(v_month, months)

    return C
