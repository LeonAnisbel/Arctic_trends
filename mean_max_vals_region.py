import numpy as np
from utils import get_var_reg, get_min_seaice, regions, get_conds
import pickle
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def reg_sel(lat, lon, data):
    reg_data = regions()
    for idx, reg_na in enumerate(list(reg_data.keys())):
        data_ds = xr.Dataset(
            data_vars=dict(
                slope=(["lat", "lon"], data.data),
            ),
            coords=dict(
                lon=("lon", lon.data),
                lat=("lat", lat.data),
            ),
        )
        conditions = get_conds(data_ds.lat, data_ds.lon)

        reg_sel_vals = get_var_reg(data_ds, conditions[idx])
        # print(reg_na, reg_sel_vals.slope.shape)

        # if reg_na == 'Central Arctic':
        #     import cartopy.crs as ccrs
        #     import matplotlib.pyplot as plt
        #     from matplotlib import ticker as mticker
        #     fig, ax = plt.subplots(nrows=1,
        #                            ncols=1,
        #                            sharex=True,
        #                            subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
        #     ax.set_extent([-180, 180, 60, 90],
        #                   ccrs.PlateCarree())
        #     cmap = plt.get_cmap('Blues', 15)
        #     cb = ax.pcolormesh(lon,
        #                        lat,
        #                        np.array(data),
        #                        cmap=cmap,
        #                        transform=ccrs.PlateCarree())
        #     ax.coastlines(color='darkgray')
        #     gl = ax.gridlines(draw_labels=True, )
        #     gl.ylocator = mticker.FixedLocator([65, 75, 85])
        #     plt.savefig(f'TEST.png')

        if reg_sel_vals.slope.shape[0] > 1:
            max_val = reg_sel_vals['slope'].max(skipna=True).values
            reg_data[reg_na]['max'] = max_val
            min_val = reg_sel_vals['slope'].min(skipna=True).values
            reg_data[reg_na]['min'] = min_val
            if abs(min_val) > abs(max_val):
                reg_data[reg_na]['max_absolute'] = float(min_val)
            else:
                reg_data[reg_na]['max_absolute'] = float(max_val)

            # print(reg_na, 'max = ', reg_data[reg_na]['max'], 'min =', reg_data[reg_na]['min'])
        else:
            reg_data[reg_na]['max_absolute'] = np.nan
            # print(reg_na, 'min = ', reg_sel_vals.slope.min())
    return reg_data


def create_df_plot_heatmap(col, col_name):
    df_vals = pd.DataFrame({'Regions': col[0],
                            col_name: col[1],
                            'Values': col[2],
                            })
    # fig = plt.figure(figsize=(6, 6))
    if col_name[:3] == 'OMF' or col_name[:3] == 'Oce':
        df_vals = df_vals[df_vals['Regions'] != 'Central Arctic']
    if col_name == ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ per unit SIC)':
        df_vals = df_vals[df_vals['Regions'] != 'Barents Sea']
    df_vals_piv = df_vals.pivot(index="Regions", columns=col_name, values="Values")

    return df_vals_piv


def plot_heatmap(df_vals_piv, col_name, fig_title):
    fig, ax = plt.subplots(1, 1,
                           figsize=(7, 5), )
    plot_each_heatmap(ax, df_vals_piv, col_name)
    plt.tight_layout()
    plt.savefig('Heatmap_' + fig_title + '.png')
    plt.close()


def plot_each_heatmap(ax, df_vals_piv, col_name):
    cmap = 'viridis'
    if col_name[:18] == 'Emission mass flux':
        axs = sns.heatmap(df_vals_piv, annot=True, cmap=cmap, norm=LogNorm(), ax=ax)
    else:
        axs = sns.heatmap(df_vals_piv, annot=True, cmap=cmap, ax=ax)

    axs.set(ylabel="", xlabel="")
    axs.xaxis.tick_top()
    axs.set_title(col_name, loc='right')

if __name__ == '__main__':

    with open("TrendsDict.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    with open("TrendsDict_per_ice.pkl", "rb") as myFile:
        variables_info_seaice = pickle.load(myFile)

    print('Aerosols from ECHAM')
    ## Calculate mean values per regions for emiss flux trends and emiss flux per unit of SIC
    panel_names = ['AER_F_POL', 'AER_F_PRO', 'AER_F_LIP', 'AER_F_SS']
    var_na_aer = ['PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$', 'SS$_{aer}$']

    lat = variables_info_yr[panel_names[0]]['lat']
    lon_360 = variables_info_yr[panel_names[0]]['lon']
    lon = ((lon_360 + 180) % 360) - 180

    reg_names = regions()

    columns1 = [[], [], []]
    columns2 = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in reg_names:
            columns1[0].append(reg_na)
            columns2[0].append(reg_na)

            slope = variables_info_yr[var_na][reg_na]['slope_aver_reg']
            columns1[1].append(var_na_aer[vidx])
            columns1[2].append(slope)

            slope = variables_info_seaice[var_na][reg_na]['slope_aver_reg']
            columns2[1].append(var_na_aer[vidx])
            columns2[2].append(slope)

    columns3 = [[], [], []]
    for reg_na in reg_names:
        columns3[0].append(reg_na)
        slope = variables_info_yr['AER_SIC'][reg_na]['slope_aver_reg']
        columns3[1].append('')#'SIC (% ${yr^{-1}}$)'
        columns3[2].append(slope)

    columns4 = [[], [], []]
    for reg_na in reg_names:
        columns4[0].append(reg_na)
        slope = variables_info_yr['AER_SST'][reg_na]['slope_aver_reg']
        columns4[1].append('')#'SST (C$^{o}$ ${yr^{-1}}$)'
        columns4[2].append(slope)

    columns5 = [[], [], []]
    for reg_na in reg_names:
        columns5[0].append(reg_na)
        slope = variables_info_yr['AER_U10'][reg_na]['slope_aver_reg']
        columns5[1].append('')#'Wind (m $s^{-1}$ ${yr^{-1}}$)'
        columns5[2].append(slope)

    col_name_emi = ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ ${yr^{-1}}$)'
    df_vals_piv_emi = create_df_plot_heatmap(columns1, col_name_emi)
    plot_heatmap(df_vals_piv_emi, col_name_emi, 'Aerosol_fluxes_')
    col_name_emi_sic = ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ per unit SIC)'
    df_vals_piv_emi_sic = create_df_plot_heatmap(columns2, col_name_emi_sic)
    plot_heatmap(df_vals_piv_emi_sic, col_name_emi_sic, 'Aerosol_fluxes_per unit_SIC_')
    col_name_sic = ''
    df_vals_piv_sic = create_df_plot_heatmap(columns3, col_name_sic)
    plot_heatmap(df_vals_piv_sic, col_name_sic, 'SIC_')
    #
    #
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax.flatten()
    # plot_each_heatmap(ax[0], df_vals_piv_sic, col_name_sic)
    # ax[0].set_title(r'$\bf{(a)}$', loc='left')
    plot_each_heatmap(ax[0], df_vals_piv_emi, col_name_emi)
    ax[0].set_title(r'$\bf{(a)}$', loc='left')
    plot_each_heatmap(ax[1], df_vals_piv_emi_sic, col_name_emi_sic)
    ax[1].set_title(r'$\bf{(b)}$', loc='left')
    plt.tight_layout()
    plt.savefig('Heatmap_EmiFlux_SIC.png')
    plt.close()


    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax.flatten()
    col_name_sic = 'SIC (% ${yr^{-1}}$)'
    df_vals_piv_sic = create_df_plot_heatmap(columns3, col_name_sic)
    plot_each_heatmap(ax[0], df_vals_piv_sic, col_name_sic)
    ax[0].set_title(r'$\bf{(a)}$', loc='left')

    col_name_sst = 'SST (C$^{o}$ ${yr^{-1}}$)'
    df_vals_piv_sst = create_df_plot_heatmap(columns4, col_name_sst)
    plot_each_heatmap(ax[1], df_vals_piv_sst, col_name_sst)
    ax[1].set_title(r'$\bf{(b)}$', loc='left')

    col_name_wind = 'Wind (m $s^{-1}$ ${yr^{-1}}$)'
    df_vals_piv_wind = create_df_plot_heatmap(columns5, col_name_wind)
    plot_each_heatmap(ax[2], df_vals_piv_wind, col_name_wind)
    ax[2].set_title(r'$\bf{(c)}$', loc='left')
    plt.tight_layout()
    plt.savefig('Heatmap_SIC_SST_Wind.png')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_each_heatmap(ax, df_vals_piv_sic, col_name_sic)
    plt.tight_layout()
    plt.savefig('Heatmap_Aerosol_flux_and_flux_per_unit_SIC.png')
    plt.close()
    print('''''')



    print('Biomolecules and OMF')
    ## Calculate max values per regions for ocean biomolecules and OMF
    panel_names = ['PCHO', 'DCAA', 'PL', 'Biom_tot', 'OMF_POL', 'OMF_PRO', 'OMF_LIP', 'OMF_tot']
    seaice_min = get_min_seaice(variables_info_yr, 'Sea_ice')
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']
    #apply min ice mask
    for var_na in panel_names:
        print(var_na)
        data_seaice_mask = np.ma.masked_where(seaice_min > 10, variables_info_yr[var_na]['slope'])
        data_seaice_mask = np.ma.masked_where(np.isnan(variables_info_yr[var_na]['pval']), data_seaice_mask)
        data_seaice_mask = data_seaice_mask.filled(np.nan)

        variables_info_yr[var_na]['regions_vals'] = reg_sel(lat, lon, data_seaice_mask)
        print('''''')

        # print(data_seaice_mask)




    reg_names = regions()
    var_na_sw_aer = ['PCHO$_{sw}$', 'DCAA$_{sw}$', 'PL$_{sw}$', 'Total$_{sw}$']
    panel_names = ['PCHO', 'DCAA', 'PL', 'Biom_tot']
    columns = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns[0].append(reg_na)
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)
    col_name_oc = 'Ocean concentration \n (mmol C ${m^{-3}}$ ${yr^{-1}}$)'
    df_vals_piv_ocean = create_df_plot_heatmap(columns, col_name_oc)
    plot_heatmap(df_vals_piv_ocean, col_name_oc, 'Ocean_conc_abs_max_')

    var_na_sw_aer = ['PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$', 'Total$_{aer}$']
    panel_names = ['OMF_POL', 'OMF_PRO', 'OMF_LIP', 'OMF_tot']
    columns = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns[0].append(reg_na)
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)
    col_name_omf = 'OMF (% ${yr^{-1}}$)'
    df_vals_piv_omf = create_df_plot_heatmap(columns, col_name_omf)
    plot_heatmap(df_vals_piv_omf, col_name_omf, 'OMF_abs_max_')


    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax.flatten()
    plot_each_heatmap(ax[0], df_vals_piv_ocean, col_name_oc)
    ax[0].set_title(r'$\bf{(a)}$', loc='left')
    plot_each_heatmap(ax[1], df_vals_piv_omf, col_name_omf)
    ax[1].set_title(r'$\bf{(b)}$', loc='left')
    plt.tight_layout()
    plt.savefig('Heatmap_Ocean_OMF.png')
    plt.close()