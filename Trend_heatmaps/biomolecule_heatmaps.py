import numpy as np
from Utils_functions import global_vars
from Utils_functions.utils import get_var_reg, get_seaice_vals, get_min_seaice, regions, get_conds
import pickle
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

def plot_test_map(reg_sel_vals, var_na, reg_na):
    """ Creates map of a specific region. It is used for checking the region definition
    :return None"""
    if var_na == 'PCHO' and reg_na == 'Greenland & Norwegian Sea':
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               sharex=True,
                               subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
        ax.set_extent([-180, 180, 60, 90],
                      ccrs.PlateCarree())
        cmap = plt.get_cmap('Blues',
                            15)
        cb = ax.pcolormesh(reg_sel_vals.lon,
                           reg_sel_vals.lat,
                           np.array(reg_sel_vals['slope']),
                           vmax=0.05,
                           cmap=cmap,
                           transform=ccrs.PlateCarree())
        ax.coastlines(color='darkgray')
        plt.colorbar(cb)
        gl = ax.gridlines(draw_labels=True, )
        gl.ylocator = mticker.FixedLocator([65, 75, 85])
        plt.savefig(f'TEST.png')


def scatter_plot(fig, axs, df, col_name, title, vm, font, no_left_labels=False, no_colorbar=False):
    """ Creates scatter plots of grid fraction with significant, increasing and decreasing trend
    :return None"""
    sc = axs.scatter(
        x=df['Variables'],
        y=df['Regions'],
        c=df[col_name],
        s=df[col_name],
        cmap='viridis',
        vmax=vm,
    )

    axs.tick_params(axis='both',
                    pad=0.2,
                    labelsize=font)
    axs.xaxis.labelpad = 0.2
    # plt.xlim((-1,1))
    axs.set_xlim((-0.5, 2.3))

    if no_left_labels:
        axs.set(yticklabels=[])
        axs.tick_params(left=False)
    axs.set(ylabel="", xlabel="")
    axs.set_title(title[0],
                  loc='right',
                  fontsize=font)

    if no_colorbar:
        plt.colorbar(sc, ax=axs).remove()
    else:
        cbar = plt.colorbar(sc, ax=axs)
        cbar.set_label(title[1],
                       fontsize=font-2
                       )
    return None

def create_df_plot_heatmap(col, col_name, return_colorbar=False):
    """ This function creates a dataframe use to plot the heatmap
    :return dataframe or dataframe, cmap and min and max values"""
    df_vals = pd.DataFrame({'Regions': col[0],
                            col_name: col[1],
                            'Values': col[2],
                            })
    if col_name[:3] == 'OMF' or col_name[:3] == 'Oce':
        df_vals = df_vals[df_vals['Regions'] != 'Central Arctic']
    # if col_name == ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ per unit SIC)':
    #     df_vals = df_vals[df_vals['Regions'] != 'Barents Sea']
    df_vals_piv = df_vals.pivot(index="Regions",
                                columns=col_name,
                                values="Values")

    if return_colorbar:
        vmin_val = min(df_vals['Values'])
        vmax_val = max(df_vals['Values'])
        cmap = 'Greens_r'
        if vmin_val < 0 and vmax_val > 0:
            cmap = 'RdBu_r'
            vmin_val = -vmax_val
        if vmin_val > 0:
            cmap = 'Reds'

        if vmax_val < 0:
            cmap = 'Blues_r'
        return df_vals_piv, [cmap, vmin_val, vmax_val]
    else:
        return df_vals_piv


def plot_each_heatmap(ax, df_vals_piv, fig_title, cmap, no_ylabel=False, right_label=True):
    """ plots each heatmap
    :return None"""
    hm = sns.heatmap(df_vals_piv,
                     annot=True,
                     vmin=cmap[1],
                     vmax=cmap[2],
                     cmap=cmap[0],
                     ax=ax)
    if no_ylabel:
        hm.set(yticklabels=[])
        ax.tick_params(left=False, bottom=False)
    hm.set(ylabel="", xlabel="")
    hm.xaxis.tick_top()
    font = 12
    if right_label:
        hm.set_title(fig_title,
                     loc='right',
                     fontsize=font)


def reg_sel(lat, lon, data, var_na):
    """ This function calculates the percent of grid with significant trend and what fraction
     of it has an increasing and a decreasing trend
    :return dictionary with this information """
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
        conditions = get_conds(data_ds.lat,
                               data_ds.lon)

        reg_sel_vals = get_var_reg(data_ds,
                                   conditions[idx])

        total_non_nan = np.sum(np.logical_not(np.isnan(reg_sel_vals.slope.values)))
        total_grid_size = len(reg_sel_vals['slope'].lat) * len(reg_sel_vals['slope'].lon)

        if reg_sel_vals.slope.shape[0] > 1 and total_non_nan > 0:
            reg_data[reg_na]['grid_signif'] = total_non_nan * 100 / total_grid_size

            max_val = reg_sel_vals['slope'].max(skipna=True).values
            reg_data[reg_na]['max'] = max_val
            min_val = reg_sel_vals['slope'].min(skipna=True).values
            reg_data[reg_na]['min'] = min_val

            reg_data[reg_na]['max_val'] = float(max_val)
            grid_posit_vals = reg_sel_vals['slope'].where(reg_sel_vals['slope'] > 0, drop=True)

            reg_data[reg_na]['fraction_grid_posit'] = np.sum(
                np.logical_not(np.isnan(grid_posit_vals.values))) * 100 / total_non_nan

            reg_data[reg_na]['min_val'] = float(min_val)
            grid_negat_vals = reg_sel_vals['slope'].where(reg_sel_vals['slope'] < 0, drop=True)
            reg_data[reg_na]['fraction_grid_negat'] = np.sum(
                np.logical_not(np.isnan(grid_negat_vals.values))) * 100 / total_non_nan


            if abs(min_val) > abs(max_val):
                if reg_data[reg_na]['fraction_grid_negat'] > reg_data[reg_na]['fraction_grid_posit']:
                    reg_data[reg_na]['max_absolute'] = float(min_val)
                else:
                    reg_data[reg_na]['max_absolute'] = float(max_val)

            else:
                if reg_data[reg_na]['fraction_grid_negat'] < reg_data[reg_na]['fraction_grid_posit']:
                    reg_data[reg_na]['max_absolute'] = float(max_val)
                else:
                    reg_data[reg_na]['max_absolute'] = float(min_val)
        else:
            reg_data[reg_na]['max_absolute'] = np.nan
            reg_data[reg_na]['grid_signif'] = np.nan

            reg_data[reg_na]['max_val'] = np.nan
            reg_data[reg_na]['fraction_grid_posit'] = np.nan

            reg_data[reg_na]['min_val'] = np.nan
            reg_data[reg_na]['fraction_grid_negat'] = np.nan
    return reg_data

def subplots_plot_heatmap_scatter(ax, df_vals_piv, df_grid_percent, col_name, cmaps, label, font, percent=True):
    """ Creates multipanel plot with heatmaps and scatter plots of fraction of grid with increasing trend
    :return None"""

    plot_each_heatmap(ax[0],
                      df_vals_piv[0],
                      col_name,
                      cmaps[0],
                      right_label=False)
    plot_each_heatmap(ax[1],
                      df_vals_piv[1],
                      col_name,
                      cmaps[1],
                      no_ylabel=True,
                      right_label=False)
    plot_each_heatmap(ax[2],
                      df_vals_piv[2],
                      col_name,
                      cmaps[2],
                      no_ylabel=True)

    if percent:
        ax[3].set_title(label[1] + '\n ',
                        loc='left',
                        fontsize=font + 2)
        vm = 100
        scatter_plot(fig, ax[3],
                     df_grid_percent,
                     col_name_grid_pos,
                     ['Fraction with \n increasing trend',
                      'Grid fraction (%)'],
                     vm,
                     font,
                     no_left_labels=False,
                     no_colorbar=True)

        scatter_plot(fig, ax[4],
                     df_grid_percent,
                     col_name_grid_neg,
                     ['Fraction with \n decreasing trend',
                      'Grid fraction (%)'],
                     vm,
                     font,
                     no_left_labels=True,
                     no_colorbar=False)
        scatter_plot(fig, ax[5],
                     df_grid_percent,
                     col_name_signif,
                     ['', 'Grid fraction with \n significant trend (%)'],
                     30,
                     font,
                     no_left_labels=True,
                     no_colorbar=False)
        ax[5].set_title(label[2] + '\n ',
                        loc='left',
                        fontsize=font + 2)
        for a in ax[3:]:
            a.tick_params(axis='x',
                              rotation=45)

    ax[0].tick_params(axis='y',
                  labelsize=font,)
    ax[0].set_title(label[0],
                    loc='left',
                    fontsize=font)
    if percent:
        for a in ax[:3]:
            a.tick_params(axis='y',
                          labelsize=font, )
        ax[0].set_title(label[0],
                        loc='left',
                        fontsize=font + 2)
    else:
        ax[0].tick_params(axis='y',
                          labelsize=font+2, )
    return None

def create_df_pivot(variables_info_yr, var_na_sw_aer, panel_names, col_name_oc):
    """ This function creates a dataframe use to plot the heatmap and scatter plot of percent
    of grid with significant trend for ocean biomolecules and OMF
    :return dataframes and cmap used to plotting"""
    columns = [[], [], []]
    columns1 = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[0].append(reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_posit']
            columns1[0].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_negat']
            columns1[1].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['grid_signif']
            columns1[2].append(max_abs)

    data_percent = {'Regions': columns[0],
                    'Variables': columns[1],
                    col_name_grid_pos: columns1[0],
                    col_name_grid_neg: columns1[1],
                    col_name_signif: columns1[2], }
    df_ocean_grid_percent = pd.DataFrame(data_percent).sort_values(by=['Regions', 'Variables'],
                                                                   ascending=[False, True])
    df_ocean_grid_percent = df_ocean_grid_percent[df_ocean_grid_percent['Regions'] != 'Central Arctic']

    df_vals_piv_ocean, cmaps = [], []
    for vidx, var_na in enumerate(panel_names):
        columns = [[], [], []]
        for reg_na in list(reg_names.keys())[1:]:
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[0].append(reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)

        df_vals_piv, cm = create_df_plot_heatmap(columns, col_name_oc, return_colorbar=True)
        df_vals_piv_ocean.append(df_vals_piv)
        cmaps.append(cm)
    return df_vals_piv_ocean, df_ocean_grid_percent, cmaps


if __name__=='__main__':
    season = global_vars.season_to_analise
    with open(f"TrendsDict_{season}_orig_data.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    print('Biomolecules and OMF')
    ## Calculate max values per regions for ocean biomolecules and OMF
    panel_names = ['PCHO', 'DCAA', 'PL', 'Biom_tot', 'OMF_POL', 'OMF_PRO', 'OMF_LIP', 'OMF_tot']
    seaice = get_seaice_vals(variables_info_yr, 'Sea_ice')
    seaice_min = get_min_seaice(variables_info_yr, 'Sea_ice')
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']

    # apply min ice mask
    for var_na in panel_names:
        # print(var_na, seaice[2].shape,seaice_min.shape, variables_info_yr[var_na]['slope'].shape)
        data_seaice_mask = np.ma.masked_where(seaice_min > 10,
                                              variables_info_yr[var_na]['slope'])
        data_seaice_mask = np.ma.masked_where(np.isnan(variables_info_yr[var_na]['significance']),
                                              data_seaice_mask)
        data_seaice_mask = data_seaice_mask.filled(np.nan)
        variables_info_yr[var_na]['regions_vals'] = reg_sel(lat,
                                                            lon,
                                                            data_seaice_mask,
                                                            var_na)

    reg_names = regions()
    col_name_oc = 'Ocean concentration (mmol C m${^{-3}}$ yr${^{-1}}$)'
    var_na_sw_aer = ['PCHO$_{sw}$', 'DCAA$_{sw}$', 'PL$_{sw}$']  # , 'Total$_{sw}$']
    panel_names = ['PCHO', 'DCAA', 'PL']  # , 'DCAA','Biom_tot']

    col_name_grid_pos = '% of grid with increasing trend'
    col_name_grid_neg = '% of grid with decreasing trend'
    col_name_signif = '% of grid with significant trend'
    df_vals_piv_ocean, df_ocean_grid_percent, cmaps = create_df_pivot(variables_info_yr,
                                                                      var_na_sw_aer,
                                                                      panel_names,
                                                                      col_name_oc)
#################################################################################################
    #OMF
    col_name_omf = 'OMF (% yr${^{-1}}$)'

    var_na_sw_aer = ['PCHO$_{aer}$','DCAA$_{aer}$', 'PL$_{aer}$', ]  #   'DCAA$_{aer}$']
    panel_names = ['OMF_POL','OMF_PRO', 'OMF_LIP', ]  # , 'OMF_tot']
    df_vals_piv_omf, df_omf_grid_percent, cmaps_omf = create_df_pivot(variables_info_yr,
                                                              var_na_sw_aer,
                                                              panel_names,
                                                              col_name_omf)

#################################
#### plot figure with absolute maximum trends only
    fig, axs = plt.subplots(1, 3,
                            figsize=(8, 4), )
    plt.subplots_adjust(wspace=0.1)
    ax = axs.flatten()

    font = 8
    label = [r'', r'' + '\n', r'' + '\n']
    subplots_plot_heatmap_scatter(ax,
                                  df_vals_piv_ocean,
                                  df_ocean_grid_percent,
                                  col_name_oc,
                                  cmaps,
                                  label,
                                  font,
                                  percent=False)
    plt.tight_layout()
    plt.savefig(f'./plots/{season}_heatmap_scatter_max_abs_biomolcules.png',
                dpi=300)
    plt.close()

#### plot figure of absolute maximum trends with grid fracions of increasing or decreasing trend
    fig, axs = plt.subplots(2, 3,
                            figsize=(7, 7),)
    plt.subplots_adjust(wspace=0.1)
    ax = axs.flatten()

    font = 10
    label = [r'$\bf{(a)}$', r'$\bf{(b)}$'+'\n', r'$\bf{(c)}$'+'\n']
    subplots_plot_heatmap_scatter(ax,
                                  df_vals_piv_ocean,
                                  df_ocean_grid_percent,
                                  col_name_oc,
                                  cmaps,
                                  label,
                                  font)
    plt.tight_layout()
    plt.savefig(f'./plots/{season}_heatmap_scatter_plots_grid_fraction_biomolcules.png',
                dpi=300)
    plt.close()

#### plot figure of absolute maximum trends with grid fractions of increasing or decreasing trend for both ocean
# concentration and OMF
    fig, axs = plt.subplots(2,
                            6,
                            figsize=(14, 8))#14, 10
    plt.subplots_adjust(wspace=0.005)
    ax = axs.flatten()
    ax1 = ax[:6]
    subplots_plot_heatmap_scatter(ax1,
                                  df_vals_piv_ocean,
                                  df_ocean_grid_percent,
                                  col_name_oc,
                                  cmaps,
                                  label,
                                  font)

    label2 = [r'$\bf{(d)}$', r'$\bf{(e)}$'+'\n', r'$\bf{(f)}$'+'\n']
    ax2 = ax[6:]
    subplots_plot_heatmap_scatter(ax2,
                                  df_vals_piv_omf,
                                  df_omf_grid_percent,
                                  col_name_omf,
                                  cmaps_omf,
                                  label2,
                                  font)
    plt.tight_layout()
    plt.savefig(f'./plots/{season}_heatmap_scatter_plots_grid_fraction_biomolcules_OMF.png',
                dpi=300)
    plt.close()
