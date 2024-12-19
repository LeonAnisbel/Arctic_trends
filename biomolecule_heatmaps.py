import numpy as np
from utils import get_var_reg, get_seaice_vals, get_min_seaice, regions, get_conds
import pickle
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_test_map(reg_sel_vals, var_na, reg_na):
    if var_na == 'PCHO' and reg_na == 'Greenland & Norwegian Sea':
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        from matplotlib import ticker as mticker
        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               sharex=True,
                               subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
        ax.set_extent([-180, 180, 60, 90],
                      ccrs.PlateCarree())
        cmap = plt.get_cmap('Blues', 15)
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


def scatter_plot(fig, axs, df, col_name, title, vm, no_left_labels=False, no_colorbar=False):
    sc = axs.scatter(
        x=df['Variables'],
        y=df['Regions'],
        c=df[col_name],
        s=df[col_name],
        cmap='viridis',
        vmax=vm,
    )

    axs.tick_params(axis='x', pad=0.2)
    axs.xaxis.labelpad = 0.2
    # plt.xlim((-1,1))
    axs.set_xlim((-0.5, 1.5))

    if no_left_labels:
        axs.set(yticklabels=[])
        axs.tick_params(left=False)
    axs.set(ylabel="", xlabel="")
    axs.set_title(title[0], loc='right')

    if no_colorbar:
        plt.colorbar(sc, ax=axs).remove()
    else:
        cbar = plt.colorbar(sc, ax=axs)
        cbar.set_label(title[1])

def create_df_plot_heatmap(col, col_name, return_colorbar=False):
    df_vals = pd.DataFrame({'Regions': col[0],
                            col_name: col[1],
                            'Values': col[2],
                            })
    if col_name[:3] == 'OMF' or col_name[:3] == 'Oce':
        df_vals = df_vals[df_vals['Regions'] != 'Central Arctic']
    # if col_name == ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ per unit SIC)':
    #     df_vals = df_vals[df_vals['Regions'] != 'Barents Sea']
    df_vals_piv = df_vals.pivot(index="Regions", columns=col_name, values="Values")

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
    if right_label:
        hm.set_title(fig_title, loc='right')


def reg_sel(lat, lon, data, var_na):
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

            #print('\n', var_na, reg_na, reg_data[reg_na]['grid_signif'],
            # reg_data[reg_na]['fraction_grid_posit'],
            # reg_data[reg_na]['fraction_grid_negat'])

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



if __name__=='__main__':
    season = 'JAS'
    with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
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
        data_seaice_mask = np.ma.masked_where(seaice_min > 10, variables_info_yr[var_na]['slope'])
        data_seaice_mask = np.ma.masked_where(np.isnan(variables_info_yr[var_na]['pval']), data_seaice_mask)
        data_seaice_mask = data_seaice_mask.filled(np.nan)
        variables_info_yr[var_na]['regions_vals'] = reg_sel(lat, lon, data_seaice_mask, var_na)
        print('''''')

        # print(data_seaice_mask)

    reg_names = regions()
    var_na_sw_aer = ['PCHO$_{sw}$', 'PL$_{sw}$']  # , 'DCAA$_{sw}$', 'Total$_{sw}$']
    panel_names = ['PCHO', 'PL']  # , 'DCAA','Biom_tot']
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

    col_name_oc = 'Ocean concentration \n (mmol C ${m^{-3}}$ ${yr^{-1}}$)'
    df_vals_piv_ocean, cmap_oc = create_df_plot_heatmap(columns, col_name_oc, return_colorbar=True)
    #    plot_heatmap(df_vals_piv_ocean, col_name_oc, f'{season}_Ocean_conc_abs_max_')

    col_name_grid_pos = '% of grid with increasing trend'
    col_name_grid_neg = '% of grid with decreasing trend'
    col_name_signif = '% of grid with significant trend'
    data_percent = {'Regions': columns[0],
                    'Variables': columns[1],
                    col_name_grid_pos: columns1[0],
                    col_name_grid_neg: columns1[1],
                    col_name_signif: columns1[2], }
    df_ocean_grid_percent = pd.DataFrame(data_percent).sort_values(by=['Regions', 'Variables'], ascending=[False, True])
    df_ocean_grid_percent = df_ocean_grid_percent[df_ocean_grid_percent['Regions'] != 'Central Arctic']

    reg_names = regions()
    var_na_sw_aer = ['PCHO$_{sw}$', 'PL$_{sw}$']  # , 'DCAA$_{sw}$', 'Total$_{sw}$']
    panel_names = ['PCHO', 'PL']  # , 'DCAA','Biom_tot']
    columns = [[], [], []]
    df_vals_piv_ocean_pol_pl, cmap_oc_pol_pl = [], []
    for vidx, var_na in enumerate(panel_names):
        columns = [[], [], []]
        for reg_na in list(reg_names.keys())[1:]:
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[0].append(reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)

        df_vals_piv, cmaps = create_df_plot_heatmap(columns, col_name_oc, return_colorbar=True)
        df_vals_piv_ocean_pol_pl.append(df_vals_piv)
        cmap_oc_pol_pl.append(cmaps)

    var_na_sw_aer = ['PCHO$_{aer}$', 'PL$_{aer}$', ]  # 'DCAA$_{aer}$',  'DCAA$_{aer}$']
    panel_names = ['OMF_POL', 'OMF_LIP', ]  # 'OMF_PRO', 'OMF_tot']
    columns1 = [[], [], []]
    columns2 = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns1[0].append(reg_na)
            columns1[1].append(var_na_sw_aer[vidx])

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_posit']
            columns2[0].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_negat']
            columns2[1].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['grid_signif']
            columns2[2].append(max_abs)

    col_name_grid_pos = '% of grid with increasing trend'
    col_name_grid_neg = '% of grid with decreasing trend'
    col_name_signif = '% of grid with significant trend'

    data_percent = {'Regions': columns1[0],
                    'Variables': columns1[1],
                    col_name_grid_pos: columns2[0],
                    col_name_grid_neg: columns2[1],
                    col_name_signif: columns2[2], }

    df_omf_grid_percent = pd.DataFrame(data_percent).sort_values(by=['Regions', 'Variables'], ascending=[False, True])
    df_omf_grid_percent = df_omf_grid_percent[df_omf_grid_percent['Regions'] != 'Central Arctic']

    var_na_sw_aer = ['PCHO$_{aer}$']  # ,'DCAA$_{aer}$']#
    panel_names = ['OMF_POL']  # ,'OMF_PRO']
    columns = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns[0].append(reg_na)
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            # print(max_abs, var_na, reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)
    col_name_omf = 'OMF \n (% ${yr^{-1}}$)'
    df_vals_piv_omf_pol, cmap_omf_pol = create_df_plot_heatmap(columns, col_name_omf, return_colorbar=True)
    #    plot_heatmap(df_vals_piv_omf, col_name_omf, f'{season}_OMF_abs_max_')

    ###################################3

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    fig = plt.figure(figsize=(14, 10))
    # ax.flatten()
    ax1 = plt.subplot2grid((11, 4), (1, 0), colspan=2, rowspan=5)
    ax2 = plt.subplot2grid((11, 4), (1, 2), rowspan=5)
    ax3 = plt.subplot2grid((11, 4), (1, 3), rowspan=5)

    ax4 = plt.subplot2grid((11, 4), (6, 0), rowspan=5)
    ax5 = plt.subplot2grid((11, 4), (6, 1), rowspan=5)
    ax6 = plt.subplot2grid((11, 4), (6, 2), rowspan=5)
    ax7 = plt.subplot2grid((11, 4), (6, 3), rowspan=5)

    plot_each_heatmap(ax1, df_vals_piv_ocean, col_name_oc, cmap_oc)
    ax1.set_title(r'$\bf{(a)}$' + '\n ', loc='left')
    plot_each_heatmap(ax2, df_vals_piv_omf_pol, col_name_omf, cmap_oc, no_ylabel=True, right_label=False)
    ax2.set_title(r'$\bf{(b)}$' + '\n ', loc='left')

    var_na_sw_aer = ['PL$_{aer}$', ]  # 'Total$_{aer}$']#
    panel_names = ['OMF_LIP', ]  # 'OMF_tot']
    columns = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns[0].append(reg_na)
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            # print(max_abs, var_na, reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)
    col_name_omf = 'OMF \n (% ${yr^{-1}}$)'
    df_vals_piv_omf, cmap_omf_pl = create_df_plot_heatmap(columns, col_name_omf, return_colorbar=True)
    #   plot_heatmap(df_vals_piv_omf, col_name_omf, f'{season}_OMF_abs_max_')

    plot_each_heatmap(ax3, df_vals_piv_omf, col_name_omf, cmap_omf_pl, no_ylabel=True)

    vm = 100
    scatter_plot(fig, ax4,
                 df_ocean_grid_percent,
                 col_name_grid_pos,
                 'Fraction with \n icreasing trend',
                 vm,
                 no_left_labels=False,
                 no_colorbar=True)
    ax4.set_title(r'$\bf{(c)}$' + '\n ', loc='left')

    scatter_plot(fig, ax5,
                 df_ocean_grid_percent,
                 col_name_grid_neg,
                 'Fraction with \n decreasing trend',
                 vm,
                 no_left_labels=True,
                 no_colorbar=False)
    # ax5.set_title('Ocean concentration '+'\n ', loc='right')

    scatter_plot(fig, ax6,
                 df_omf_grid_percent,
                 col_name_grid_pos,
                 'Fraction with \n increasing trend',
                 vm,
                 no_left_labels=True,
                 no_colorbar=True)
    ax6.set_title(r'$\bf{(d)}$' + '\n ', loc='left')

    scatter_plot(fig, ax7,
                 df_omf_grid_percent,
                 col_name_grid_neg,
                 'Fraction with \n decreasing trend',
                 vm,
                 no_left_labels=True,
                 no_colorbar=False)
    # ax7.set_title('OMF '+'\n ', loc='right')

    plt.tight_layout()
    # plt.savefig(f'{season}_Heatmap_Ocean_OMF.png',dpi = 300)
    # plt.close()

    #################################

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax.flatten()
    vm = 10
    scatter_plot(fig, ax[0],
                 df_ocean_grid_percent,
                 col_name_signif,
                 'Ocean',
                 30,
                 no_left_labels=False,
                 no_colorbar=False)
    ax[0].set_title(r'$\bf{(a)}$' + '\n ', loc='left')

    scatter_plot(fig, ax[1],
                 df_omf_grid_percent,
                 col_name_signif,
                 'OMF',
                 25,
                 no_left_labels=True,
                 no_colorbar=False)
    # ax7.set_title('OMF '+'\n ', loc='right')
    plt.tight_layout()
    # plt.savefig(f'{season}_scatter_plots_grid_fraction.png',dpi = 300)
    # plt.close()

    #################################

#### plot figure with grid fracions of increasing or decreasing trend
    fig, axs = plt.subplots(1, 5, figsize=(14, 5))#14, 10
    ax = axs.flatten()
    plot_each_heatmap(ax[0], df_vals_piv_ocean_pol_pl[0], col_name_oc, cmap_oc_pol_pl[0], right_label=False)
    ax[0].set_title(r'$\bf{(a)}$' + '\n ', loc='left')
    plot_each_heatmap(ax[1], df_vals_piv_ocean_pol_pl[1], col_name_oc, cmap_oc_pol_pl[1], no_ylabel=True)
    scatter_plot(fig, ax[4],
                 df_ocean_grid_percent,
                 col_name_signif,
                 ['', 'Grid fraction with \n significant trend (%)'],
                 30,
                 no_left_labels=True,
                 no_colorbar=False)
    ax[2].set_title(r'$\bf{(b)}$' + '\n ', loc='left')
    vm = 100
    scatter_plot(fig, ax[2],
                 df_ocean_grid_percent,
                 col_name_grid_pos,
                 ['Fraction with \n icreasing trend',
                  'Grid fraction (%)'],
                 vm,
                 no_left_labels=True,
                 no_colorbar=True)
    ax[4].set_title(r'$\bf{(c)}$' + '\n ', loc='left')

    scatter_plot(fig, ax[3],
                 df_ocean_grid_percent,
                 col_name_grid_neg,
                 ['Fraction with \n decreasing trend',
                  'Grid fraction (%)'],
                 vm,
                 no_left_labels=True,
                 no_colorbar=False)

    plt.tight_layout()
    plt.savefig(f'./plots/{season}_heatmap_scatter_plots_grid_fraction_biomolcules.png', dpi=300)
    plt.close()

    plot_each_heatmap(ax[5], df_vals_piv_omf_pol, col_name_omf, cmap_omf_pol, right_label=False)
    ax[5].set_title(r'$\bf{(d)}$' + '\n ', loc='left')
    plot_each_heatmap(ax[6], df_vals_piv_omf, col_name_omf, cmap_omf_pl, no_ylabel=True)
    scatter_plot(fig, ax[9],
                 df_omf_grid_percent,
                 col_name_signif,
                 ['', 'Grid fraction with \n significant trend (%)'],
                 25,
                 no_left_labels=True,
                 no_colorbar=False)
    ax[7].set_title(r'$\bf{(e)}$' + '\n ', loc='left')
    scatter_plot(fig, ax[7],
                 df_omf_grid_percent,
                 col_name_grid_pos,
                 ['Fraction with \n increasing trend',
                  'Grid fraction (%)'],
                 vm,
                 no_left_labels=True,
                 no_colorbar=True)
    ax[9].set_title(r'$\bf{(f)}$' + '\n ', loc='left')

    scatter_plot(fig, ax[8],
                 df_omf_grid_percent,
                 col_name_grid_neg,
                 ['Fraction with \n decreasing trend',
                  'Grid fraction (%)'],
                 vm,
                 no_left_labels=True,
                 no_colorbar=False)

    plt.tight_layout()
    plt.savefig(f'./plots/{season}_heatmap_scatter_plots_grid_fraction_biomolcules_OMF.png', dpi=300)
    plt.close()
