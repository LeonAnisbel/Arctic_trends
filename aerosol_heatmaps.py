import numpy as np
import biomolecule_heatmaps
import global_vars
from utils import regions
import pickle
import matplotlib.pyplot as plt

def plot_heatmap(df_vals_piv, col_name, fig_title):
    fig, ax = plt.subplots(1, 1,
                           figsize=(7, 5), )
    biomolecule_heatmaps.plot_each_heatmap(ax, df_vals_piv, col_name)
    plt.tight_layout()
    plt.savefig('./plots/Heatmap_' + fig_title + '.png')
    plt.close()


def percent_icrease(variables_info_yr, vv, reg_na, decade):
    pval = variables_info_yr[vv][reg_na][decade]['significance']
    if pval < 0.05:
        interc = variables_info_yr[vv][reg_na][decade]['intercept_aver_reg']
        slope = variables_info_yr[vv][reg_na][decade]['slope_aver_reg']
        vals = variables_info_yr[vv][reg_na][decade]['data_aver_reg']
        last_val = slope * 30 + interc
        perc_inc = (last_val / interc - 1) * 100 / 30
        perc_inc = (slope/interc) * 100
        print(vv, reg_na, slope, perc_inc, interc, pval)
    else:
        perc_inc = np.nan
    return perc_inc


def cols_df(variables_info_yr, panel_names, var_na_title, decade, type):
    panels = len(panel_names)
    columns = panels * [[]]
    for col in range(len(columns)):
        columns[col] = [[] for i in range(4)]
    print(columns)
    reg_names = regions()
    for vidx, var_na in enumerate(panel_names):
        if type[:5] == 'slope' and var_na[:5]=='AER_F':
            factor = global_vars.factor_eim_heatmaps
        elif type[:5] == 'slope' and var_na[:7]=='AER_SIC':
            factor = global_vars.factor_sic_heatmaps
        else:
            factor = 1.
        for reg_na in reg_names:
            columns[vidx][0].append(reg_na)
            slope1 = variables_info_yr[var_na][reg_na][decade]['slope_aver_reg']*factor
            percent_icr = percent_icrease(variables_info_yr, var_na, reg_na, decade)
            columns[vidx][1].append(var_na_title[vidx])  # 'SIC (% ${yr^{-1}}$)'
            columns[vidx][2].append(slope1)
            columns[vidx][3].append(percent_icr)
    return columns


def plot_heatmap_multipanel(variables_info, panel_names, var_na_aer, right_label_show, no_ylabel_show, col_name,
                            decade, type, label_loc, panel, settitle=False):
    fig, axs = plt.subplots(panel[0][0], panel[0][1],
                            figsize=(panel[1][0], panel[1][1]))
    ax = axs.flatten()
    for idx in range(len(panel_names)):
        columns = cols_df(variables_info,
                          panel_names[idx],
                          var_na_aer[idx],
                          decade,
                          type)
        if type[:5] == 'slope':
            columns_for_heatmap = columns[0]
        else:
            columns_for_heatmap = [columns[0][0], columns[0][1], columns[0][-1]]

        df_vals_piv, cmap = biomolecule_heatmaps.create_df_plot_heatmap(columns_for_heatmap,
                                                   col_name[idx],
                                                   return_colorbar=True)
        biomolecule_heatmaps.plot_each_heatmap(ax[idx],
                          df_vals_piv,
                          col_name[idx],
                          cmap,
                          no_ylabel=no_ylabel_show[idx],
                          right_label=right_label_show[idx])

        for l in range(len(label_loc[0])):
            ax[label_loc[0][l]].set_title(label_loc[1][l], loc='left')

    if settitle:
        ax[-3].set_title(col_name[0])
    plt.tight_layout()
    plt.savefig(f'./plots/{season}_Heatmap_Emission_SIC_SST_Wind_{type}_{decade}.png', dpi=300)
    plt.close()


if __name__ == '__main__':

    season = global_vars.season_to_analise
    with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    with open(f"TrendsDict_per_ice_{season}.pkl", "rb") as myFile:
        variables_info_seaice = pickle.load(myFile)

    print('Aerosols from ECHAM')
    panel_names = ['AER_F_POL_m', 'AER_F_PRO_m', 'AER_F_LIP_m', 'AER_F_SS_m']
    var_na_aer = ['PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$', 'SS$_{aer}$']
    lat = variables_info_yr[panel_names[0]]['lat']
    lon_360 = variables_info_yr[panel_names[0]]['lon']
    lon = ((lon_360 + 180) % 360) - 180

    decade = '1990-2019'
    columns_emi = cols_df(variables_info_yr, panel_names, var_na_aer, decade, 'slope')

    columns_sic = cols_df(variables_info_yr, ['AER_SIC'], [''], decade, 'slope')
    columns_sst = cols_df(variables_info_yr, ['AER_SST'], [''], decade, 'slope')
    columns_u10 = cols_df(variables_info_yr, ['AER_U10'], [''], decade, 'slope')

    ###############################
    decades = ['1990-2019', '1990-2004', '2005-2019']

    panel_names = [['AER_SIC_area_px'], ['AER_SST'], ['AER_F_SS_m'], ['AER_F_POL_m'], ['AER_F_PRO_m'], ['AER_F_LIP_m']]
    var_na_aer = [['Sea Ice \n area'], ['SST'], ['SS$_{aer}$'], ['PCHO$_{aer}$'], ['DCAA$_{aer}$'], ['PL$_{aer}$']]
    right_label_show = [True, True, True, False, False, True]
    no_ylabel_show = [False, True, True, False, True, True]
    col_emi_name_sl = 4 * [' Emission \n (10$^{-7}$ Tg ${month^{-1}}$ ${yr^{-1}}$) \n']
    col_name_sl = ['\n Sea Ice area \n (10$^{6}$ ${m^{3}}$ ${yr^{-1}}$) \n',
                   '\n SST \n (C$^{o}$ ${yr^{-1}}$)']

    col_emi_name_ic = 4 * [' Emission \n (% ${yr^{-1}}$)']
    col_name_ic = ['\n SIC \n (% ${yr^{-1}}$)',
                   '\n SST \n (% ${yr^{-1}}$)']

    for c, cc in zip(col_emi_name_sl, col_emi_name_ic):
        col_name_sl.append(c)
        col_name_ic.append(cc)
    label_loc = [[0, 1, 2, 3],
                 [r'$\bf{(a)}$' + '\n ',
                  r'$\bf{(b)}$' + '\n ',
                  r'$\bf{(c)}$' + '\n ',
                  r'$\bf{(d)}$']]
    for dec in decades:
        type = 'slope'
        plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                col_name_sl, dec, type, label_loc, [[2, 3], [8, 8]])

        type = 'percent'
        plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                col_name_ic, dec, type, label_loc, [[2, 3], [8, 8]])

    ###############################

    panel_names = [['AER_SIC_area_px'], ['AER_F_POL_m'], ['AER_F_PRO_m'], ['AER_F_LIP_m'], ['AER_F_SS_m']]
    var_na_aer = [['SIC'], ['PCHO$_{aer}$'], ['DCAA$_{aer}$'], ['PL$_{aer}$'], ['SS$_{aer}$']]
    right_label_show = [True, False, False, False, True]
    no_ylabel_show = [False, True, True, True, True]
    col_name_sl = ['\n (% ${yr^{-1}}$) \n']

    for c in col_emi_name_sl:
        col_name_sl.append(c)

    col_name_ic = 5 * [' Percent of increase per year' + '\n ']
    label_loc = [[0, 1, 2, 3, 4],
                 [r'$\bf{(a)}$',
                  r'$\bf{(b)}$',
                  r'$\bf{(c)}$',
                  r'$\bf{(d)}$',
                  r'$\bf{(e)}$']]

    for dec in decades:
        type = 'slope_emi_only'
        plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                col_name_sl, dec, type, label_loc, [[1, 5], [10, 4]])

        type = 'percent_emiss_only'
        plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                col_name_ic, dec, type, label_loc, [[1, 5], [10, 4]], settitle=True)
    ###############################