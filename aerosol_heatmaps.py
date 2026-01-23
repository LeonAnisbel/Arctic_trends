from math import isnan

import numpy as np
import pandas as pd
from scipy.stats import alpha

import biomolecule_heatmaps
import global_vars
import utils
from process_statsmodels import process_array_slope_per_ice
from utils import regions
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

def plot_heatmap(df_vals_piv, col_name, fig_title):
    fig, ax = plt.subplots(1, 1,
                           figsize=(7, 5), )
    biomolecule_heatmaps.plot_each_heatmap(ax, df_vals_piv, col_name)
    plt.tight_layout()
    plt.savefig('./plots/heatmap_' + fig_title + '.png')
    plt.close()


def percent_icrease(variables_info_yr, vv, reg_na, decade, cond, tau_values=False):
    """ This function computes the percent of increase per year
    :return an array containing only significant values when cond == 'significant'
    and all values if cond != 'significant'"""
    pval = variables_info_yr[vv][reg_na][decade]['pval_aver_reg']

    if vv == 'Sea_ice_area_px' or vv == 'AER_SIC_area_px' or vv[-2:] == '_m' or vv[:10] == 'AER_burden':
        data_type_mean_or_sum = 'data_sum_reg'
    else:
        data_type_mean_or_sum = 'data_aver_reg'

    if cond == 'significant':
        if pval < 0.05:
            perc_inc =(100*variables_info_yr[vv][reg_na][decade]['slope_aver_reg']/
                       variables_info_yr[vv][reg_na][decade][data_type_mean_or_sum].mean().values)
        else:
            perc_inc = 0
            pval = 0
    else:
        perc_inc =(100*variables_info_yr[vv][reg_na][decade]['slope_aver_reg']/
                   variables_info_yr[vv][reg_na][decade][data_type_mean_or_sum].mean().values)
        if np.isnan(perc_inc):
            perc_inc = 0


    if tau_values:
        return perc_inc, pval
    else:
        return perc_inc


def cols_df(variables_info_yr, panel_names, var_na_title, decade, type):
    """ This function selects the slope values for each variable
    :return lists of slope values and percent of change when the trend is statistically significant"""
    panels = len(panel_names)
    columns = panels * [[]]
    for col in range(len(columns)):
        columns[col] = [[] for i in range(4)]
    reg_names = regions()
    for vidx, var_na in enumerate(panel_names):
        if type[:5] == 'slope' and var_na[:5]=='AER_F':
            factor = global_vars.factor_eim_heatmaps
        elif type[:5] == 'slope' and var_na[:7]=='AER_SIC':
            factor = global_vars.factor_sic_heatmaps
        elif type[:5] == 'slope' and var_na[-2:]=='_m':
            factor = global_vars.factor_eim_tot_heatmaps
        else:
            factor = 1.
        for reg_na in reg_names:
            columns[vidx][0].append(reg_na)
            if variables_info_yr[var_na][reg_na][decade]['significance'] < 0.05:
                slope1 = variables_info_yr[var_na][reg_na][decade]['slope_aver_reg'] * factor
            else:
                slope1 = np.nan
            percent_icr = percent_icrease(variables_info_yr, var_na, reg_na, decade, 'significant')
            columns[vidx][1].append(var_na_title[vidx])  # 'SIC (% ${yr^{-1}}$)'
            columns[vidx][2].append(slope1)
            columns[vidx][3].append(percent_icr)
    return columns





def plot_heatmap_multipanel(variables_info, panel_names, var_na_aer, right_label_show, no_ylabel_show, col_name,
                            decade, type, label_loc, panel, settitle=False):
    """ Creates multipanel figure of heatmap plots
    :return None"""
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
    plt.savefig(f'./plots/{season}_heatmap_Emission_SIC_SST_Wind_{type}_{decade}.png', dpi=300)
    plt.close()
    return None


font = 12
def each_panel_fig(data, names_var, ax, title, lims, upper_panel = False, thesis_plot=False, one_biom=False):
    """ Create bar plots of percent of change per year for each Arctic subregion and marine aerosol specie
    :return None"""
    pl = sns.barplot(data=data,
                     x='Regions',
                     y='% per year',
                     hue='variables',
                     palette=['lightgrey', 'lightgrey', 'lightgrey', 'lightgrey'],
                     errorbar=None,
                     edgecolor='black',
                     width=0.8,
                     alpha=0.8,
                     ax=ax)
    palette = global_vars.colors_arctic_reg
    ax.set_title(title,
                 loc='center',
                 weight='bold',)

    pl.legend(loc='upper right',
              bbox_to_anchor=(0.85, 1.5),
              ncol=4,
              fontsize=font)

        # # Add pvalues on top of the bars
    hue_col = data["Regions"].unique()
    labels = [[], [], [], []]
    for col in hue_col:
        for i, var in enumerate(names_var):
            data_reg = data[data["Regions"] == col]
            pval = data_reg[data_reg["variables"]==var]['% per year'].values[0]
            if data_reg[data_reg["variables"]==var]['% per year'].values[0] == 0.0:
                labels[i].append('')
            else:
                labels[i].append(round(pval,1))#f"{pval:.1e}".replace("e+", "×10^")f"{pval:.1g}"

    for c, lab in zip(ax.containers, labels):
        # add the name annotation to the top of the bar
        ax.bar_label(c, labels=lab, padding=3, fontsize=font-4, rotation=90)  #  if needed
        ax.margins(y=0.1)


    list_hatch = ['////', '----', '....', 'xxxx']
    for bars, hatch, legend_handle in zip(ax.containers, list_hatch, pl.legend_.legend_handles):
        for bar, color in zip(bars, palette):
            bar.set_facecolor(color)
            bar.set_hatch(hatch)
        # update the existing legend, use twice the hatching pattern to make it denser
        legend_handle.set_hatch(hatch + hatch)

    hue_col = data["variables"].unique()
    all_pvals = []
    for col in hue_col:
        pval = data[data["variables"]==col]['pval'].values
        for p in pval:
            all_pvals.append(p)
    for bar, val in zip(ax.patches, all_pvals):
        if val > 0.05:
            bar.set_alpha(0.1)

    ax.tick_params(axis='x', labelsize=font, rotation=65)
    ax.set(xlabel=None)
    ax.set_ylim(lims[0], lims[1])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(lims[2]))
    ax.xaxis.get_label().set_fontsize(font)
    ax.yaxis.get_label().set_fontsize(font)
    ax.tick_params(axis='both',
                   labelsize=font)

    ax.grid(linestyle='--',
            linewidth=0.4)

    if upper_panel and not thesis_plot:
        ax.set_xticklabels([])
    elif not upper_panel and thesis_plot:
        ax.legend_.remove()

    # if title!= r'$\bf{(e)}$':
    #     ax.set(xlabel=None)
    #     ax.set_xticklabels([])
    # ax.set_title(title,
    #              loc='left',
    #              fontsize=font)


    return None


if __name__ == '__main__':

    season = global_vars.season_to_analise
    with open(f"TrendsDict_{season}_orig_data.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)


    print('Aerosols from ECHAM')
    # panel_names = ['AER_F_POL_m', 'AER_F_PRO_m', 'AER_F_LIP_m', 'AER_F_SS_m']
    # var_na_aer = ['PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$', 'SS$_{aer}$']
    # lat = variables_info_yr[panel_names[0]]['lat']
    # lon_360 = variables_info_yr[panel_names[0]]['lon']
    # lon = ((lon_360 + 180) % 360) - 180
    #
    # decade = '1990-2019'
    # columns_emi = cols_df(variables_info_yr, panel_names, var_na_aer, decade, 'slope')
    #
    # columns_sic = cols_df(variables_info_yr, ['AER_SIC'], [''], decade, 'slope')
    # columns_sst = cols_df(variables_info_yr, ['AER_SST'], [''], decade, 'slope')
    # columns_u10 = cols_df(variables_info_yr, ['AER_U10'], [''], decade, 'slope')

    ###############################
    decades = ['1990-2019', '1990-2004', '2005-2019']

    names_var = [['SS', 'PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$']]
    fig_title = ['flux', 'concentration']



    fig, axs = plt.subplots( 3, 1, figsize=(8, 9))
    axs.flatten()
    limits = global_vars.seasons_info[global_vars.season_to_analise]['bar_plot_lims']
    var_list = [[['AER_F_SS_m'], ['AER_F_POL_m'], ['AER_F_PRO_m'], ['AER_F_LIP_m']],
                [['AER_SS'], ['AER_POL'], ['AER_PRO'], ['AER_LIP']],
                [['AER_burden_SS'], ['AER_burden_POL'], ['AER_burden_PRO'], ['AER_burden_LIP']],
                [['AER_INP_POL'], ['AER_INP_POL'], ['AER_INP_POL'], ['AER_INP_POL']]
                ]

    # for cond in ['not significant', 'significant']:
    data_df_list = []
    title_list = []
    for a, ax in enumerate(axs):
        col_var = []
        col_reg = []
        col_pval = []
        col_perc = []
        region = utils.regions()
        data_dict = {}
        for reg, idx in region.items():
            for v,var in enumerate(var_list[a]):
                stat = variables_info_yr[var[0]][reg][decades[0]]
                col_var.append(names_var[0][v])
                col_reg.append(reg)
                perc_increase, pval = percent_icrease(variables_info_yr, var[0], reg, decades[0], 'not significant', tau_values=True)
                col_pval.append(pval)
                col_perc.append(perc_increase)

        data_dict['variables'] = col_var
        data_dict['Regions']= col_reg
        data_dict['% per year']= col_perc
        data_dict['pval']= col_pval
        data_df = pd.DataFrame(data=data_dict)
        data_df_list.append(data_df)

        if a == 0:
            panel = True
            title = 'Total emission mass flux'
        else:
            panel = False
            title = 'Aerosol concentration'
        title_list.append(title)
        # each_panel_fig(data_df, names_var[0], ax, title, limits[a], upper_panel=panel)
        # plt.tight_layout()
        # plt.savefig(f'plots/bar_plot_{global_vars.season_to_analise}.png', dpi=300)
    plt.close()

###############################
    #thesis plots
    fig, axs = plt.subplots( 1, 1, figsize=(8, 7))
    each_panel_fig(data_df_list[0], names_var[0], axs, title_list[0], limits[0], upper_panel=True, thesis_plot=True)
    plt.tight_layout()
    plt.savefig(f'plots/bar_plot_emiss_{global_vars.season_to_analise}.png', dpi=300)
    plt.close()

    fig, axs = plt.subplots( 1, 1, figsize=(8, 7))
    each_panel_fig(data_df_list[1], names_var[0], axs, title_list[1], limits[1], upper_panel=True, thesis_plot=True)
    plt.tight_layout()
    plt.savefig(f'plots/bar_plot_conc_{global_vars.season_to_analise}.png', dpi=300)
    plt.close()

    fig, axs = plt.subplots( 1, 1, figsize=(8, 7))
    each_panel_fig(data_df_list[2], names_var[0], axs, 'Accumulated atmospheric burden',
                   limits[2], upper_panel=True, thesis_plot=True)
    plt.tight_layout()
    plt.savefig(f'plots/bar_plot_burden_{global_vars.season_to_analise}.png', dpi=300)
    plt.close()

    # fig, axs = plt.subplots( 1, 1, figsize=(8, 7))
    # each_panel_fig(data_df_list[3], names_var[0], axs, 'Average INP burden',
    #                limits[2], upper_panel=True, thesis_plot=True)
    # plt.tight_layout()
    # plt.savefig(f'plots/bar_plot_inp_burden_{global_vars.season_to_analise}.png', dpi=300)
    # plt.close()
###############################

    panel_names_var = [[['AER_SIC_area_px'], ['AER_SST'], ['AER_F_SS'], ['AER_F_POL'], ['AER_F_PRO'], ['AER_F_LIP']],
                       [['AER_SIC_area_px'], ['AER_SST'], ['AER_SS'], ['AER_POL'], ['AER_PRO'], ['AER_LIP']]]
    for j, panel_names in enumerate(panel_names_var):
        var_na_aer = [['Sea Ice \n area'], ['SST'], ['SS$_{aer}$'], ['PCHO$_{aer}$'], ['DCAA$_{aer}$'], ['PL$_{aer}$']]
        right_label_show = [True, True, True, False, False, True]
        no_ylabel_show = [False, True, True, False, True, True]
        col_emi_name_sl = 4 * [' Emission mass flux \n (10$^{-2}$ ng m$^{-2}$ s$^{-1}$ yr$^{-1}$) \n']

        col_name_sl = ['\n Sea Ice area \n (10$^{6}$ m${^{3}}$ yr${^{-1}}$) \n',
                       '\n SST \n (C$^{o}$ yr${^{-1}}$)']

        col_emi_name_ic = 4 * [' Emission \n (% yr${^{-1}}$)']
        col_name_ic = ['\n SIC \n (% yr${^{-1}}$)',
                       '\n SST \n (% yr${^{-1}}$)']

        for c, cc in zip(col_emi_name_sl, col_emi_name_ic):
            col_name_sl.append(c)
            col_name_ic.append(cc)
        label_loc = [[0, 1, 2, 3],
                     [r'$\bf{(a)}$' + '\n ',
                      r'$\bf{(b)}$' + '\n ',
                      r'$\bf{(c)}$' + '\n ',
                      r'$\bf{(d)}$']]
        for dec in decades:
            type = 'slope_'+fig_title[j]
            plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                    col_name_sl, dec, type, label_loc, [[2, 3], [8, 8]])

            type = 'percent_'+fig_title[j]
            plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                    col_name_ic, dec, type, label_loc, [[2, 3], [8, 8]])

###############################
    panel_names_var = [
        [['AER_SIC'], ['AER_F_POL'], ['AER_F_PRO'], ['AER_F_LIP'], ['AER_F_SS']],
        [['AER_SIC'], ['AER_POL'], ['AER_PRO'], ['AER_LIP'], ['AER_SS']]]
    fig_title = ['flux', 'concentration']

    for j, panel_names in enumerate(panel_names_var):
        var_na_aer = [['SIC'], ['PCHO$_{aer}$'], ['DCAA$_{aer}$'], ['PL$_{aer}$'], ['SS$_{aer}$']]
        right_label_show = [True, False, False, False, True]
        no_ylabel_show = [False, True, True, True, True]
        col_name_sl = ['\n (% yr${^{-1}}$) \n']

        for c in col_emi_name_sl:
            col_name_sl.append(c)

        col_name_ic = 5 * [' Percent of increase per year' + '\n ']
        label_loc = [[0, 1, 2, 3, 4],
                     [r'$\bf{(a)}$',
                      r'$\bf{(b)}$',
                      r'$\bf{(c)}$',
                      r'$\bf{(d)}$',
                      r'$\bf{(e)}$']]

        decades = ['1990-2019']
        for dec in decades:
            type = 'slope_emi_only_'+fig_title[j]
            plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                    col_name_sl, dec, type, label_loc, [[1, 5], [10, 4]])

            type = 'percent_emiss_only_'+fig_title[j]
            plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                    col_name_ic, dec, type, label_loc, [[1, 5], [10, 4]], settitle=True)
    ###############################
    panel_names_var = [['AER_F_POL_m'], ['AER_F_PRO_m'], ['AER_F_LIP_m'], ['AER_F_SS_m']]
    fig_title = ['flux', 'concentration']

    var_na_aer = [['PCHO$_{aer}$'], ['DCAA$_{aer}$'], ['PL$_{aer}$'], ['SS$_{aer}$']]
    right_label_show = [False, False, False, True]
    no_ylabel_show = [False, True, True, True]
    col_emi_name_sl = 4 * [' Total emission mass flux \n (10$^{-3}$ Tg season$^{-1}$ yr$^{-1}$) \n']


    label_loc = [[0, 1, 2, 3],
                 [r'$\bf{(a)}$',
                  r'$\bf{(b)}$',
                  r'$\bf{(c)}$',
                  r'$\bf{(d)}$']]

    decades = ['1990-2019']
    for dec in decades:
        type = 'slope_tot_emi_only_'+fig_title[j]
        plot_heatmap_multipanel(variables_info_yr, panel_names, var_na_aer, right_label_show, no_ylabel_show,
                                col_emi_name_sl, dec, type, label_loc, [[1, 4], [9, 4]])

