import numpy as np
import pickle
import plots
import utils

def fill_with_nan(panel_var_trend):
    panel_var_trend = np.ma.masked_where(panel_var_trend > 0., panel_var_trend)
    panel_var_tren = panel_var_trend.filled(np.nan)
    return panel_var_tren


def percent_icrease(variables_info_yr, vv, decade):
    pval = variables_info_yr[vv]['significance']
    #if pval < 0.05:
    interc = variables_info_yr[vv]['intercept']
    slope = variables_info_yr[vv]['slope']

    #interc_sign = np.ma.masked_where(pval>0.05, interc)
    #slope_sign = np.ma.masked_where(pval>0.05, slope)
    print(vv, vv[-3:])
    if vv[-3:] == 'POL': l = 0.0001
    if vv[-3:] == 'PRO': l = 0.001
    if vv[-3:] == 'LIP': l = 0.01
    interc_sign = np.ma.masked_where(interc<l, interc)
    slope_sign = np.ma.masked_where(slope==0, slope)

    last_val = slope_sign * 30 + interc_sign
    perc_inc = (last_val / interc_sign - 1) * 100 / 30
    perc_inc = (slope_sign/interc_sign )*30 * 100 / 30

#print(slope, perc_inc, interc, pval)
    #else:
     #   perc_inc = np.nan
    print(perc_inc.max())
    return perc_inc


def plot_trend_emission(variables_info_seaice, variables_info_yr, seaice, season, decade):

    seaice_aer = utils.get_seaice_vals(variables_info_yr, 'AER_SIC')
    panel_names = ['AER_POL', 'AER_PRO', 'AER_LIP', 'AER_SS']

    lat_aer = variables_info_yr[panel_names[0]]['lat']
    lon_aer = variables_info_yr[panel_names[0]]['lon']

    panel_lim = [3.5, 3.5, 3.5, 3.5]
    panel_var_trend, panel_var_pval, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr)
    _, panel_unit = utils.get_perc_increase(variables_info_yr, panel_names)

    fig_titles = [['PCHO$_{aer}$', r'$\bf{(a)}$'], ['DCAA$_{aer}$', r'$\bf{(b)}$'], ['PL$_{aer}$', r'$\bf{(c)}$'],
                  ['SS$_{aer}$', r'$\bf{(d)}$']],

    percent_increase_yr, _ = utils.get_perc_increase(variables_info_yr, panel_names)
    plots.plot_4_pannel_trend(percent_increase_yr,
                              seaice_aer,
                              panel_var_pval,
                              lat_aer,
                              lon_aer,
                              panel_lim,
                              panel_unit,
                              fig_titles,
                              f'{season}_Aerosol_conc_percent_trend_with_mean',
                              not_aerosol=False,
                              percent_increase=False)

    # panel_names = ['AER_F_POL', 'AER_F_PRO', 'AER_F_LIP', 'AER_F_SS']
    # lat_aer = variables_info[panel_names[0]]['lat']
    # lon_aer = variables_info[panel_names[0]]['lon']
    # fig_titles = [['PCHO', r'$\bf{(a)}$'], ['DCAA', r'$\bf{(b)}$'], ['PL', r'$\bf{(c)}$'], ['SS', r'$\bf{(d)}$']],
    # panel_var_trend, panel_var_pval, panel_unit = utils.alloc_metadata(panel_names, variables_info)
    # panel_std = []
    # for i in panel_names:
    #     std = variables_info[i]['data_season_reg'].std(dim='time')
    #     print(i, 'std max', std.max().values)
    #     panel_std.append(std)
    # plots.plot_4_pannel_trend(panel_std,
    #                           seaice_aer,
    #                           panel_var_pval,
    #                           lat_aer,
    #                           lon_aer,
    #                           [0.005, 0.02, 0.9, 18],
    #                           panel_unit,
    #                           fig_titles,
    #                           'Emission_flux_std')

    #
    # print('Plot all PMOA emission trend and emission per SIC')
    # panel_names = ['AER_F_POL', 'AER_F_PRO', 'AER_F_LIP']
    # panel_var_trend, panel_var_pval, panel_unit = utils.alloc_metadata(panel_names,
    #                                                                     variables_info_seaice)
    #
    # panel_var_perc_incr = []
    # for vv in panel_names:
    #     panel_var_perc_incr.append(percent_icrease(variables_info_yr, vv, decade))
    #
    # panel_var_trend_tr, panel_var_pval_tr, panel_unit_tr = utils.alloc_metadata(panel_names,
    #                                                                               variables_info_yr,
    #                                                                               trends=True)
    #
    # fig_titles = [[['PCHO$_{aer}$ emission trend', r'$\bf{(a)}$'], ['PCHO$_{aer}$  per unit SIC', r'$\bf{(b)}$'],
    #                ['DCAA$_{aer}$ emission trend', r'$\bf{(c)}$'], ['DCAA$_{aer}$  per unit SIC', r'$\bf{(d)}$'],
    #                ['PL$_{aer}$  emission trend', r'$\bf{(e)}$'], ['PL$_{aer}$  per unit of SIC', r'$\bf{(f)}$']]]
    #
    # panel_var_trend_new = [panel_var_trend_tr[0], fill_with_nan(panel_var_trend[0]),
    #                        panel_var_trend_tr[1], fill_with_nan(panel_var_trend[1]),
    #                        panel_var_trend_tr[2], fill_with_nan(panel_var_trend[2])]
    #
    # panel_var_perc_incr_new = [panel_var_perc_incr[0], fill_with_nan(panel_var_trend[0]),
    #                            panel_var_perc_incr[1], fill_with_nan(panel_var_trend[1]),
    #                            panel_var_perc_incr[2], fill_with_nan(panel_var_trend[2])]
    #
    # panel_unit_new = [panel_unit_tr[0], panel_unit[0],
    #                   panel_unit_tr[1], panel_unit[1],
    #                   panel_unit_tr[2], panel_unit[2]]
    # panel_var_pval_new = [panel_var_pval_tr[0], panel_var_pval[0],
    #                       panel_var_pval_tr[1], panel_var_pval[1],
    #                       panel_var_pval_tr[2], panel_var_pval[2]]
    # panel_var_trend_new_signif = []
    # for i, pval in enumerate(panel_var_pval_new):
    #     panel_var_trend_new_signif.append(np.ma.masked_where(np.isnan(pval), panel_var_trend_new[i]))
    #
    # plots.plot_6_2_pannel_trend(panel_var_perc_incr_new,  # panel_var_trend_new,
    #                             seaice,
    #                             panel_var_pval_new,
    #                             lat_aer,
    #                             lon_aer,
    #                             [7, 0.003, 7, 0.03, 7, 0.8],  # [0.00015, 0.003, 0.0007, 0.03, 0.03, 0.8],
    #                             panel_unit_new,
    #                             fig_titles,
    #                             f'{season}_all_Emission_flux_trends_and_per_ice',
    #                             not_aerosol=False, )

    #######################
    print('Plot 10 m wind and SSt trend')
    panel_names = ['AER_U10', 'AER_SST']
    fig_titles = [['Wind10', r'$\bf{(a)}$'], ['Surface temperature', r'$\bf{(b)}$']],
    panel_var_trend, panel_var_pval, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr,
                                                                                  trends=True)
    panel_var_trend_signif = []
    for i, pval in enumerate(panel_var_pval):
        panel_var_trend_signif.append(np.ma.masked_where(np.isnan(pval), panel_var_trend[i]))
    #
    plots.plot_2_pannel_trend(panel_var_trend,  # panel_var_trend_signif
                              seaice,
                              panel_var_pval,
                              lat_aer,
                              lon_aer,
                              [0.05, 0.1],
                              panel_unit,
                              fig_titles,
                              'wind_sst_Arctic',
                              not_aerosol=False,
                              #                          percent_increase=True,
                              )
    #######################
    print('')
    print('FINALL six panel plot emission and emiss/SIC')
    panel_names = ['AER_F_LIP', 'AER_F_SS']
    # panel_var_trend, panel_var_pval, panel_unit = utils.alloc_metadata(panel_names, variables_info_seaice)
    panel_names = ['AER_SIC', 'AER_F_LIP', 'AER_F_SS']
    panel_var_trend_tr, panel_var_pval_tr, panel_unit_tr = utils.alloc_metadata(panel_names,
                                                                                              variables_info_yr,
                                                                                              trends=True)

    fig_titles = [[['SIC', r'$\bf{(a)}$'], ['SIC trend', r'$\bf{(b)}$'],
                   ['PL$_{aer}$ emission trend', r'$\bf{(c)}$'], ['PL$_{aer}$  per unit SIC', r'$\bf{(d)}$'],
                   ['SS emission trend', r'$\bf{(e)}$'], ['SS per unit of SIC', r'$\bf{(f)}$']]]

    panel_var_trend_new = [seaice_aer[1], panel_var_trend_tr[0],
                           panel_var_trend_tr[1], fill_with_nan(panel_var_trend[0]),
                           panel_var_trend_tr[2], fill_with_nan(panel_var_trend[1])]
    panel_unit_new = ['%', panel_unit_tr[0], panel_unit_tr[1], panel_unit[0], panel_unit_tr[2], panel_unit[1]]
    panel_var_pval_new = [0, panel_var_pval_tr[0], panel_var_pval_tr[1], panel_var_pval[0], panel_var_pval_tr[2],
                          panel_var_pval[1]]

    panel_var_trend_new_signif = [seaice_aer[1]]
    for i, pval in enumerate(panel_var_pval_new[1:]):
        panel_var_trend_new_signif.append(np.ma.masked_where(np.isnan(pval), panel_var_trend_new[i + 1]))

    vlims = utils.find_max_lim(panel_var_trend_new)

    #### UNCOMENt
    plots.plot_6_2_pannel_trend(panel_var_trend_new,  # panel_var_trend_new_signif
                                seaice,
                                panel_var_pval_new,
                                lat_aer,
                                lon_aer,
                                [100, 1.8, 0.03, 0.8, 0.5, 10],  # [100,  1.5,  0.02, 0.8, 0.3, 10],
                                panel_unit_new,
                                fig_titles,
                                f'{season}_SeaIce_Emission_flux_trends_and_per_ice',
                                not_aerosol=False,
                                seaice_conc=True)


def plot_trend_aer_concentration(variables_info_yr, seaice, season):
    panel_names = ['AER_POL', 'AER_PRO', 'AER_LIP', 'AER_SS']
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']
    fig_titles = [[['PCHO$_{aer}$', r'$\bf{(a)}$'],
                   ['DCAA$_{aer}$', r'$\bf{(c)}$'],
                   ['PL$_{aer}$', r'$\bf{(b)}$'],
                   ['SS', r'$\bf{(d)}$']]]
    vlims = [0.005, 0.03, 0.9, 4]
    panel_var_trend, panel_var_pval, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr,
                                                                                  trends=True)

    plots.plot_4_pannel_trend(panel_var_trend,
                              seaice,
                              panel_var_pval,
                              lat,
                              lon,
                              vlims,
                              panel_unit,
                              fig_titles,
                              f'{season}_concentration_trends',
                              not_aerosol=False,
                              percent_increase=False,
                              )