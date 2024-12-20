import plots
import utils


def plot_trend(variables_info_seaice, variables_info_yr, seaice, season):
    print('Trends of OMF and biom')
    panel_names = ['PCHO', 'PL', 'Biom_tot', 'OMF_tot']
    lat = variables_info_seaice[panel_names[0]]['lat']
    lon = variables_info_seaice[panel_names[0]]['lon']
    fig_titles = [[['PCHO$_{sw}$', r'$\bf{(a)}$'],
                   ['PL$_{sw}$', r'$\bf{(b)}$'],
                   ['Total concentration$_{sw}$', r'$\bf{(c)}$'],
                   ['Total OMF', r'$\bf{(d)}$']]]

    # panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr, trends=True)
    # vlims = utils.find_max_lim(panel_var_trend)
    # plots.plot_4_pannel_trend(panel_var_trend,
    #                           seaice,
    #                           panel_var_pval,
    #                           lat,
    #                           lon,
    #                           [0.04, 0.008, 0.05, 0.25],#[0.05, 0.005, 0.05, 0.2],
    #                           panel_unit,
    #                           fig_titles,
    #                           f'{season}_Biom_OMF_trends_with_ice',
    #                           not_aerosol=True,
    #                           percent_increase=False)


    panel_names = ['PCHO', 'PL', 'Sea_ice', 'NPP']
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']
    fig_titles = [[['PCHO$_{sw}$', r'$\bf{(a)}$'],
                   ['PL$_{sw}$', r'$\bf{(b)}$'],
                   ['SIC', r'$\bf{(c)}$'],
                   ['NPP', r'$\bf{(d)}$']]]
    vlims = [0.04, 0.008, 3, 1.]
    panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr,
                                                                                  trends=True)

    plots.plot_4_pannel_trend(panel_var_trend,
                              seaice,
                              panel_var_pval,
                              lat,
                              lon,
                              vlims,
                              panel_unit,
                              fig_titles,
                              f'{season}_Biom_SIC_NPP_trends',
                              not_aerosol=True,
                              percent_increase=False,
                              )

    panel_names = ['PCHO', 'DCAA', 'PL', 'OMF_POL', 'OMF_PRO', 'OMF_LIP']
    fig_titles = [['PCHO$_{sw}$', r'$\bf{(a)}$'], ['DCAA$_{sw}$', r'$\bf{(b)}$'], ['PL$_{sw}$', r'$\bf{(c)}$'],
                  ['PCHO$_{aer}$ OMF', r'$\bf{(d)}$'], ['DCAA$_{aer}$ OMF', r'$\bf{(e)}$'],
                  ['PL$_{aer}$ OMF', r'$\bf{(f)}$']],
    panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr)
    vlims = utils.find_max_lim(panel_var_trend)

    # plots.plot_6_pannel_trend(panel_var_trend,
    #                         seaice,
    #                        panel_var_pval,
    ##                       lat,
    #                     lon,
    #                    [0.04, 0.01, 0.008, 0.002, 0.008, 0.2],  #[0.05, 0.01, 0.005, 0.002, 0.008, 0.2],
    #                   panel_unit,
    ##                  fig_titles,
    #                f'_{season}_',
    #               not_aerosol=True,
    #              percent_increase=False,
    #             )


    panel_names = ['PCHO', 'OMF_POL', 'PL', 'OMF_LIP', 'Sea_ice', 'NPP']
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']
    fig_titles = [[['PCHO$_{sw}$', r'$\bf{(a)}$'],
                   ['PCHO$_{aer}$ OMF', r'$\bf{(b)}$'],
                   ['PL$_{sw}$', r'$\bf{(c)}$'],
                   ['PL$_{aer}$ OMF', r'$\bf{(d)}$'],
                   ['SIC', r'$\bf{(e)}$'],
                   ['NPP', r'$\bf{(f)}$']]]
    vlims = [0.04, 0.002, 0.008, 0.25, 3, 1.]
    panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr,
                                                                                  trends=True)

    plots.plot_6_2_pannel_trend(panel_var_trend,
                                seaice,
                                panel_var_pval,
                                lat,
                                lon,
                                vlims,
                                panel_unit,
                                fig_titles,
                                f'{season}_Biom_OMF_SIC_NPP_trends',
                                not_aerosol=True,
                                percent_increase=False, )

    print('finish now with OMF and biom')

    ####-------------------------------------------------------------------------------------------------------------------
    print('Start now with other ocean variables')
    panel_names = ['Sea_ice', 'SST', 'NPP']
    fig_titles = [['Sea ice', r'$\bf{(a)}$'], ['SST', r'$\bf{(b)}$'], ['NPP', r'$\bf{(c)}$']],
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']
    panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr,
                                                                                  trends=True)
    plots.plot_3_pannel_trend(panel_var_trend,
                              seaice,
                              panel_var_pval,
                              lat,
                              lon,
                              [3, 0.2, 2],
                              panel_unit,
                              fig_titles,
                              f'{season}_no_icemask_',
                              not_aerosol=False)
