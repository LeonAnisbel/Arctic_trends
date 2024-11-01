import pickle
import plots, utils
import numpy as np
import matplotlib.pyplot as plt

def fill_with_nan(panel_var_trend):
    panel_var_trend = np.ma.masked_where(panel_var_trend > 0., panel_var_trend)
    panel_var_tren = panel_var_trend.filled(np.nan)
    return panel_var_tren

season = 'JAS'
# season='AMJ'
# season = 'JJA'

with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
    variables_info_yr = pickle.load(myFile)
seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice')

seaice_lin = variables_info_yr['AER_SIC_area_px']
biomol = variables_info_yr['AER_F_tot_yr']  # AER_LIP #AER_F_tot_yr

region = ['Chukchi Sea', 'Greenland & Norwegian Sea', 'East-Siberian Sea', 'Kara Sea']
for reg in region:

    sst = variables_info_yr['AER_SST'][reg]
    wind = variables_info_yr['AER_U10'][reg]
    decades = ['1990-2004', '2005-2019']

    for idx, dec in enumerate(decades):
        print(reg, dec, 'mean SST', sst[dec]['data_aver_reg'].mean().values, )
        print(reg, dec, 'mean wind', wind[dec]['data_aver_reg'].mean().values, )

with open(f"TrendsDict_per_ice_{season}.pkl", "rb") as myFile:
    variables_info_seaice = pickle.load(myFile)

# variables_info = variables_info_seaice
unit = '(Tg ${yr^{-1}}$)'

fig, axes = plt.subplots(3, 1,
                         figsize=(8, 6), )
axs = axes.flatten()

decade = '1990-2019'
reg = 'Arctic'

colors = ['darkblue', 'darkred']
leg = plots.plot_fit_trends(axs[0],
                            [seaice_lin[reg], biomol[reg]],
                            [reg, r'$\bf{(a)}$'],
                            ['Sea Ice Area \n (millions of km$^{2}$)', f'PMOA emission mass \n flux {unit}'],
                            # ['Sea ice \n Concentration ($million km^{2})$', 'Total biomolecule \n concentration'],
                            # [[6, 10.5], [0.1, 0.1]],
                            [[4.3, 7.5], [0.1, 0.1]],
                            colors,
                            ['Sea ice', 'PMOA'],
                            # ['Sea Ice', 'Biomolecules'],
                            f'{season}_Ocean_Flux_PL',
                            echam_data=True,
                            seaice=True,
                            multipanel=True)

plt.legend(handles=[leg[0], leg[2][0], leg[2][1], leg[1], leg[3][0], leg[3][1]], ncol=2,
           bbox_to_anchor=(0.3, 1.), loc='lower left', fontsize=8)

reg = 'Kara Sea'
leg = plots.plot_fit_trends(axs[1],
                            [seaice_lin[reg], biomol[reg]],
                            [reg, r'$\bf{(b)}$'],
                            ['Sea Ice Area \n (millions of km$^{2}$)', f'PMOA emission mass \n flux {unit}'],
                            # ['Sea ice \n Concentration ($million km^{2})$', 'Total biomolecule \n concentration'],
                            # [[0, 0.7], [0.1, 0.1]],
                            [[0., 0.6], [0.1, 0.1]],
                            colors,
                            ['SIC Concentration', 'PMOA emission'],
                            # ['Sea Ice', 'Biomolecules'],
                            f'{season}_Ocean_Flux_PL',
                            echam_data=True,
                            seaice=True,
                            multipanel=True)
print(leg[2][0], leg[2][1], leg[3][0], leg[3][1])
plt.legend(handles=[leg[2][0], leg[2][1], leg[3][0], leg[3][1]], ncol=2,  # ncol=len(axs[0].lines),
           bbox_to_anchor=(0.3, 1.), loc='lower left', fontsize=8)

reg = 'Barents Sea'
leg = plots.plot_fit_trends(axs[2],
                            [seaice_lin[reg], biomol[reg]],
                            [reg, r'$\bf{(c)}$'],
                            ['Sea Ice Area \n (millions of km$^{2}$)', f'PMOA emission mass \n flux {unit}'],
                            # ['Sea ice \n Concentration ($million km^{2})$', 'Total biomolecule \n concentration'],
                            # [[0, 0.4], [0.1, 0.1]],
                            [[0, 0.23], [0.1, 0.1]],
                            colors,
                            ['SIC Concentration', 'PMOA emission'],
                            # ['Sea Ice', 'Biomolecules'],
                            f'{season}_Ocean_Flux_PL',
                            echam_data=True,
                            seaice=True,
                            multipanel=True)
plt.legend(handles=[leg[2][0], leg[2][1], leg[3][0], leg[3][1]], ncol=2,  # ncol=len(axs[0].lines),
           bbox_to_anchor=(0.3, 1.), loc='lower left', fontsize=8)
plt.tight_layout()

plt.savefig(f'./plots/{season}_multipanel_time_series.png', dpi=200)

print('Finish time series plot')

seaice_aer = utils.get_seaice_vals(variables_info_yr, 'AER_SIC')
panel_names = ['AER_F_POL', 'AER_F_PRO', 'AER_F_LIP', 'AER_F_SS']
# fig_titles = [['PCHO$_{aer}$', r'$\bf{(a)}$'], ['DCAA$_{aer}$', r'$\bf{(b)}$'], ['PL$_{aer}$', r'$\bf{(c)}$'],
#               ['SS$_{aer}$', r'$\bf{(d)}$']],
lat_aer = variables_info_yr[panel_names[0]]['lat']
lon_aer = variables_info_yr[panel_names[0]]['lon']
# panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info)
# panel_lim = [3, 3, 3, 3]
# percent_increase_yr, panel_unit = get_perc_increase(variables_info, panel_names)
# plots.plot_4_pannel_trend(percent_increase_yr,
#                           seaice_aer,
#                           panel_var_pval,
#                           lat_aer,
#                           lon_aer,
#                           panel_lim,
#                           panel_unit,
#                           fig_titles,
#                           'Aerosol_mass_flux_percent_trend',
#                           not_aerosol=True,
#                           percent_increase=True)

# panel_names = ['AER_F_POL', 'AER_F_PRO', 'AER_F_LIP', 'AER_F_SS']
# lat_aer = variables_info[panel_names[0]]['lat']
# lon_aer = variables_info[panel_names[0]]['lon']
# fig_titles = [['PCHO', r'$\bf{(a)}$'], ['DCAA', r'$\bf{(b)}$'], ['PL', r'$\bf{(c)}$'], ['SS', r'$\bf{(d)}$']],
# panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info)
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


print('Plot all PMOA emission trend and emission per SIC')
panel_names = ['AER_F_POL', 'AER_F_PRO', 'AER_F_LIP']
panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names,
                                                                              variables_info_seaice)
panel_var_trend_tr, panel_var_pval_tr, panel_lim_tr, panel_unit_tr = utils.alloc_metadata(panel_names,
                                                                                          variables_info_yr,
                                                                                          trends=True)

fig_titles = [[['PCHO$_{aer}$ emission trend', r'$\bf{(a)}$'], ['PCHO$_{aer}$  per unit SIC', r'$\bf{(b)}$'],
               ['DCAA$_{aer}$ emission trend', r'$\bf{(c)}$'], ['DCAA$_{aer}$  per unit SIC', r'$\bf{(d)}$'],
               ['PL$_{aer}$  emission trend', r'$\bf{(e)}$'], ['PL$_{aer}$  per unit of SIC', r'$\bf{(f)}$']]]


panel_var_trend_new = [panel_var_trend_tr[0], fill_with_nan(panel_var_trend[0]),
                       panel_var_trend_tr[1], fill_with_nan(panel_var_trend[1]),
                       panel_var_trend_tr[2], fill_with_nan(panel_var_trend[2])]
panel_unit_new = [panel_unit_tr[0], panel_unit[0],
                  panel_unit_tr[1], panel_unit[1],
                  panel_unit_tr[2], panel_unit[2]]
panel_var_pval_new = [panel_var_pval_tr[0], panel_var_pval[0],
                      panel_var_pval_tr[1], panel_var_pval[1],
                      panel_var_pval_tr[2], panel_var_pval[2]]
panel_var_trend_new_signif = []
for i, pval in enumerate(panel_var_pval_new):
    panel_var_trend_new_signif.append(np.ma.masked_where(np.isnan(pval), panel_var_trend_new[i]))

plots.plot_6_2_pannel_trend(panel_var_trend_new,  # panel_var_trend_new_signif
                            seaice,
                            panel_var_pval_new,
                            lat_aer,
                            lon_aer,
                            [0.00015, 0.003, 0.0007, 0.03, 0.03, 0.8],  # [100,  1.5,  0.02, 0.8, 0.3, 10],
                            panel_unit_new,
                            fig_titles,
                            f'{season}_all_Emission_flux_trends_and_per_ice',
                            not_aerosol=False,)

#######################
print('Plot 10 m wind and SSt trend')
panel_names = ['AER_U10', 'AER_SST']
fig_titles = [['Wind10', r'$\bf{(a)}$'], ['Surface temperature', r'$\bf{(b)}$']],
panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_yr,
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
panel_var_trend, panel_var_pval, panel_lim, panel_unit = utils.alloc_metadata(panel_names, variables_info_seaice)
panel_names = ['AER_SIC', 'AER_F_LIP', 'AER_F_SS']
panel_var_trend_tr, panel_var_pval_tr, panel_lim_tr, panel_unit_tr = utils.alloc_metadata(panel_names,
                                                                                          variables_info_yr,
                                                                                          trends=True)

fig_titles = [[['SIC', r'$\bf{(a)}$'], ['SIC trend', r'$\bf{(b)}$'],
               ['PL$_{aer}$ emission trend', r'$\bf{(c)}$'], ['PL$_{aer}$  per unit SIC', r'$\bf{(d)}$'],
               ['SS$_{aer}$  emission trend', r'$\bf{(e)}$'], ['SS$_{aer}$  per unit of SIC', r'$\bf{(f)}$']]]

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

print('Trends of OMF and biom')
####-------------------------------------------------------------------------------------------------------------------
print('Start now with OMF and biom')
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
                            percent_increase=False,
                            )

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
