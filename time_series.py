import pickle

import global_vars
import plots, utils
import matplotlib.pyplot as plt

from utils import regions

if __name__ == '__main__':
    season = global_vars.season_to_analise

    with open(f"TrendsDict_{season}_orig_data.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    unit = '(Tg ${month^{-1}}$)'
    fig, axes = plt.subplots(3, 1,
                             figsize=(8, 6), )
    axs = axes.flatten()
    colors = ['darkblue', 'darkred']

    seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice')
    seaice_lin = variables_info_yr['AER_SIC_area_px']
    var_type = ['Biom_tot', 'biom', 'Biomolecule concentration']
    var_type = ['AER_tot', 'aer_conc', 'Aerosol concentration']
    var_type = ['AER_F_tot', 'aer_flux', 'Aerosol emission flux']
    # var_type = ['AER_F_tot_anom', 'aer_flux_anom', 'Aerosol emission flux anomaly']

    flux = variables_info_yr[var_type[0]]  # AER_LIP #AER_F_tot_yr
    conc = variables_info_yr['AER_tot']  # AER_LIP #AER_F_tot_yr
    region = {'Arctic':{'lims':[[4.3, 7.5], [0.1, 0.1]]},
            'Kara Sea':{'lims':[[0., 0.6], [0.1, 0.1]]},
            'Barents Sea':{'lims':[[0, 0.23], [0.1, 0.1]]}}

    region = {'Arctic':{'lims':[[10, 7.5], [0.1, 0.1]]},
            'Kara Sea':{'lims':[[0.7, 0.6], [0.1, 0.1]]},
            'Barents Sea':{'lims':[[0.25, 0.23], [0.1, 0.1]]}}


    for idx, reg in enumerate(list(region.keys())):
        plots.autocorrelation([seaice_lin[reg], flux[reg]],
                        [reg, r'$\bf{(a)}$'],
                         var_type)
        plots.create_histogram([seaice_lin[reg], flux[reg]],
                        [reg, r'$\bf{(a)}$'],
                         var_type)
        leg = plots.plot_fit_trends(axs[idx],
                                    [seaice_lin[reg], flux[reg]],
                                    [reg, r'$\bf{(a)}$'],
                                    ['Sea Ice Area \n (millions of km$^{2}$)', f'PMOA emission mass \n flux {unit}'],
                                    # ['Sea ice \n Concentration ($million km^{2})$', 'Total biomolecule \n concentration'],
                                    # [[6, 10.5], [0.1, 0.1]],
                                    region[reg]['lims'],
                                    colors,
                                    ['Sea ice', 'PMOA'],
                                    # ['Sea Ice', 'Biomolecules'],
                                    f'{season}_Ocean_Flux_PL',
                                    var_type,
                                    echam_data=True,
                                    seaice=True,
                                    multipanel=True)
        if idx == 0:
            plt.legend(handles=[leg[0], leg[2][0], leg[3][0], leg[1], leg[2][1], leg[3][1]], ncol=2,
                       bbox_to_anchor=(0.3, 1.), loc='lower left', fontsize=8)
        else:
            plt.legend(handles=[leg[2][0], leg[3][0], leg[2][1], leg[3][1]], ncol=2,  # ncol=len(axs[0].lines),
                       bbox_to_anchor=(0.3, 1.), loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'./plots/{season}_multipanel_time_series.png', dpi=300)

    region = utils.regions()
    for reg, idx in region.items():
        decades = ['1990-2004', '2005-2019']
        for idx, dec in enumerate(decades):
            #weights = utils.get_weights(conc[reg][dec]['data_aver_reg'])

            with open("Region_means.txt", "a") as f:
                print(reg, dec,
                      'mean SIC',
                      seaice_lin[reg][dec]['data_aver_reg'].mean(skipna=True).values,
                      '\n',
                      file=f)
                print(reg, dec,
                      'mean PMOA surface concentration',
                      conc[reg][dec]['data_aver_reg'].mean(skipna=True).values,
                      '\n',
                      file=f)
                print(reg, dec,
                      'mean PMOA emission flux',
                      flux[reg][dec]['data_aver_reg'].mean(skipna=True).values,
                      file=f)
                print('\n\n',
                      file=f)


