import pickle

import global_vars
import plots, utils
import matplotlib.pyplot as plt

if __name__ == '__main__':
    season = global_vars.season_to_analise

    with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    with open(f"TrendsDict_per_ice_{season}.pkl", "rb") as myFile:
        variables_info_seaice = pickle.load(myFile)



    unit = '(Tg ${month^{-1}}$)'
    fig, axes = plt.subplots(3, 1,
                             figsize=(8, 6), )
    axs = axes.flatten()
    colors = ['darkblue', 'darkred']

    seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice')
    seaice_lin = variables_info_yr['AER_SIC_area_px']
    biomol = variables_info_yr['AER_F_tot_m']  # AER_LIP #AER_F_tot_yr

    region = {'Arctic':{'lims':[[4.3, 7.5], [0.1, 0.1]]},
            'Kara Sea':{'lims':[[0., 0.6], [0.1, 0.1]]},
            'Barents Sea':{'lims':[[0, 0.23], [0.1, 0.1]]}}

    for reg, idx in region.items():
        decades = ['1990-2004', '2005-2019']
        for idx, dec in enumerate(decades):
            print('mean Sea ice', seaice_lin[reg][dec]['data_aver_reg'].mean(skipna=True).values)
            print('mean emission flux', biomol[reg][dec]['data_aver_reg'].mean(skipna=True).values)
            print('')
            print('min Sea ice', seaice_lin[reg][dec]['data_aver_reg'].min(skipna=True).values)
            print('min emission flux', biomol[reg][dec]['data_aver_reg'].min(skipna=True).values)
            print('')
            print('max Sea ice', seaice_lin[reg][dec]['data_aver_reg'].max(skipna=True).values)
            print('max emission flux', biomol[reg][dec]['data_aver_reg'].max(skipna=True).values)
        decade = '1990-2019'
        leg = plots.plot_fit_trends(axs[idx],
                                    [seaice_lin[reg], biomol[reg]],
                                    [reg, r'$\bf{(a)}$'],
                                    ['Sea Ice Area \n (millions of km$^{2}$)', f'PMOA emission mass \n flux {unit}'],
                                    # ['Sea ice \n Concentration ($million km^{2})$', 'Total biomolecule \n concentration'],
                                    # [[6, 10.5], [0.1, 0.1]],
                                    region[reg]['lims'],
                                    colors,
                                    ['Sea ice', 'PMOA'],
                                    # ['Sea Ice', 'Biomolecules'],
                                    f'{season}_Ocean_Flux_PL',
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

    plt.savefig(f'./plots/{season}_multipanel_time_series.png', dpi=200)