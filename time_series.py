import pickle
import plots, utils
import matplotlib.pyplot as plt

if __name__ == '__main__':

    season = 'JAS'
    # season='AMJ'
    # season = 'JJA'

    with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)
    seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice')

    seaice_lin = variables_info_yr['AER_SIC_area_px']
    biomol = variables_info_yr['AER_F_tot_m']  # AER_LIP #AER_F_tot_yr

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
    unit = '(Tg ${month^{-1}}$)'

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

    plt.legend(handles=[leg[0], leg[2][0], leg[3][0], leg[1], leg[2][1], leg[3][1]], ncol=2,
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
    plt.legend(handles=[leg[2][0], leg[3][0], leg[2][1], leg[3][1]], ncol=2,  # ncol=len(axs[0].lines),
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
    plt.legend(handles=[leg[2][0], leg[3][0], leg[2][1], leg[3][1]], ncol=2,  # ncol=len(axs[0].lines),
               bbox_to_anchor=(0.3, 1.), loc='lower left', fontsize=8)
    plt.tight_layout()

    plt.savefig(f'./plots/{season}_multipanel_time_series.png', dpi=200)