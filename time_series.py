import os
import pickle

import global_vars
import plots, utils
import matplotlib.pyplot as plt

from global_vars import thesis_plot
from utils import regions

if __name__ == '__main__':
    season = global_vars.season_to_analise

    with open(f"TrendsDict_{season}_orig_data.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    unit = '(Tg season${^{-1}}$)'
    if global_vars.thesis_plot:
        fig, axes = plt.subplots(3, 1,
                                 figsize=(6, 8), )
        axs = axes.flatten()

    else:

        plot_heights = [2, 1, 2, 1, 2,1]
        height_ratios = [
            2.2, 0.01,
            1.3, 0.1,
            2.2, 0.01,
            1.3, 0.1,
            2.2, 0.01,
            1.3,
        ]
        fig, axes = plt.subplots(len(height_ratios), 1,
                                 figsize=(10, 16),
                                 gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.1},
                                 constrained_layout = True,
        )
        ax = axes.flatten()
        for i, a in enumerate(ax):
            if i == 1 or i == 3 or i == 5 or i == 7 or i == 9:
                a.axis('off')
        axs = [ax[0], ax[4], ax[8]]
        axs1 = [ax[2], ax[6], ax[10]]
    colors = ['darkblue', 'darkred', 'gray']

    seaice = utils.get_seaice_vals(variables_info_yr, 'Sea_ice')
    var = 'AER_SST' #'AER_SIC_area_px'
    sst = variables_info_yr['AER_SST']
    omf = variables_info_yr['OMF_tot']
    u10 = variables_info_yr['AER_U10']

    seaice_lin = variables_info_yr['AER_SIC_area_px']
    var_type = ['Biom_tot', 'biom', 'Biomolecule concentration']
    var_type = ['AER_tot', 'aer_conc', 'Aerosol concentration']
    var_type = ['AER_F_tot', 'aer_flux', 'Aerosol emission flux']
    var_type = ['AER_F_tot_anom', 'aer_flux_anom', 'Aerosol emission flux anomaly']

    flux = variables_info_yr[var_type[0]]  # AER_LIP #AER_F_tot_yr
    region = {'Arctic':{'lims':[[4.3, 7.5], [-0.7e-5, 0.7e-5]],
                        'label': r'$\bf{(a)}$'},
            'Beaufort Sea':{'lims':[[0.45, 0.6], [-0.7e-5, 0.7e-5]],
                        'label': r'$\bf{(b)}$'},
            'Barents Sea':{'lims':[[0, 0.23], [-2.5e-5, 2.5e-5]],
                        'label': r'$\bf{(c)}$'}}

    # region = {'Arctic':{'lims':[[10, 7.5], [0.1, 0.1]]},
    #         'Kara Sea':{'lims':[[0.7, 0.6], [0.1, 0.1]]},
    #         'Barents Sea':{'lims':[[0.25, 0.23], [0.1, 0.1]]}}


    for idx, reg in enumerate(list(region.keys())):
        plots.autocorrelation([seaice_lin[reg], flux[reg]],
                        [reg, r'$\bf{(a)}$'],
                         var_type)
        plots.create_histogram([seaice_lin[reg], flux[reg]],
                        [reg, r'$\bf{(a)}$'],
                         var_type)
        leg = plots.plot_fit_trends(axs[idx],
                            [seaice_lin[reg], flux[reg], sst[reg]],
                            [reg, region[reg]['label']],
                            [f'Sea Ice Area \n (10$^{6}$ km$^{2}$)',
                             f'PMOA emission \n anomalies {unit}', 'SST ($^{o}$C)'],
                            # ['Sea ice \n Concentration ($million km^{2})$', 'Total biomolecule \n concentration'],
                            # [[6, 10.5], [0.1, 0.1]],
                            region[reg]['lims'],
                            colors,
                            ['Sea ice', 'PMOA', 'SST'],
                            # ['Sea Ice', 'Biomolecules'],
                            f'{season}_Ocean_Flux_PL',
                            var_type,
                            echam_data=True,
                            seaice=True,
                            multipanel=True,
                            thesis_plot=True)
        if idx == 0:
            plt.legend(handles=[leg[0], leg[2][0], leg[1], leg[3][0]], ncol=2,
                       bbox_to_anchor=(0.1, 1.), loc='lower left', fontsize=15)
        else:
            plt.legend(handles=[leg[2][0],  leg[3][0]], ncol=2,  # ncol=len(axs[0].lines),
                       bbox_to_anchor=(0.1, 1.), loc='lower left', fontsize=15)

        if not thesis_plot:
            leg = plots.plot_fit_trends_sst(axs1[idx],
                                        sst[reg],
                                        [reg, region[reg]['label']],
                                        'SST ($^{o}$C)',
                                        'gray')

    # plt.tight_layout()
    plt.savefig(f'./plots/{season}_multipanel_time_series.png',
                dpi=300)
    plt.close()

    fig2, axes2 = plt.subplots(3, 1,
                             figsize=(7, 10), )
    axs2 = axes2.flatten()
    # region = {'Kara Sea':{'lims':[[4.3, 7.5], [-0.7e-5, 0.7e-5]],
    #                     'label': r'$\bf{(a)}$'},
    #         'Laptev Sea':{'lims':[[0., 0.6], [-0.7e-5, 0.7e-5]],
    #                     'label': r'$\bf{(b)}$'},
    #         'Chukchi Sea':{'lims':[[0, 0.23], [-2.5e-5, 2.5e-5]],
    #                     'label': r'$\bf{(c)}$'}}



    sst = variables_info_yr['AER_SST']
    omf = variables_info_yr['OMF_tot']
    u10 = variables_info_yr['AER_U10']
    seaice_lin = variables_info_yr['AER_SIC']
    var_type = ['AER_F_tot', 'aer_flux', 'Aerosol emission flux']

    names = ['SIC', 'SST', 'u10', 'OMF']
    for idx, reg in enumerate(list(utils.regions().keys())):
        variables = []
        variables.append(100-seaice_lin[reg]['1990-2019']['data_aver_reg'].values)
        for i in [sst, u10, omf, flux]:
            variables.append(i[reg]['1990-2019']['data_aver_reg'].values)
        sl, itc, significance, rval = plots.get_lin_regression(variables[-1], variables[:-1],)
        print('\n', reg, ' correlation with ', var_type[-1])
        for sdx, s in enumerate(significance):
            if s < 0.05:
                n = f'r = {rval[sdx]}'
                print(names[sdx], n)

    plt.tight_layout()
    plt.savefig(f'./plots/{season}_multipanel_time_series.png', dpi=300)

    flux = variables_info_yr['AER_F_tot_m']  # AER_LIP #AER_F_tot_yr
    fluxss = variables_info_yr['AER_F_SS_m']  # AER_LIP #AER_F_tot_yr
    biom = variables_info_yr['Biom_tot']  # AER_LIP #AER_F_tot_yr
    conc = variables_info_yr['AER_tot']  # AER_LIP #AER_F_tot_yr
    region = utils.regions()
    os.remove(f"Region_means_{season}.txt")
    for reg, idx in region.items():
        decades = ['1990-2004', '2005-2019']
        for idx, dec in enumerate(decades):
            with open(f"Region_means_{season}.txt", "a") as f:
                print(reg, dec,
                      'mean SIC',
                      seaice_lin[reg][dec]['data_sum_reg'].mean(skipna=True).values,
                      seaice_lin[reg][dec]['data_sum_reg'].std(skipna=True).values,
                      '\n',
                      file=f)
                print(reg, dec,
                      'Total PMOA emission flux',
                      flux[reg][dec]['data_sum_reg'].mean(skipna=True).values,
                      flux[reg][dec]['data_sum_reg'].std(skipna=True).values,
                      file=f)
                print(reg, dec,
                      'mean PMOA surface concentration',
                      conc[reg][dec]['data_aver_reg'].mean(skipna=True).values,
                      conc[reg][dec]['data_aver_reg'].std(skipna=True).values,
                      file=f)
                print(reg, dec,
                      'Total SS emission flux',
                      fluxss[reg][dec]['data_sum_reg'].mean(skipna=True).values,
                      fluxss[reg][dec]['data_sum_reg'].std(skipna=True).values,
                      file=f)
                print(reg, dec,
                      'mean biomolecule ocean concentration',
                      fluxss[reg][dec]['data_aver_reg'].mean(skipna=True).values,
                      fluxss[reg][dec]['data_aver_reg'].std(skipna=True).values,
                      file=f)
                print('\n\n',
                      file=f)


