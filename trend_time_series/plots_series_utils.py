import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pymannkendall as mk
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from utils_functions import global_vars
from scipy import stats


def format_func(value, tick_number):
    """ This function will create the year labels considering that 1990 is year 0, it returns 1
    :returns value+1"""
    N = int(value + 1990)
    return N

def get_lin_fit(sl, itc, C):
    """ Plots a line with the fitted function
    :returns None"""
    p_fit = [p * sl + itc for p in C]
    eq = f'{sl[0]:.1e}x + {itc[0]:.1e}'
    return p_fit, eq

def plot_fit(ax, t_ax, p_fit, eq, color, a, s):
    """ Plots a line with the fitted function
    :returns None"""
    return ax.plot(t_ax, p_fit, color, linestyle=s, label=eq, linewidth=1.7, alpha=a)

def get_lin_regression(C, variables):
    """ Computes linear regression coefficients between variables
    :returns slope, intercept, significance and r-value"""
    sl, itc, signf, rval = [], [], [], []
    for var in variables:
        model = stats.linregress(C, var)
        sl.append(model.slope)
        itc.append(model.intercept)
        signf.append(model.pvalue)
        rval.append(round(model.rvalue, 1))
    return sl, itc, signf, rval

def create_histogram(C, title, var_type):
    """ Creates histogram of data normal distribution
    :returns None"""
    y = C[1]['1990-2019']['data_aver_reg'].values
    fig, ax = plt.subplots(1, 1,
                               figsize=(5, 4), )
    mu = np.nanmean(y)
    sigma = np.nanstd(y)
    plt.hist(y, bins='auto', density=True, alpha=0.6)
    x = np.linspace(y.min(), y.max(), 100)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))

    # Overlay the normal distribution curve
    plt.plot(x, pdf, linewidth=2, label = 'Normal Dist.')
    plt.axvline(mu, color='k', linestyle='dashed', linewidth=2, label = 'Mean')
    plt.axvline(np.nanmedian(y), color='g', linestyle='dashed', linewidth=2, label = 'Median')
    plt.legend()
    plt.title('Histogram with Normal Distribution Fit \n '+ var_type[2] + ' in the '+ title[0])
    plt.xlabel('Value')
    plt.ylabel('Density')
    fig.tight_layout()
    plt.savefig(f'../plots/histogram_{var_type[1]}_{title[0]}.png')
    plt.close()

def autocorrelation(C, title, var_type):
    """ Creates autocorrelation plots with 95 % confidence in shaded blue
    :returns None"""
    y = C[1]['1990-2019']['data_aver_reg'].values
    fig, ax = plt.subplots(1,
                           1,
                               figsize=(5, 4), )
    acf_values = acf(y,
                     nlags=15,
                     fft=False)
    sm.graphics.tsa.plot_acf(y,
                             lags=15,
                             ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid()
    r1 = acf_values
    # print(f"Lag-1 autocorrelation (r1): {r1}")
    plt.title('Autocorrelation Function')
    plt.savefig(f'../plots/autocorrelation_{var_type[1]}_{title[0]}.png')
    plt.close()

def correlation_plots(ax, C, title, axis_label, vm, colors, leg, fig_name, var_type,
                    echam_data=True, seaice=False, multipanel=False):
    """ Plots with linear correlation between emission and emission drivers
    :returns subplot objects to later add the legend """
    font = 10
    if multipanel:
        pass
    else:
        fig, ax = plt.subplots(1, 1,
                               figsize=(8, 3), )
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.2))

    C_ice, C_sst, C_u10, C_omf, C_biom = (C[0]['1990-2019']['data_sum_reg'].values,
                                C[1]['1990-2019']['data_aver_reg'].values,
                                C[2]['1990-2019']['data_aver_reg'].values,
                                C[3]['1990-2019']['data_aver_reg'].values,
                                C[4]['1990-2019']['data_aver_reg'].values)

    colors = ['b', 'r', 'g']

    ax.scatter(C_biom, C_ice, c ='b', s=7)
    ax2.scatter(C_biom, C_sst, c ='r', s=7)
    ax3.scatter(C_biom, C_u10, c ='g', s=7)

    # p2, = ax2.plot(t_ax, C_biom, colors[1], label=leg[1], linewidth=1.)

    a = [0.5, 1]
    f1_list, f2_list, f3_list = [], [], []
    if global_vars.season_to_analise == 'JAS' or global_vars.season_to_analise == 'AMJ':
        dec = '1990-2019'
        idx = 1

        sl, itc, significance, rval = get_lin_regression(C_biom, [C_ice, C_sst, C_u10, C_omf])

        p_fit, eq = get_lin_fit(sl[0], itc[0], C_biom)
        if significance[0]<0.05:
            f1, = plot_fit(ax, C_biom, p_fit, eq, colors[0], a[idx], 'solid')
            f1_list.append(f1)
            n = f'r = {rval[0]}'
            print('sic', n)
            ax.text(-0.25, 0.8, n,
                    color=colors[0],
                    horizontalalignment='right',
                    transform=ax.transAxes,
                    fontsize='medium',
                    bbox={'facecolor': colors[0], 'boxstyle': 'round',
                          'pad': 0.1, 'alpha': 0.1})

        else:
            print('sic no significant correlation')

        p_fit, eq = get_lin_fit(sl[1], itc[1], C_biom)
        if significance[1]<0.05:
            f2, = plot_fit(ax2, C_biom, p_fit, eq, colors[1], a[idx], 'solid')
            f2_list.append(f2)
            n = f'r = {rval[1], 1}'
            print('sst', n)
            ax.text(-0.25, 0.6, n,
                    color=colors[1],
                    horizontalalignment='right',
                    transform=ax.transAxes,
                    fontsize='medium',
                    bbox={'facecolor': colors[1], 'boxstyle': 'round',
                          'pad': 0.1, 'alpha': 0.1})
        else:
            print('sst no significant correlation')

        p_fit, eq = get_lin_fit(sl[2], itc[2], C_biom)
        if significance[2]<0.05:
            f3, = plot_fit(ax3, C_biom, p_fit, eq, colors[2], a[idx], 'solid')
            f3_list.append(f3)
            n = f'r = {rval[2]}'
            print('u10', n)
            ax.text(-0.25, 0.4, n,
                    color=colors[2],
                    horizontalalignment='right',
                    transform=ax.transAxes,
                    fontsize='medium',
                    bbox={'facecolor': colors[2], 'boxstyle': 'round',
                          'pad': 0.1, 'alpha': 0.1})
        else:
            print('u10 no significant correlation')


        if significance[3]<0.05:
            n = f'r = {rval[3]}'
            print('omf', n)
        else:
            print('omf no significant correlation')


    ax.set_xlabel(axis_label[0], fontsize=font)
    ax.set_ylabel(axis_label[1], fontsize=font)
    ax2.set_ylabel(axis_label[2], fontsize=font)
    ax3.set_ylabel(axis_label[3], fontsize=font)

    ax.tick_params(axis='x')
    ax.grid(linestyle='--', linewidth=0.3)

    ax.yaxis.get_label().set_fontsize(font)
    ax.yaxis.set_tick_params(labelsize=font)
    ax2.yaxis.set_tick_params(labelsize=font, color=colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.yaxis.label.set_color(colors[1])


    ax3.yaxis.set_tick_params(labelsize=font, color=colors[2])
    ax3.tick_params(axis='y', colors=colors[2])
    ax3.yaxis.label.set_color(colors[2])


    if title[0] == 'Arctic':
        tt = title[0] + '\n \n \n'
    else:
        tt = '\n ' + title[0] + '\n \n \n'

    ax.set_title(tt, loc='center', fontsize=10)
    ax.set_title(title[1], loc='left', fontsize=10)

    # ax.set_ylim(vm[0][0], None)
    # ax2.set_ylim(vm[1][0], vm[1][1])

    # ax.grid(linestyle='--', linewidth=0.4)
    ax.xaxis.set_tick_params(labelsize=font)
    ax.yaxis.set_tick_params(labelsize=font, color=colors[0])
    ax.tick_params(axis='y', colors=colors[0])
    ax.yaxis.label.set_color(colors[0])

    ax2.ticklabel_format(axis="y", useMathText=True, useLocale=True)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    if multipanel:
        pass
    else:
        plt.savefig(f'../plots/{fig_name}.png', dpi=200)
    return [f1_list, f2_list, f3_list]


def plot_fit_trends_sst(ax, C, title, axis_label, colors):
    """ Plots yearly data of sea ice area and emission anomalies and calculates the 30-year trend
    :returns subplot objects to later add the legend """
    C_sst = C['1990-2019']['data_aver_reg']
    f = 15

    t_ax = C_sst.time.values

    new_line_color = colors[0]
    if global_vars.season_to_analise == 'JAS' or global_vars.season_to_analise == 'AMJ':
        dec = '1990-2019'
        model_sst = mk.original_test(C_sst)

        sl = model_sst.slope
        itc = model_sst.intercept
        significance = model_sst.h

        ax.plot(t_ax, C_sst, colors, label='SST', linewidth=1.7)
        ax.scatter(t_ax, C_sst, s=8, c=colors)

        p_fit = [p * sl + itc for p in np.arange(len(t_ax))]
        eq = f'{sl:.1e}x + {itc:.1e}'
        f3_list=[]
        if significance:
            f3, = plot_fit(ax, t_ax, p_fit, eq, colors, 1, 'dashed')
        ax.legend(handles=[f3], loc='lower right', fontsize=f)

    ax.set_ylabel(axis_label, fontsize=f)
    ax.tick_params(axis='x')

    ax.yaxis.get_label().set_fontsize(f)
    ax.yaxis.set_tick_params(labelsize=f)
    ax.set_ylabel(axis_label, fontsize=f)
    ax.yaxis.set_tick_params(labelsize=f, color=colors)
    ax.tick_params(axis='y', colors=colors)
    ax.yaxis.label.set_color(colors)

    ax.xaxis.set_tick_params(labelsize=f)
    ax.yaxis.set_tick_params(labelsize=f, color=colors)
    ax.tick_params(axis='y', colors=colors)
    ax.yaxis.label.set_color(colors)
    ax.grid(linestyle='--', linewidth=0.3)


    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))



def plot_fit_trends(ax, C, title, axis_label, vm, colors, leg, fig_name, var_type,
                    echam_data=True, seaice=False, multipanel=False,
                    thesis_plot=False):
    """ Plots yearly data of sea ice area and emission anomalies and calculates the 30-year trend
    :returns subplot objects to later add the legend """
    f = 15
    if multipanel:
        pass
    else:
        fig, ax = plt.subplots(1, 1,
                               figsize=(8, 3), )

    C_ice = C[0]['1990-2019']['data_sum_reg']
    C_emi = C[1]['1990-2019']['data_aver_reg']
    C_sst = C[2]['1990-2019']['data_aver_reg']

    print('MEAN VALS',
          np.mean(C[2]['1990-2004']['data_aver_reg']),
          np.mean(C[2]['2005-2019']['data_aver_reg']))

    t_ax = C_ice.time.values

    ax2 = ax.twinx()

    new_line_color = colors[0]
    if seaice:
        new_line_color = 'lightblue'
        ax.fill_between(t_ax, 0, C_ice, alpha=0.2, color=new_line_color)
        p1, = ax.plot(t_ax, C_ice, new_line_color, label=leg[0], linewidth=1.7)
    ax.scatter(t_ax, C_ice, s=10, c=new_line_color)

    p2, = ax2.plot(t_ax, C_emi, colors[1], label=leg[1], linewidth=1.7)
    ax2.scatter(t_ax, C_emi, s=10, c=colors[1])
    ax2.axhline(y=0.0, c=colors[1], linewidth=0.5, alpha=0.5)


    a = [0.5, 1]
    f1_list, f2_list, f3_list = [], [], []
    if global_vars.season_to_analise == 'JAS' or global_vars.season_to_analise == 'AMJ':
        dec = '1990-2019'
        idx = 1
        t_ax = C[0][dec]['data_sum_reg'].time.values

        model_sic = mk.original_test(C_ice)
        model_emi = mk.original_test(C_emi)
        model_sst = mk.original_test(C_sst)

        sl = [model_sic.slope, model_emi.slope, model_sst.slope]
        itc = [model_sic.intercept, model_emi.intercept, model_sst.intercept]
        significance = [model_sic.h, model_emi.h, model_sst.h]
        trend = [model_sic.trend, model_emi.trend, model_sst.trend]

        p_fit = [p * sl[0] + itc[0] for p in np.arange(len(t_ax))]
        eq = f'{sl[0]:.1e}x + {itc[0]:.1e}'
        if significance[0]:
            f1, = plot_fit(ax, t_ax, p_fit, eq, colors[0], a[idx], 'dashed')
            f1_list.append(f1)
        else:
            f1_list.append(p1)

        p_fit = [p * sl[1] + itc[1] for p in np.arange(len(t_ax))]
        eq = f'{sl[1]:.1e}x + {itc[1]:.1e}'
        if significance[1]:
            f2, = plot_fit(ax2, t_ax, p_fit, eq, colors[1], a[idx], 'dashed')
            f2_list.append(f2)
        else:
            f2_list.append(p2)

        if not thesis_plot:
            ax3 = ax.twinx()
            ax3.spines.right.set_position(("axes", 1.4))
            p3, = ax3.plot(t_ax, C_sst, colors[2], label=leg[2], linewidth=1.)
            ax3.scatter(t_ax, C_sst, s=7, c=colors[2])
            p_fit = [p * sl[2] + itc[2] for p in np.arange(len(t_ax))]
            eq = f'{sl[2]:.1e}x + {itc[2]:.1e}'
            if significance[2]:
                f3, = plot_fit(ax3, t_ax, p_fit, eq, colors[2], a[idx], 'dashed')
                f3_list.append(f3)
            else:
                f3_list.append(p3)
            ax_list = [ax2, ax3]
            ax.xaxis.set_tick_params(labelsize=f)
            if echam_data:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        else:
            ax_list = [ax2]
            ax.tick_params(axis='x', labelbottom=False)
        # print('\n', 'MIN and MAX SST', np.min(C_sst), C_sst.argmin(),
        #       np.max(C_sst), C_sst.argmax(), '\n')
        print('\n', 'MIN and MAX SIC', np.min(C_ice), C_ice.argmin(),
              C_ice,
              np.max(C_ice), C_ice.argmax(), '\n')


    ax.set_ylabel(axis_label[0], fontsize=f)
    ax.tick_params(axis='x')

    ax.yaxis.get_label().set_fontsize(f)
    ax.yaxis.set_tick_params(labelsize=f)


    for idx, a in enumerate(ax_list):
        a.set_ylabel(axis_label[idx+1], fontsize=f)
        a.yaxis.set_tick_params(labelsize=f, color=colors[idx+1])
        a.tick_params(axis='y', colors=colors[idx+1])
        a.yaxis.label.set_color(colors[idx+1])

    if title[0] == 'Arctic':
        tt = title[0] + '\n \n \n'
    else:
        tt = '\n '+ title[0] + '\n \n '

    ax.set_title(tt, loc='center', fontsize=f+2)
    ax.set_title(title[1], loc='left', fontsize=f+2)


    ax.set_ylim(vm[0][0], None)
    ax2.set_ylim(vm[1][0], vm[1][1])

    #ax.grid(linestyle='--', linewidth=0.4)
    ax.yaxis.set_tick_params(labelsize=f, color=colors[0])
    ax.tick_params(axis='y', colors=colors[0])
    ax.yaxis.label.set_color(colors[0])

    ax2.ticklabel_format(axis="y", useMathText=True, useLocale=True)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    ax.grid(linestyle='--', linewidth=0.3)

    if multipanel:
        pass
    else:
        plt.savefig(f'../plots/{fig_name}.png', dpi=200)

    return [p1, p2, f1_list, f2_list]
