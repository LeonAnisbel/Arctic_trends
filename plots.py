import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib import ticker as mticker
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
import pymannkendall as mk
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib import ticker as mticker
import global_vars
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.0, maxval=0.5, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap.name}_truncated", new_colors)

def format_func(value, tick_number):
    """ This function will create the year labels considering that 1990 is year 0, it returns 1
    :returns value+1"""
    N = int(value + 1990)
    return N

def plot_fit(ax, t_ax, p_fit, eq, color, a):
    return ax.plot(t_ax, p_fit, color, linestyle='dashed', label=eq, linewidth=1, alpha=a)

def create_histogram(C, title, var_type):
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
    plt.savefig(f'./plots/histogram_{var_type[1]}_{title[0]}.png')
    plt.close()

def autocorrelation(C, title, var_type):
    y = C[1]['1990-2019']['data_aver_reg'].values
    fig, ax = plt.subplots(1, 1,
                               figsize=(5, 4), )
    acf_values = acf(y, nlags=15, fft=False)
    sm.graphics.tsa.plot_acf(y, lags=15, ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid()
    r1 = acf_values
    # print(f"Lag-1 autocorrelation (r1): {r1}")
    plt.title('Autocorrelation Function')
    plt.savefig(f'./plots/autocorrelation_{var_type[1]}_{title[0]}.png')
    plt.close()

def plot_fit_trends(ax, C, title, axis_label, vm, colors, leg, fig_name, var_type,
                    echam_data=True, seaice=False, multipanel=False):
    """ Plots yearly data of sea ice area and emission anomalies and calculates the 30-year trend
    :returns subplot objects to later add the legend """
    if multipanel:
        pass
    else:
        fig, ax = plt.subplots(1, 1,
                               figsize=(8, 3), )

    C_ice, C_biom = C[0]['1990-2019']['data_sum_reg'], C[1]['1990-2019']['data_aver_reg']
    t_ax = C_ice.time.values

    ax2 = ax.twinx()
    new_line_color = colors[0]
    if seaice:
        new_line_color = 'lightblue'
        ax.fill_between(t_ax, 0, C_ice, alpha=0.2, color=new_line_color)
        p1, = ax.plot(t_ax, C_ice, new_line_color, label=leg[0], linewidth=1.)
    ax.scatter(t_ax, C_ice, s=7, c=new_line_color)

    p2, = ax2.plot(t_ax, C_biom, colors[1], label=leg[1], linewidth=1.)
    ax2.scatter(t_ax, C_biom, s=7, c=colors[1])
    ax2.axhline(y=0.0, c=colors[1], linewidth=0.5, alpha=0.2)

    decades = ['1990-1999', '2000-2009', '2010-2019']
    decades = ['1990-2004', '2005-2019']

    a = [0.5, 1]
    f1_list, f2_list = [], []
    # for idx, dec in enumerate(decades):
    if global_vars.season_to_analise == 'JAS' or global_vars.season_to_analise == 'AMJ':
        dec = '1990-2019'
        idx = 1
        t_ax = C[0][dec]['data_sum_reg'].time.values

        C_ice = C[0][dec]['data_sum_reg'].values
        model_sic = mk.original_test(C_ice)

        C_emi = C[1][dec]['data_aver_reg'].values
        model_emi = mk.original_test(C_emi)

        sl = [model_sic.slope, model_emi.slope]
        itc = [model_sic.intercept, model_emi.intercept]
        significance = [model_sic.h, model_emi.h]
        trend = [model_sic.trend, model_emi.trend]

        print('SIC TAU',C_ice, model_sic.Tau, '\n',min(C_ice), max(C_ice), '\n')
        print('Emission TAU', C_emi, model_emi.Tau, '\n',min(C_emi), max(C_emi), '\n\n')

        p_fit = [p * sl[0] + itc[0] for p in np.arange(len(t_ax))]
        eq = f'{sl[0]:.1e}x + {itc[0]:.1e}'
        if significance[0]:
            f1, = plot_fit(ax, t_ax, p_fit, eq, colors[0], a[idx])
            f1_list.append(f1)
        else:
            f1_list.append(p1)

        p_fit = [p * sl[1] + itc[1] for p in np.arange(len(t_ax))]
        eq = f'{sl[1]:.1e}x + {itc[1]:.1e}'
        if significance[1]:
            f2, = plot_fit(ax2, t_ax, p_fit, eq, colors[1], a[idx])
            f2_list.append(f2)
        else:
            f2_list.append(p2)

    ax.set_ylabel(axis_label[0], fontsize=8)
    ax.tick_params(axis='x')

    ax.yaxis.get_label().set_fontsize(8)
    ax2.set_ylabel(axis_label[1], fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8, color=colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.yaxis.label.set_color(colors[1])

    if title[0] == 'Arctic':
        tt = title[0] + '\n \n \n'
    else:
        tt = '\n '+ title[0] + '\n \n '

    ax.set_title(tt, loc='center', fontsize=10)
    ax.set_title(title[1], loc='left', fontsize=10)


    ax.set_ylim(vm[0][0], None)
    ax2.set_ylim(vm[1][0], vm[1][1])

    #ax.grid(linestyle='--', linewidth=0.4)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8, color=colors[0])
    ax.tick_params(axis='y', colors=colors[0])
    ax.yaxis.label.set_color(colors[0])

    ax2.ticklabel_format(axis="y", useMathText=True, useLocale=True)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    if echam_data:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    if multipanel:
        pass
    else:
        plt.savefig(f'./plots/{fig_name}.png', dpi=200)

    return [p1, p2, f1_list, f2_list]


def add_ice_colorbar(fig, ic, ic2):
    cbar_ax = fig.add_axes([0.37, -0.05, 0.25, 0.01])  # (left, bottom, width, height)
    ic_bar = fig.colorbar(ic, extendfrac='auto', shrink=0.005,
                          cax=cbar_ax, orientation='horizontal', )
    ic_bar.set_label('Sea ice concentration (%)', fontsize='12')
    ic_bar.ax.tick_params(labelsize=12)

    handles2, _ = ic2.legend_elements()
    plt.legend(handles2,
               ["sic 10%"],
               # loc='lower right',
               bbox_to_anchor=(0.75, 6))  # x,y


def plot_trend(subfig, trend, ice, pval, lat, lon, titles, vlim, unit, cm, vlim0,
               not_aerosol=True, percent_increase=False, seaice_conc=False, burden=False):
    """ Plots each map of Arctic trends with statistically significant grid cells as hatched areas
    :returns  """
    ax = subfig.subplots(nrows=1,
                         ncols=1,
                         sharex=True,
                         subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
    ax.set_extent([-180, 180, 60, 90],
                  ccrs.PlateCarree())
    if titles[0] == 'SIC' or titles[0] == 'SIC trend' and global_vars.lat_arctic_lim==66:
        ax.set_title(titles[0], loc='center', fontsize=12)
    else:
        ax.set_title(titles[0], loc='right', fontsize=12)

    ax.set_title(titles[1], loc='left', fontsize=12)

    cmap = plt.get_cmap(cm, 15)
    if titles[0][-16:] == ' per unit of SIC':
        vlim = 0
        cmap_original = plt.cm.coolwarm
        cmap = truncate_colormap(cmap_original, 0.0, 0.5)

    cb = ax.pcolormesh(lon,
                       lat,
                       trend,
                       vmin=vlim0,
                       vmax=vlim,
                       cmap=cmap,
                       transform=ccrs.PlateCarree())

    trend_signif = np.ma.masked_where(np.isnan(pval), trend)

    if percent_increase:
        pass
    else:
        plt.pcolor(lon,
                   lat,
                   trend_signif,
                   linewidth=0.001,
                   hatch='///', alpha=0.,
                   transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land',
                                                '110m', edgecolor='face',
                                                linewidth=0.05,
                                                facecolor='white'))

    cbar = plt.colorbar(cb,
                        extend='both',
                        orientation='horizontal',
                        fraction=0.05,
                        # format='%.0e',
                        pad=0.07)
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    cbar.set_label(label=unit,
                   fontsize=12, )
    # weight='bold')

    # # compute circle
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.coastlines(color='darkgray')
    gl = ax.gridlines(draw_labels=True,
                        x_inline=False,   # force labels to sit outside the map boundary
                        y_inline=False,)
    gl.bottom_labels = False
    gl.left_labels = False
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([65, 75, 85])
    gl.yformatter = LATITUDE_FORMATTER
    gl.xpadding = 8  # moves the lon labels (top & bottom) outward
    gl.ypadding = 8

    if burden:
        cbar.remove()  # later, remove it
        return cb

    if not_aerosol:
        orig_cmap = plt.get_cmap('Greys_r')
        colors = orig_cmap(np.linspace(0.1, 1, 4))
        cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

        ic = ax.contourf(ice[0].lon, ice[0].lat, ice[0], np.arange(10, 110, 30), cmap=cmap,
                         transform=ccrs.PlateCarree())

        ic2 = ax.contour(ice[1].lon, ice[1].lat,
                         ice[1], levels=[10],
                         linestyles=('solid',), colors='green', linewidths=1.,
                         transform=ccrs.PlateCarree())

        # plt.savefig(fig_title, dpi=300)
        # plt.close()

        return [ic, ic2]
    else:
        return




def iterate_subfig(fig, subfigs, fig_name, trend_vars, ice_var,
                   pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=True, percent_increase=False, seaice_conc=False):
    """ Organize and Plots calls function to plot each map of Arctic trends
    :returns  """
    if seaice_conc:
        cm, vlim0 = 'Blues_r', 0
        idx = 0
        print('fig1', idx, title_vars[0][idx])
        ic = plot_trend(subfigs[0],
                        trend_vars[idx],
                        ice_var,
                        pval_vars[idx],
                        lat,
                        lon,
                        title_vars[0][idx],
                        vlim_vars[idx],
                        unit_vars[idx],
                        cm, vlim0,
                        percent_increase=True,
                        not_aerosol=not_aerosol,
                        seaice_conc=seaice_conc)
        for idx, subf in enumerate(subfigs[1:]):
            idx = idx + 1
            cm, vlim0 = 'coolwarm', -vlim_vars[idx]
            print('fig2-', idx, title_vars[0])
            ic = plot_trend(subf,
                            trend_vars[idx],
                            ice_var,
                            pval_vars[idx],
                            lat,
                            lon,
                            title_vars[0][idx],
                            vlim_vars[idx],
                            unit_vars[idx],
                            cm, vlim0,
                            percent_increase=percent_increase,
                            not_aerosol=not_aerosol,
                            seaice_conc=seaice_conc)
    else:

        for idx, subf in enumerate(subfigs):
            if not_aerosol:
                if title_vars[0][idx][0] == 'SIC':
                    not_aerosol_new = False
                else:
                    not_aerosol_new = not_aerosol
            else:
                not_aerosol_new = not_aerosol

            cm, vlim0 = 'coolwarm', -vlim_vars[idx]
            ic_bar = plot_trend(subf,
                                trend_vars[idx],
                                ice_var,
                                pval_vars[idx],
                                lat,
                                lon,
                                title_vars[0][idx],
                                vlim_vars[idx],
                                unit_vars[idx],
                                cm, vlim0,
                                percent_increase=percent_increase,
                                not_aerosol=not_aerosol_new,
                                seaice_conc=seaice_conc)

            if not_aerosol_new:
                ic = ic_bar

    if not_aerosol:
        add_ice_colorbar(fig, ic[0], ic[1])
    plt.savefig(f'./plots/{fig_name}', dpi=200, bbox_inches="tight")

    #     fig.tight_layout()


def plot_6_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True, percent_increase=False):
    fig = plt.figure(constrained_layout=True, figsize=(10, 7))

    (subfig1, subfig2, subfig3), (subfig4, subfig5, subfig6) = fig.subfigures(nrows=2, ncols=3)
    subfigs = [subfig1, subfig2, subfig3, subfig4, subfig5, subfig6]
    fig_name = f'Six_panel_ocean_omf{fig_name}_Arctic_trends.png'
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=not_aerosol, percent_increase=percent_increase)


def plot_2_panel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False):
    """ Creates a plot of the 2-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True, figsize=(7, 4))

    (subfig1, subfig2) = fig.subfigures(nrows=1, ncols=2)
    subfigs = [subfig1, subfig2]

    fig_name = f'two_panel_{fig_name}_trends.png'
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase)


def plot_3_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True):
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))

    (subfig1, subfig2, subfig3) = fig.subfigures(nrows=1, ncols=3)
    subfigs = [subfig1, subfig2, subfig3]

    fig_name = f'three_panel_ocean{fig_name}_vars_trends.png'
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=not_aerosol)


def plot_4_panel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False, seaice_conc=False):
    """ Creates a plot of the 4-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2, ncols=2)

    fig_name = f'four_panel_{fig_name}.png'
    print(title_vars[0][0])

    subfigs = [subfig1, subfig2, subfig3, subfig4]
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars,
                   unit_vars, title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase, seaice_conc=seaice_conc)


def plot_4_panel_trend_burden(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False, seaice_conc=False):
    """ Creates a plot of the 4-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2, ncols=2)

    fig_name = f'four_panel_{fig_name}.png'
    print(title_vars[0][0])

    subfigs = [subfig1, subfig2, subfig3, subfig4]
    fig_name = f'four_panel_{fig_name}.png'
    print(title_vars[0][0])

    for idx  in range(len(trend_vars)):
        cm, vlim0 = 'coolwarm', -vlim_vars[idx]
        print('fig2-', idx, title_vars[0])
        cb = plot_trend(subfigs[idx],
                        trend_vars[idx],
                        seaice,
                        pval_vars[idx],
                        lat,
                        lon,
                        title_vars[0][idx],
                        vlim_vars[idx],
                        unit_vars[idx],
                        cm, vlim0,
                        percent_increase=percent_increase,
                        not_aerosol=not_aerosol,
                        seaice_conc=seaice_conc,
                        burden=True)

    cbar_ax = fig.add_axes([0.05, -0.05, 0.9, 0.03])  # (left, bottom, width, height)

    cbar = plt.colorbar(cb,
                        cax=cbar_ax,
                        extend='both',
                        orientation='horizontal',
                        fraction=0.05,
                        # format='%.0e',
                        pad=0.12)
    # cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    cbar.set_label(label='% yr$^{-1}$',
                   fontsize=14, )
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(f'./plots/{fig_name}', dpi=200, bbox_inches="tight")


def plot_6_2_panel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                          not_aerosol=True, percent_increase=False, seaice_conc=False):
    """ Creates a plot of the 6-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True, figsize=(7, 10))

    (subfig1, subfig2), (subfig3, subfig4), (subfig5, subfig6) = fig.subfigures(nrows=3, ncols=2)
    subfigs = [subfig1, subfig2, subfig3, subfig4, subfig5, subfig6]
    fig_name = f'Six_panel_ocean_omf{fig_name}_Arctic_trends.png'
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=not_aerosol, percent_increase=percent_increase, seaice_conc=seaice_conc)

    return None
