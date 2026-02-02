import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib import ticker as mticker
from utils_functions import global_vars
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap, minval=0.0, maxval=0.5, n=256):
    """ This function creates a colormap for sea ice
    :returns  """
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap.name}_truncated",
                                             new_colors)

def add_ice_colorbar(fig, ic, ic2, yloc):
    """ Adds the ice colorbar to the given figure.
    :returns  None"""
    cbar_ax = fig.add_axes([0.37, -0.05, 0.25, 0.01])  # (left, bottom, width, height)
    ic_bar = fig.colorbar(ic,
                          extendfrac='auto',
                          shrink=0.005,
                          cax=cbar_ax,
                          orientation='horizontal', )
    ic_bar.set_label('Sea ice concentration (%)',
                     fontsize='12')
    ic_bar.ax.tick_params(labelsize=12)

    handles2, _ = ic2.legend_elements()
    plt.legend(handles2,
               ["sic 10%"],
               # loc='lower right',
               bbox_to_anchor=(0.75, yloc))  # x,y


def plot_trend(subfig, trend, ice, pval, lat, lon, titles, vlim, unit, cm, vlim0,
               not_aerosol=True, percent_increase=False, seaice_conc=False,burden=False):
    """ Plots each map of Arctic trends with statistically significant grid cells as hatched areas
    :returns  """
    ax = subfig.subplots(nrows=1,
                         ncols=1,
                         sharex=True,
                         subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
    ax.set_extent([-180, 180, 60, 90],
                  ccrs.PlateCarree())
    if titles[0] == 'SIC' or titles[0] == 'SIC trend' and global_vars.lat_arctic_lim==66:
        ax.set_title(titles[0],
                     loc='center',
                     fontsize=12)
    else:
        ax.set_title(titles[0],
                     loc='right',
                     fontsize=12)

    ax.set_title(titles[1],
                 loc='left',
                 fontsize=12)
    if titles[0] == 'PCHO$_{aer}$ emission trend':
        ax.title.set_x(0.7)

    cmap = plt.get_cmap(cm,
                        15)
    if titles[0][-16:] == ' per unit of SIC':
        vlim = 0
        cmap_original = plt.cm.coolwarm
        cmap = truncate_colormap(cmap_original,
                                 0.0,
                                 0.5)

    cb = ax.pcolormesh(lon,
                       lat,
                       trend,
                       vmin=vlim0,
                       vmax=vlim,
                       cmap=cmap,
                       transform=ccrs.PlateCarree())

    trend_signif = np.ma.masked_where(np.isnan(pval),
                                      trend)

    if percent_increase:
        pass
    else:
        plt.pcolor(lon,
                   lat,
                   trend_signif,
                   linewidth=0.001,
                   hatch='///', alpha=0.,
                   transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.NaturalEarthFeature('physical',
                                                'land',
                                                '110m',
                                                edgecolor='face',
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
        cmap = mcolors.LinearSegmentedColormap.from_list("mycmap",
                                                         colors)

        ic = ax.contourf(ice[0].lon,
                         ice[0].lat,
                         ice[0],
                         np.arange(10, 110, 30),
                         cmap=cmap,
                         transform=ccrs.PlateCarree())

        ic2 = ax.contour(ice[1].lon,
                         ice[1].lat,
                         ice[1],
                         levels=[10],
                         linestyles=('solid',),
                         colors='green',
                         linewidths=1.,
                         transform=ccrs.PlateCarree())

        # plt.savefig(../plots/fig_title, dpi=300)
        # plt.close()

        return [ic, ic2]
    else:
        return




def iterate_subfig(fig, subfigs, fig_name, trend_vars, ice_var,
                   pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=True, percent_increase=False, seaice_conc=False, dcaa_plot=False):
    """ Organize and Plots calls function to plot each map of Arctic trends
    :returns None """
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
            if dcaa_plot:
                ice_var_idx = ice_var[idx]
                yloc = 10
            else:
                ice_var_idx = ice_var
                yloc = 6
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
                                ice_var_idx,
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
        add_ice_colorbar(fig,
                         ic[0],
                         ic[1],
                         yloc)
    plt.savefig(f'../plots/{fig_name}',
                dpi=200,
                bbox_inches="tight")


def plot_6_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True, percent_increase=False):
    fig = plt.figure(constrained_layout=True, figsize=(10, 7))

    (subfig1, subfig2, subfig3), (subfig4, subfig5, subfig6) = fig.subfigures(nrows=2,
                                                                              ncols=3)
    subfigs = [subfig1, subfig2, subfig3, subfig4, subfig5, subfig6]
    fig_name = f'Six_panel_ocean_omf{fig_name}_Arctic_trends.png'
    iterate_subfig(fig,
                   subfigs,
                   fig_name,
                   trend_vars,
                   seaice,
                   pval_vars,
                   lat,
                   lon,
                   vlim_vars,
                   unit_vars,
                   title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase)


def plot_2_panel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False,
                        dcaa_plot=False):
    """ Creates a plot of the 2-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True, figsize=(7, 4))

    (subfig1, subfig2) = fig.subfigures(nrows=1,
                                        ncols=2)
    subfigs = [subfig1, subfig2]

    fig_name = f'two_panel_{fig_name}_trends.png'
    iterate_subfig(fig,
                   subfigs, fig_name,
                   trend_vars,
                   seaice,
                   pval_vars,
                   lat,
                   lon,
                   vlim_vars,
                   unit_vars,
                   title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase,
                   dcaa_plot=dcaa_plot)


def plot_3_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True):
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))

    (subfig1, subfig2, subfig3) = fig.subfigures(nrows=1,
                                                 ncols=3)
    subfigs = [subfig1, subfig2, subfig3]

    fig_name = f'three_panel_ocean{fig_name}_vars_trends.png'
    iterate_subfig(fig,
                   subfigs,
                   fig_name,
                   trend_vars,
                   seaice,
                   pval_vars,
                   lat,
                   lon,
                   vlim_vars,
                   unit_vars,
                   title_vars,
                   not_aerosol=not_aerosol)


def plot_4_panel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False, seaice_conc=False):
    """ Creates a plot of the 4-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2,
                                                            ncols=2)

    fig_name = f'four_panel_{fig_name}.png'
    print(title_vars[0][0])

    subfigs = [subfig1, subfig2, subfig3, subfig4]
    iterate_subfig(fig,
                   subfigs,
                   fig_name,
                   trend_vars,
                   seaice,
                   pval_vars,
                   lat,
                   lon,
                   vlim_vars,
                   unit_vars,
                   title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase, seaice_conc=seaice_conc)


def plot_4_panel_trend_burden(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False, seaice_conc=False, three_panel=False):
    """ Creates a plot of the 4-panel trend
    :returns  None"""
    if three_panel:
        fig = plt.figure(constrained_layout=True,
                         figsize=(10, 7))
        subfig1, subfig2, subfig3 = fig.subfigures(nrows=1,
                                                   ncols=3)
        trend_vars = trend_vars[:3]
        subfigs = [subfig1, subfig2, subfig3]
    else:
        fig = plt.figure(constrained_layout=True,
                         figsize=(7, 7))
        (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2,
                                                                ncols=2)
        subfigs = [subfig1, subfig2, subfig3, subfig4]

    fig_name = f'four_panel_{fig_name}.png'

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
                        burden=True,)

    cbar_ax = fig.add_axes([0.05, -0.05, 0.9, 0.03])  # (left, bottom, width, height)

    tick = np.linspace(-vlim0,
                       vlim0,
                       11)
    cbar = plt.colorbar(cb,
                        cax=cbar_ax,
                        extend='both',
                        orientation='horizontal',
                        fraction=0.05,
                        ticks=tick,
                        # format='%.0e',
                        pad=0.12)
    # cbar.ax.set_yticklabels(tick)

    # cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    cbar.set_label(label='% yr$^{-1}$',
                   fontsize=14, )
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(f'../plots/{fig_name}',
                dpi=300,
                bbox_inches="tight")


def plot_6_2_panel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                          not_aerosol=True, percent_increase=False, seaice_conc=False):
    """ Creates a plot of the 6-panel trend
    :returns  None"""
    fig = plt.figure(constrained_layout=True,
                     figsize=(7, 10))

    (subfig1, subfig2), (subfig3, subfig4), (subfig5, subfig6) = fig.subfigures(nrows=3,
                                                                                ncols=2)
    subfigs = [subfig1, subfig2, subfig3, subfig4, subfig5, subfig6]
    fig_name = f'Six_panel_ocean_omf{fig_name}_Arctic_trends.png'
    iterate_subfig(fig,
                   subfigs, fig_name,
                   trend_vars,
                   seaice,
                   pval_vars,
                   lat,
                   lon,
                   vlim_vars,
                   unit_vars,
                   title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase,
                   seaice_conc=seaice_conc)

    return None
