import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib import ticker as mticker
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from pyparsing import alphas
from sklearn.linear_model import LinearRegression
from decimal import Decimal


def plot_fit(ax, t_ax, p_fit, eq, color, a):
    return ax.plot(t_ax, p_fit, color, linestyle='dashed', label=eq, linewidth=1, alpha=a)


def plot_fit_trends(ax, C, title, axis_label, vm, colors, leg, fig_name,
                    echam_data=True, seaice=False, multipanel=False):
    if multipanel:  # ,limits):
        pass
    else:
        fig, ax = plt.subplots(1, 1,
                               figsize=(8, 3), )

    C_ice, C_biom = C[0]['1990-2019']['data_aver_reg'], C[1]['1990-2019']['data_aver_reg']
    t_ax = C_ice.time.values

    ax2 = ax.twinx()
    new_line_color = colors[0]
    if seaice:
        new_line_color = 'lightblue'
        ax.fill_between(t_ax, 0, C_ice, alpha=0.2, color=new_line_color)  # C_ice.min() - C_ice.min()/10
    #         ax.fill_between(t_ax[:16],C_ice_doc[:16]-0.4,C_ice_doc[:16], alpha=0.5,color=new_line_color)
    p1, = ax.plot(t_ax, C_ice, new_line_color, label=leg[0], linewidth=1.)
    ax.scatter(t_ax, C_ice, s=15, c=new_line_color)
    p2, = ax2.plot(t_ax, C_biom, colors[1], label=leg[1], linewidth=1.)
    ax2.scatter(t_ax, C_biom, s=15, c=colors[1])

    decades = ['1990-1999', '2000-2009', '2010-2019']
    decades = ['1990-2004', '2005-2019']

    a = [0.5, 1]
    f1_list, f2_list = [], []
    for idx, dec in enumerate(decades):
        # print(dec, 'min SIC', C[0][dec]['data_aver_reg'].min().values,
        #         'max SIC', C[0][dec]['data_aver_reg'].max().values,
        #         'mean SIC',  C[0][dec]['data_aver_reg'].mean().values)
        # print(dec, 'min PMOA', C[1][dec]['data_aver_reg'].min().values,
        #         'max PMOA', C[1][dec]['data_aver_reg'].max().values,
        #         'mean PMOA', C[1][dec]['data_aver_reg'].mean().values, '\n')

        # sl = [C[0][dec]['slope_aver_reg'], C[1][dec]['slope_aver_reg']]
        # itc = [C[0][dec]['intercept_aver_reg'], C[1][dec]['intercept_aver_reg']]
        # pval = [C[0][dec]['pval_aver_reg'], C[1][dec]['pval_aver_reg']]

        t_ax = C[0][dec]['data_aver_reg'].time.values

        x = t_ax.reshape((-1, 1))
        y = C[0][dec]['data_aver_reg'].values
        print(x, y)
        model = LinearRegression()
        model_sic = model.fit(x, y)
        print(f"coefficient of determination SIC: {model.score(x, y)}")

        y = C[1][dec]['data_aver_reg'].values
        model1 = LinearRegression()
        model_emi = model1.fit(x, y)
        print(f"coefficient of determination EMI: {model1.score(x, y)}")

        sl = [model_sic.coef_[0], model_emi.coef_[0]]
        itc = [model_sic.intercept_, model_emi.intercept_]

        p_fit = [p * sl[0] + itc[0] for p in t_ax]
        m = "{:.2E}".format(Decimal(float(sl[0])))
        b = "{:.2E}".format(Decimal(float(itc[0])))
        eq = m+' x + '+b

        eq = f'{sl[0]:.1e}x + {itc[0]:.1e}'

        f1, = plot_fit(ax, t_ax, p_fit, eq, colors[0], a[idx])
        f1_list.append(f1)

        p_fit = [p * sl[1] + itc[1] for p in t_ax]
        eq = f'{sl[1]:.1e}x + {itc[1]:.1e}'
        f2, = plot_fit(ax2, t_ax, p_fit, eq, colors[1], a[idx])
        f2_list.append(f2)

    ax.set_ylabel(axis_label[0], fontsize=8)
    ax.tick_params(axis='x')

    ax.yaxis.get_label().set_fontsize(8)
    ax2.set_ylabel(axis_label[1], fontsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax2.yaxis.set_tick_params(labelsize=8, color=colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.yaxis.label.set_color(colors[1])

    ax.set_title(title[0] + '\n', loc='right', fontsize=10)
    ax.set_title(title[1], loc='left', fontsize=10)

    # ax.legend(loc='lower left', fontsize=8)
    # ax2.legend(loc='upper right', fontsize=8)
    ax.set_ylim(vm[0][0], vm[0][1])
    # ax2.set_ylim(vm[1][0], vm[1][1])

    ax.grid(linestyle='--', linewidth=0.4)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8, color=colors[0])
    ax.tick_params(axis='y', colors=colors[0])
    ax.yaxis.label.set_color(colors[0])

    ax2.ticklabel_format(axis="y", useMathText=True, useLocale=True)

    if echam_data:
        def format_func(value, tick_number):
            N = int(value + 1990)
            return N

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
               not_aerosol=True, percent_increase=False, seaice_conc=False):
    # fig, ax = plt.subplots(1, 1,
    #                        figsize=(5, 4),
    #                        subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
    ax = subfig.subplots(nrows=1,
                         ncols=1,
                         sharex=True,
                         subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
    ax.set_extent([-180, 180, 60, 90],
                  ccrs.PlateCarree())
    ax.set_title(titles[0], loc='right', fontsize=12)
    ax.set_title(titles[1], loc='left', fontsize=12)

    cmap = plt.get_cmap(cm, 15)
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
                        pad=0.07)
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
    gl = ax.gridlines(draw_labels=True, )
    gl.bottom_labels = False
    gl.left_labels = False
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator([65, 75, 85])
    gl.yformatter = LATITUDE_FORMATTER

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


# def plot_each_fig(subfig,C,titles,vm, units, colorbar):
#     axes = subfig.subplots(nrows=1, ncols=1, sharex=True,
#                          subplot_kw={'projection': ccrs.Robinson()})
#     cmap = plt.get_cmap(colorbar, 11)    # 11 discrete colors
#     im = axes.pcolormesh(C.lon, C.lat, C,
#                         cmap=cmap, transform=ccrs.PlateCarree(),
#                        vmin = 0,vmax = vm)
#     axes.set_title(titles[0],loc='right', fontsize = 12)
#     axes.set_title(titles[1], loc='left', fontsize = 12)
#     axes.coastlines()
#
#
#     cbar = subfig.colorbar(im, orientation="horizontal", extend = 'max')#,cax = cbar_ax
#     cbar.ax.tick_params(labelsize=12)
#     cbar.set_label(label=units, size='large', weight='bold')


def iterate_subfig(fig, subfigs, fig_name, trend_vars, ice_var,
                   pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=True, percent_increase=False, seaice_conc=False):
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


def plot_2_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False):
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


def plot_4_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                        not_aerosol=True,
                        percent_increase=False, seaice_conc=False):
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2, ncols=2)

    fig_name = f'four_panel_{fig_name}.png'
    print(title_vars[0][0])

    subfigs = [subfig1, subfig2, subfig3, subfig4]
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars,
                   unit_vars, title_vars,
                   not_aerosol=not_aerosol,
                   percent_increase=percent_increase, seaice_conc=seaice_conc)


def plot_6_2_pannel_trend(trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars, fig_name,
                          not_aerosol=True, percent_increase=False, seaice_conc=False):
    fig = plt.figure(constrained_layout=True, figsize=(7, 10))

    (subfig1, subfig2), (subfig3, subfig4), (subfig5, subfig6) = fig.subfigures(nrows=3, ncols=2)
    subfigs = [subfig1, subfig2, subfig3, subfig4, subfig5, subfig6]
    fig_name = f'Six_panel_ocean_omf{fig_name}_Arctic_trends.png'
    iterate_subfig(fig, subfigs, fig_name, trend_vars, seaice, pval_vars, lat, lon, vlim_vars, unit_vars, title_vars,
                   not_aerosol=not_aerosol, percent_increase=percent_increase, seaice_conc=seaice_conc)

    return None
