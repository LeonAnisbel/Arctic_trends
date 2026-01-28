import pickle
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.path as mpath
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib import ticker as mticker


if __name__ == '__main__':
    with open(f"Spearman_corr_emiss_drivers_reg_AMJ.pkl", "rb") as myFile:
        sp_corr_AMJ = pickle.load(myFile)
    with open(f"Spearman_corr_emiss_drivers_reg_JAS.pkl", "rb") as myFile:
        sp_corr_JAS = pickle.load(myFile)
    print(sp_corr_AMJ)
    print(sp_corr_JAS)
    # for var in list(sp_corr_AMJ.keys()):
    #     print(var)
    #     for reg in list(sp_corr_AMJ[var].keys()):
    #         print(sp_corr_AMJ[var][reg], sp_corr_JAS[var][reg],
    #               sp_corr_AMJ[var][reg], sp_corr_JAS[var][reg],
    #               sp_corr_AMJ[var][reg], sp_corr_JAS[var][reg],
    #               sp_corr_AMJ[var][reg], sp_corr_JAS[var][reg],)


    with open(f"Spearman_corr_emiss_drivers_AMJ.pkl", "rb") as myFile:
        sp_corr_AMJ = pickle.load(myFile)


    with open(f"Spearman_corr_emiss_drivers_JAS.pkl", "rb") as myFile:
        sp_corr_JAS = pickle.load(myFile)

    fig, ax = plt.subplots(2,
                           4,
                           figsize=(12, 7),
                            constrained_layout = True,
                           subplot_kw={'projection': ccrs.NorthPolarStereo()},
                           )

    axs = ax.flatten()
    drivers = [sp_corr_AMJ.sic,
               sp_corr_AMJ.sst,
               sp_corr_AMJ.u10,
               sp_corr_AMJ.omf,
               sp_corr_JAS.sic,
               sp_corr_JAS.sst,
               sp_corr_JAS.u10,
               sp_corr_JAS.omf,]
    title_lab = [['AMJ', 'SIC'],
                 ['AMJ', 'SST'],
                 ['AMJ', 'Wind 10m'],
                 ['AMJ', 'OMF'],
                 ['JAS','SIC'],
                 ['JAS','SST'],
                 ['JAS','Wind 10m'],
                 ['JAS','OMF'],
                 ]
    for i, dv in enumerate(drivers):
        axs[i].set_extent([-180, 180, 60, 90],
                      ccrs.PlateCarree())
        pl = axs[i].pcolormesh(dv.lon,
                               dv.lat,
                               dv,
                               vmin=-1,
                               vmax=1,
                               cmap='coolwarm',
                               transform=ccrs.PlateCarree())
        axs[i].set_title(title_lab[i][0],
                         loc='left',
                         weight='bold',)
        axs[i].set_title(title_lab[i][1],
                         loc='right',
                         weight='bold',)

        # # compute circle
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

        axs[i].coastlines(color='darkgray')
        gl = axs[i].gridlines(draw_labels=True,
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

    # plt.text(0.3,
    #          3,
    #          'July-August-September')
    # plt.text(0.3,
    #          1,
    #          'April-May-June',
    #          wrap = True)
    cbar_ax = fig.add_axes([0.11, 0.05, 0.8, 0.03])  # (left, bottom, width, height)
    tick = np.linspace(-1,
                       1,
                       11)
    cbar = plt.colorbar(pl,
                        cax=cbar_ax,
                        orientation='horizontal',
                        fraction=0.05,
                        ticks=tick,
                        # format='%.0e',
                        pad=0.12)
    cbar.ax.tick_params(labelsize=14)

    # fig.tight_layout()
    fig.savefig('./plots/Spearman_corr_emiss_drivers.png',
                # bbox_inches='tight',
                dpi=300)


