import numpy as np
import xarray as xr
import statsmodels.api as sm
from Compute_trends.process_statsmodels import process_array_slope
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
from matplotlib import ticker as mticker

ftype = np.float64
 
def read_files_data(path_dir):
    data = xr.open_mfdataset(path_dir,concat_dim='time', combine='nested')
    return data
 
if __name__ == '__main__':
    C_omf = read_files_data("../OMF_data/fesom_recon_oceanfilms_omf_*")
 
    C_omf_month = C_omf.where(C_omf.time.dt.month == 7,drop=True)['OMF_LIP']
    X = C_omf_month.time.dt.year.values.astype(ftype)
 
    X = sm.add_constant(X)
 
    C_omf_month = C_omf_month.where(C_omf_month.lat>63,drop=True)
    Y = C_omf_month.values.astype(ftype)
 
    x_lat, y_lon  = Y.shape[1:]
 
    C_omf_month_slope = np.empty((x_lat,y_lon), dtype=ftype)
    C_omf_month_p_value = np.empty((x_lat,y_lon), dtype=ftype)
 
    process_array_slope(Y, X, C_omf_month_slope, C_omf_month_p_value)

    
    fig,ax = plt.subplots(1,1,figsize=(5, 4),
                      subplot_kw={'projection': ccrs.NorthPolarStereo()},)
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

    cb = ax.pcolormesh(C_omf_month.lon, C_omf_month.lat,
                       C_omf_month_p_value, cmap = 'jet',
                       transform = ccrs.PlateCarree())
    #hatch = ax.fill_between([0,1],0,1,
     #                        hatch='///',
      #                       color="none",
       #                      edgecolor='black',
        #                     transform = ax.transAxes) 
#    C_omf_month_slope_signif = np.ma.masked_where(C_omf_month_p_value==0, C_omf_month_slope)
 #   plt.pcolor(C_omf_month.lon, 
  #          C_omf_month.lat, 
   #         C_omf_month_slope_signif, 
    #        linewidth=0.1,
     #       hatch='////', alpha=0.,
      #      transform = ccrs.PlateCarree())    
    #print(C_omf_month_p_value.min(),C_omf_month_p_value.max())
#    C_omf_month_slope_signif = np.ma.masked_where(C_omf_month_p_value==0, C_omf_month_slope)
 #   cb = ax.pcolormesh(C_omf_month.lon, C_omf_month.lat,

  #                     C_omf_month_slope_signif, vmin = -0.01, vmax=0.01,
   #                    cmap = 'coolwarm',
    #                   transform = ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land',
                        '10m', edgecolor='face',
                        facecolor='lightgray'))
 


    plt.colorbar(cb)

    ax.coastlines(color= 'gray')
    gl = ax.gridlines(draw_labels=True,)
    plt.savefig('P_value_OMF_LiP.png',dpi = 300)
    gl.ylocator = mticker.FixedLocator([65, 75, 85])
    gl.yformatter = LATITUDE_FORMATTER

