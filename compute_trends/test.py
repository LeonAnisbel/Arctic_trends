import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(15, 8),
                               subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
ax.coastlines(resolution='110m', color='gray')
ax.set_extent([-180, 180, 60, 90],
              ccrs.PlateCarree())
gl = ax.gridlines(draw_labels=False,
                  x_inline=False,
                  y_inline=False)
gl.top_labels = False
gl.right_labels = False

data = xr.open_dataset('../test_scripts/ac3_arctic_199005.01_vphysc.nc')
da_t = data.mean(dim='time')['tsw']-273
# da_t = da_t.where(da_t < 5)
print(da_t)
gl = ax.pcolormesh(da_t.lon, da_t.lat, da_t,
                       transform=ccrs.PlateCarree() , cmap='Reds')
plt.colorbar(gl, ax = ax)
plt.show()
