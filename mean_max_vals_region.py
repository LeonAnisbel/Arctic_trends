import numpy as np
from utils import get_var_reg, get_seaice_vals, get_min_seaice, regions, get_conds
import pickle
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def vicinity_grid_vals(reg_sel_vals, max_val):
    ds_max_val = reg_sel_vals['slope'].where(reg_sel_vals['slope']==max_val, drop=True)
    #print(ds_max_val, 'ds_max_val')
    #if len(ds_max_val.lon.values) > 1:
     #   lon_max_val = ds_max_val.lon.values.min()

    lat_max_val, lon_max_val = ds_max_val.lat.values.min(), ds_max_val.lon.values.min()

    size = reg_sel_vals['slope'].lat.values[1]-reg_sel_vals['slope'].lat.values[0]
    lat_list = np.arange(lat_max_val-3*size, lat_max_val+3*size, size)
    lon_list = np.arange(lon_max_val-3*size, lon_max_val+3*size, size)
    sel_vicinity_max = reg_sel_vals['slope'].sel(lat=lat_list, lon=lon_list, method="nearest")
    return np.sum(abs(np.logical_not(np.isnan(sel_vicinity_max.values))))
    

def reg_sel(lat, lon, data, var_na):
    reg_data = regions()
    for idx, reg_na in enumerate(list(reg_data.keys())):
        data_ds = xr.Dataset(
            data_vars=dict(
                slope=(["lat", "lon"], data.data),
            ),
            coords=dict(
                lon=("lon", lon.data),
                lat=("lat", lat.data),
            ),
        )
        conditions = get_conds(data_ds.lat, data_ds.lon)

        reg_sel_vals = get_var_reg(data_ds, conditions[idx])

        if var_na=='PCHO' and reg_na == 'Greenland & Norwegian Sea':
#            print(reg_na, reg_sel_vals.slope.max().values)
        #
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
            from matplotlib import ticker as mticker
            fig, ax = plt.subplots(nrows=1,
                                   ncols=1,
                                   sharex=True,
                                   subplot_kw={'projection': ccrs.NorthPolarStereo()}, )
            ax.set_extent([-180, 180, 60, 90],
                          ccrs.PlateCarree())
            cmap = plt.get_cmap('Blues', 15)
            cb = ax.pcolormesh(reg_sel_vals.lon,
                               reg_sel_vals.lat,
                               np.array(reg_sel_vals['slope']),
                               vmax=0.05,
                               cmap=cmap,
                               transform=ccrs.PlateCarree())
            ax.coastlines(color='darkgray')
            plt.colorbar(cb)
            gl = ax.gridlines(draw_labels=True, )
            gl.ylocator = mticker.FixedLocator([65, 75, 85])
            plt.savefig(f'TEST.png')

        total_non_nan = np.sum(np.logical_not(np.isnan(reg_sel_vals.slope.values)))
        total_non_nan_1 = reg_sel_vals.slope.where(reg_sel_vals.slope>=0,drop=True)
        total_non_nan_2 = reg_sel_vals.slope.where(reg_sel_vals.slope<0,drop=True)

#print(var_na, reg_na,total_non_nan,len(total_non_nan_1.lat)*len(total_non_nan_1.lon)+len(total_non_nan_2.lat)*len(total_non_nan_2.lon))
        total_grid_size = len(reg_sel_vals['slope'].lat)*len(reg_sel_vals['slope'].lon)
#        print(var_na, reg_na,'total not nan', total_grid_size, total_non_nan, reg_data[reg_na]['grid_signif'])

        if reg_sel_vals.slope.shape[0] > 1 and total_non_nan > 0:
            reg_data[reg_na]['grid_signif'] = total_non_nan*100/total_grid_size
            #print(var_na, reg_na,'total not nan', total_grid_size, total_non_nan, reg_data[reg_na]['grid_signif'])

            reference_lenght = 5
            max_val = reg_sel_vals['slope'].max(skipna=True).values 
            reg_data[reg_na]['max'] = max_val
            min_val = reg_sel_vals['slope'].min(skipna=True).values
            reg_data[reg_na]['min'] = min_val


            vicinity_max_vals = vicinity_grid_vals(reg_sel_vals, max_val)
            vicinity_min_vals = vicinity_grid_vals(reg_sel_vals, min_val)
            
          
         
          
            abs_max = max([abs(min_val), abs(max_val)])
            reg_data[reg_na]['max_val'] = float(max_val)
            grid_posit_vals = reg_sel_vals['slope'].where(reg_sel_vals['slope']>0, drop=True)

            reg_data[reg_na]['fraction_grid_posit'] = np.sum(np.logical_not(np.isnan(grid_posit_vals.values)))*100/total_non_nan
                #len(grid_posit_vals.lat)*len(grid_posit_vals.lon)*100/total_non_nan

            reg_data[reg_na]['min_val'] = float(min_val)
            grid_negat_vals = reg_sel_vals['slope'].where(reg_sel_vals['slope']<0, drop=True)
            reg_data[reg_na]['fraction_grid_negat'] = np.sum(np.logical_not(np.isnan(grid_negat_vals.values)))*100/total_non_nan
            #len(grid_negat_vals.lat)*len(grid_negat_vals.lon)*100/total_non_nan

            print('\n', var_na, reg_na, reg_data[reg_na]['grid_signif'], reg_data[reg_na]['fraction_grid_posit'],reg_data[reg_na]['fraction_grid_negat'])

            if abs(min_val) > abs(max_val):
                if reg_data[reg_na]['fraction_grid_negat']>reg_data[reg_na]['fraction_grid_posit']:
                    reg_data[reg_na]['max_absolute'] = float(min_val)
                else:
                    reg_data[reg_na]['max_absolute'] = float(max_val)

            else:
                if reg_data[reg_na]['fraction_grid_negat']<reg_data[reg_na]['fraction_grid_posit']:
                    reg_data[reg_na]['max_absolute'] = float(max_val)
                else:
                    reg_data[reg_na]['max_absolute'] = float(min_val)

            
   
            #reg_data[reg_na]['max_absolute'] = float(reg_sel_vals.slope.mean(skipna=True))

            #if abs(min_val) > abs(max_val):  
             #   if vicinity_min_vals > reference_lenght :
              #      reg_data[reg_na]['max_absolute'] = float(min_val)
               # elif vicinity_max_vals > reference_lenght:
                #    reg_data[reg_na]['max_absolute'] = float(max_val)
            #else:
             #   if vicinity_max_vals > reference_lenght:
              #      reg_data[reg_na]['max_absolute'] = float(max_val)
               # elif vicinity_min_vals > reference_lenght:
                #    reg_data[reg_na]['max_absolute'] = float(min_val)

            #if vicinity_max_vals < reference_lenght and vicinity_min_vals < reference_lenght:
             #   if vicinity_max_vals > vicinity_min_vals:
              #      reg_data[reg_na]['max_absolute'] = float(max_val)
               # else:
                #    reg_data[reg_na]['max_absolute'] = float(min_val)
               
            # print(reg_na, 'max = ', reg_data[reg_na][decade]['max'], 'min =', reg_data[reg_na][decade]['min'])
        else:
            reg_data[reg_na]['max_absolute'] = np.nan
            reg_data[reg_na]['grid_signif'] = np.nan


            reg_data[reg_na]['max_val'] = np.nan
            reg_data[reg_na]['fraction_grid_posit'] = np.nan

            reg_data[reg_na]['min_val'] = np.nan
            reg_data[reg_na]['fraction_grid_negat'] = np.nan
            # print(reg_na, 'min = ', reg_sel_vals.slope.min())
    return reg_data


def create_df_plot_heatmap(col, col_name, return_colorbar=False):

    df_vals = pd.DataFrame({'Regions': col[0],
                            col_name: col[1],
                            'Values': col[2],
                            })
    # fig = plt.figure(figsize=(6, 6))
    if col_name[:3] == 'OMF' or col_name[:3] == 'Oce':
        df_vals = df_vals[df_vals['Regions'] != 'Central Arctic']
    # if col_name == ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ per unit SIC)':
    #     df_vals = df_vals[df_vals['Regions'] != 'Barents Sea']
    df_vals_piv = df_vals.pivot(index="Regions", columns=col_name, values="Values")

    if return_colorbar:
        vmin_val = min(df_vals['Values'])
        vmax_val = max(df_vals['Values'])
        if vmin_val < 0 and vmax_val > 0:
            cmap = 'RdBu_r'
            vmin_val = -vmax_val
        if vmin_val > 0:
            cmap = 'Reds'
            
        if vmax_val < 0:
            cmap = 'Blues_r'

        return df_vals_piv, [cmap, vmin_val, vmax_val]
    else:
        return df_vals_piv




def plot_heatmap(df_vals_piv, col_name, fig_title):
    fig, ax = plt.subplots(1, 1,
                           figsize=(7, 5), )
    plot_each_heatmap(ax, df_vals_piv, col_name)
    plt.tight_layout()
    plt.savefig('Heatmap_' + fig_title + '.png')
    plt.close()


def plot_each_heatmap(ax, df_vals_piv, fig_title, cmap, no_ylabel=False, right_label=True):

    # if col_name[:18] == 'Emission mass flux':
    #     axs = sns.heatmap(df_vals_piv, annot=True, cmap=cmap, norm=LogNorm(), ax=ax)
    # else:
    hm = sns.heatmap(df_vals_piv, 
            annot=True,
            vmin = cmap[1],
            vmax = cmap[2],
            cmap=cmap[0], 
            ax=ax)
    if no_ylabel:
        hm.set(yticklabels=[])
        ax.tick_params(left=False, bottom=False)
    hm.set(ylabel="", xlabel="")
    hm.xaxis.tick_top()
    if right_label:
        hm.set_title(fig_title, loc='right')


def get_df_data(variables_info_yr, decade, reg_names, panel_names, var_na_aer):

    columns1 = [[], [], []]
    columns2 = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in reg_names:
            columns1[0].append(reg_na)
            columns2[0].append(reg_na)

            slope = variables_info_yr[var_na][reg_na][decade]['slope_aver_reg']
            slope = percent_icrease(variables_info_yr, var_na, reg_na, decade)
            columns1[1].append(var_na_aer[vidx])
            columns1[2].append(slope)

            slope = variables_info_yr[var_na][reg_na][decade]['slope_aver_reg']
            slope = percent_icrease(variables_info_yr, var_na, reg_na, decade)

            #variables_info_seaice[var_na][reg_na]['slope_aver_reg']
            columns2[1].append(var_na_aer[vidx])
            columns2[2].append(slope)
    return columns1, columns2

def scatter_plot(fig, axs, df, col_name, title,vm, no_left_labels=False, no_colorbar=False):
   
    sc = axs.scatter(
            x = df['Variables'],
            y = df['Regions'],
            c = df[col_name],
            s = df[col_name],
            cmap= 'viridis', 
            vmax = vm,
            )

    axs.tick_params(axis = 'x', pad=0.2)
    axs.xaxis.labelpad = 0.2
    #plt.xlim((-1,1))
    axs.set_xlim((-0.5,1.5))

    if no_left_labels:
        axs.set(yticklabels=[])
        axs.tick_params(left=False)
    axs.set(ylabel="", xlabel="")

    axs.set_title(title[0], loc='right')

    if no_colorbar:
        plt.colorbar(sc, ax=axs).remove()
    else:
        cbar = plt.colorbar(sc, ax=axs)
        cbar.set_label(title[1])
       
def percent_icrease(variables_info_yr, vv, reg_na, decade):
    pval = variables_info_yr[vv][reg_na][decade]['pval_aver_reg']
    if pval < 0.05:
        interc = variables_info_yr[vv][reg_na][decade]['intercept_aver_reg']
        slope = variables_info_yr[vv][reg_na][decade]['slope_aver_reg']
        vals = variables_info_yr[vv][reg_na][decade]['data_aver_reg']
        last_val = slope*30+interc
        perc_inc = (last_val/interc-1)*100/30
        print(slope, perc_inc,interc, pval)
    else:
        perc_inc=np.nan
    return perc_inc


if __name__ == '__main__':

    season = 'JAS'
    with open(f"TrendsDict_{season}.pkl", "rb") as myFile:
        variables_info_yr = pickle.load(myFile)

    with open(f"TrendsDict_per_ice_{season}.pkl", "rb") as myFile:
        variables_info_seaice = pickle.load(myFile)

    print('Aerosols from ECHAM')
    ## Calculate mean values per regions for emiss flux trends and emiss flux per unit of SIC
    panel_names = ['AER_F_POL_yr', 'AER_F_PRO_yr', 'AER_F_LIP_yr', 'AER_F_SS_yr']
    var_na_aer = ['PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$', 'SS$_{aer}$']
    lat = variables_info_yr[panel_names[0]]['lat']
    lon_360 = variables_info_yr[panel_names[0]]['lon']
    lon = ((lon_360 + 180) % 360) - 180
    decade = '1990-2019'
    reg_names = regions()
    columns1, columns2 = get_df_data(variables_info_yr, decade, reg_names, panel_names, var_na_aer)

    columns3 = [[], [], []]
    for reg_na in reg_names:
        columns3[0].append(reg_na)
        vv = 'AER_SIC'
        slope = variables_info_yr[vv][reg_na][decade]['slope_aver_reg'] 

        slope = percent_icrease(variables_info_yr, vv, reg_na, decade)
        columns3[1].append('')#'SIC (% ${yr^{-1}}$)'
        columns3[2].append(slope)
        

    columns4 = [[], [], []]
    for reg_na in reg_names:
        columns4[0].append(reg_na)
        vv = 'AER_SST'
        slope = variables_info_yr[vv][reg_na][decade]['slope_aver_reg']
        
        slope = percent_icrease(variables_info_yr, vv, reg_na, decade)
        columns4[1].append('')#'SST (C$^{o}$ ${yr^{-1}}$)'
        columns4[2].append(slope)

    columns5 = [[], [], []]
    for reg_na in reg_names:
        columns5[0].append(reg_na)
        vv = 'AER_U10'
        slope = variables_info_yr[vv][reg_na][decade]['slope_aver_reg']

        slope = percent_icrease(variables_info_yr, vv, reg_na, decade)
        columns5[1].append('')#'Wind (m $s^{-1}$ ${yr^{-1}}$)'
        columns5[2].append(slope)

    # col_name_emi = ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ ${yr^{-1}}$)'
    # df_vals_piv_emi = create_df_plot_heatmap(columns1, col_name_emi)
    # plot_heatmap(df_vals_piv_emi, col_name_emi, 'Aerosol_fluxes_')
    # col_name_emi_sic = ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ per unit SIC)'
    # df_vals_piv_emi_sic = create_df_plot_heatmap(columns2, col_name_emi_sic)
    # plot_heatmap(df_vals_piv_emi_sic, col_name_emi_sic, 'Aerosol_fluxes_per unit_SIC_')
    # col_name_sic = ''
    # df_vals_piv_sic = create_df_plot_heatmap(columns3, col_name_sic)
    # plot_heatmap(df_vals_piv_sic, col_name_sic, 'SIC_')
    # #
    # #
    # fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    # ax.flatten()
    # # plot_each_heatmap(ax[0], df_vals_piv_sic, col_name_sic)
    # # ax[0].set_title(r'$\bf{(a)}$', loc='left')
    # plot_each_heatmap(ax[0], df_vals_piv_emi, col_name_emi)
    # ax[0].set_title(r'$\bf{(a)}$', loc='left')
    # plot_each_heatmap(ax[1], df_vals_piv_emi_sic, col_name_emi_sic)
    # ax[1].set_title(r'$\bf{(b)}$', loc='left')
    # plt.tight_layout()
    # plt.savefig('Heatmap_EmiFlux_SIC.png')
    # plt.close()


    # fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    # ax.flatten()
    # col_name_sic = 'SIC (% ${yr^{-1}}$)'
    # df_vals_piv_sic = create_df_plot_heatmap(columns3, col_name_sic)
    # plot_each_heatmap(ax[0], df_vals_piv_sic, col_name_sic)
    # ax[0].set_title(r'$\bf{(a)}$', loc='left')
    #
    # col_name_sst = 'SST (C$^{o}$ ${yr^{-1}}$)'
    # df_vals_piv_sst = create_df_plot_heatmap(columns4, col_name_sst)
    # plot_each_heatmap(ax[1], df_vals_piv_sst, col_name_sst)
    # ax[1].set_title(r'$\bf{(b)}$', loc='left')
    #
    # col_name_wind = 'Wind (m $s^{-1}$ ${yr^{-1}}$)'
    # df_vals_piv_wind = create_df_plot_heatmap(columns5, col_name_wind)
    # plot_each_heatmap(ax[2], df_vals_piv_wind, col_name_wind)
    # ax[2].set_title(r'$\bf{(c)}$', loc='left')
    # plt.tight_layout()
    # plt.savefig('Heatmap_SIC_SST_Wind.png')
    # plt.close()

###############################

    fig, axs = plt.subplots(2, 3, figsize=(8, 8))
    ax = axs.flatten()


    panel_names = ['AER_F_POL_yr']
    var_na_aer = ['PCHO$_{aer}$']
    decade = '1990-2019'
    col_name_emi = ' Emission mass flux \n (ng ${m^{-2}}$ ${s^{-1}}$ ${yr^{-1}}$)'

    reg_names = regions()
    columns1, columns2 = get_df_data(variables_info_yr, decade, reg_names, panel_names, var_na_aer)
    df_vals_piv_emi,cmap = create_df_plot_heatmap(columns1, col_name_emi, return_colorbar=True)
    ax[3].set_title(r'$\bf{(d)}$', loc='left')
    plot_each_heatmap(ax[3], df_vals_piv_emi, col_name_emi, cmap, right_label=False)

    panel_names = ['AER_F_LIP_yr']
    var_na_aer = [ 'PL$_{aer}$']
    reg_names = regions()
    columns1, columns2 = get_df_data(variables_info_yr, decade, reg_names, panel_names, var_na_aer)
    df_vals_piv_emi, cmap = create_df_plot_heatmap(columns1, col_name_emi, return_colorbar=True)
    plot_each_heatmap(ax[4], df_vals_piv_emi, col_name_emi, cmap, no_ylabel=True,  right_label=False)

    panel_names = [ 'AER_F_SS_yr']
    var_na_aer = [ 'SS$_{aer}$']
    reg_names = regions()
    columns1, columns2 = get_df_data(variables_info_yr, decade, reg_names, panel_names, var_na_aer)
    df_vals_piv_emi, cmap = create_df_plot_heatmap(columns1, col_name_emi, return_colorbar=True)
    plot_each_heatmap(ax[5], df_vals_piv_emi, col_name_emi, cmap, no_ylabel=True)

    # panel_names = [ 'AER_F_SS']
    # var_na_aer = [ 'SS']
    # reg_names = regions()
    # columns1, columns2 = get_df_data(variables_info_yr, decade, reg_names, panel_names, var_na_aer)
    # df_vals_piv_emi = create_df_plot_heatmap(columns1, col_name_emi)
    # plot_each_heatmap(ax[3], df_vals_piv_emi, col_name_emi, no_ylabel=True)

    col_name_sic = '\n SIC \n (% ${yr^{-1}}$)'
    df_vals_piv_sic, cmap = create_df_plot_heatmap(columns3, col_name_sic, return_colorbar=True)
    plot_each_heatmap(ax[0], df_vals_piv_sic, col_name_sic, cmap)
    ax[0].set_title(r'$\bf{(a)}$ '+'\n ', loc='left')

    col_name_sst = '\n SST \n (C$^{o}$ ${yr^{-1}}$)'
    df_vals_piv_sst, cmap = create_df_plot_heatmap(columns4, col_name_sst, return_colorbar=True)
    plot_each_heatmap(ax[1], df_vals_piv_sst, col_name_sst, cmap, no_ylabel=True)
    ax[1].set_title(r'$\bf{(b)}$'+'\n ', loc='left')

    #col_name_wind = '\n Wind \n (m $s^{-1}$ ${yr^{-1}}$)'
    #df_vals_piv_wind, cmap = create_df_plot_heatmap(columns5, col_name_wind, return_colorbar=True)
    #plot_each_heatmap(ax[2], df_vals_piv_wind, col_name_wind, cmap, no_ylabel=True)
    #ax[2].set_title(r'$\bf{(c)}$'+'\n ', loc='left')


    plt.tight_layout()
    plt.savefig(f'{season}_Heatmap_Emission_SIC_SST_Wind.png')
    plt.close()
    ###############################

    #fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    #plot_each_heatmap(ax, df_vals_piv_sic, col_name_sic)
    #plt.tight_layout()
    #plt.savefig(f'{season}_Heatmap_Aerosol_flux_and_flux_per_unit_SIC.png')
    #plt.close()
    print('''''')






    print('Biomolecules and OMF')
    ## Calculate max values per regions for ocean biomolecules and OMF
    panel_names = ['PCHO', 'DCAA', 'PL', 'Biom_tot', 'OMF_POL', 'OMF_PRO', 'OMF_LIP', 'OMF_tot']
    seaice = get_seaice_vals(variables_info_yr, 'Sea_ice')
    #seaice_min = seaice[0]
    seaice_min = get_min_seaice(variables_info_yr, 'Sea_ice')
    lat = variables_info_yr[panel_names[0]]['lat']
    lon = variables_info_yr[panel_names[0]]['lon']
    #apply min ice mask
    for var_na in panel_names:
        #print(var_na, seaice[2].shape,seaice_min.shape, variables_info_yr[var_na]['slope'].shape)
        data_seaice_mask = np.ma.masked_where(seaice_min > 10, variables_info_yr[var_na]['slope'])
        data_seaice_mask = np.ma.masked_where(np.isnan(variables_info_yr[var_na]['pval']), data_seaice_mask)
        data_seaice_mask = data_seaice_mask.filled(np.nan)

        variables_info_yr[var_na]['regions_vals'] = reg_sel(lat, lon, data_seaice_mask, var_na)
        print('''''')

        # print(data_seaice_mask)




    reg_names = regions()
    var_na_sw_aer = ['PCHO$_{sw}$', 'PL$_{sw}$']#, 'DCAA$_{sw}$', 'Total$_{sw}$']
    panel_names = ['PCHO', 'PL']#, 'DCAA','Biom_tot']
    columns = [[], [], []]
    columns1 = [[], [], []]
    

    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
           
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[0].append(reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_posit'] 
            columns1[0].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_negat']
            columns1[1].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['grid_signif']
            columns1[2].append(max_abs)

    col_name_oc = 'Ocean concentration \n (mmol C ${m^{-3}}$ ${yr^{-1}}$)'
    df_vals_piv_ocean, cmap_oc = create_df_plot_heatmap(columns, col_name_oc, return_colorbar=True)
#    plot_heatmap(df_vals_piv_ocean, col_name_oc, f'{season}_Ocean_conc_abs_max_')

    col_name_grid_pos = '% of grid with increasing trend'
    col_name_grid_neg = '% of grid with decreasing trend'
    col_name_signif = '% of grid with significant trend'
    data_percent = {'Regions': columns[0], 
            'Variables': columns[1], 
            col_name_grid_pos: columns1[0],
            col_name_grid_neg: columns1[1],
            col_name_signif: columns1[2],}
    df_ocean_grid_percent = pd.DataFrame(data_percent).sort_values(by=['Regions','Variables'], ascending=[False, True])
    df_ocean_grid_percent = df_ocean_grid_percent[df_ocean_grid_percent['Regions'] != 'Central Arctic'] 
  



    reg_names = regions()
    var_na_sw_aer = ['PCHO$_{sw}$', 'PL$_{sw}$']#, 'DCAA$_{sw}$', 'Total$_{sw}$']
    panel_names = ['PCHO', 'PL']#, 'DCAA','Biom_tot']
    columns = [[], [], []]
    df_vals_piv_ocean_pol_pl, cmap_oc_pol_pl = [], []
    for vidx, var_na in enumerate(panel_names):
        columns = [[], [], []]
        for reg_na in list(reg_names.keys())[1:]:

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            columns[0].append(reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)

        df_vals_piv, cmaps = create_df_plot_heatmap(columns, col_name_oc, return_colorbar=True)
        df_vals_piv_ocean_pol_pl.append(df_vals_piv)
        cmap_oc_pol_pl.append(cmaps)
    

    var_na_sw_aer = ['PCHO$_{aer}$', 'PL$_{aer}$',]# 'DCAA$_{aer}$',  'DCAA$_{aer}$']
    panel_names = ['OMF_POL', 'OMF_LIP', ]#'OMF_PRO', 'OMF_tot']
    columns1 = [[], [], []]
    columns2 = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns1[0].append(reg_na)
            columns1[1].append(var_na_sw_aer[vidx])

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_posit']
            columns2[0].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['fraction_grid_negat']
            columns2[1].append(max_abs)

            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['grid_signif']
            columns2[2].append(max_abs)

    col_name_grid_pos = '% of grid with increasing trend'
    col_name_grid_neg = '% of grid with decreasing trend'
    col_name_signif = '% of grid with significant trend'

    data_percent = {'Regions': columns1[0],
            'Variables': columns1[1],
            col_name_grid_pos: columns2[0],
            col_name_grid_neg: columns2[1],            
            col_name_signif: columns2[2],}
    
    df_omf_grid_percent = pd.DataFrame(data_percent).sort_values(by=['Regions','Variables'], ascending=[False, True])
    df_omf_grid_percent = df_omf_grid_percent[df_omf_grid_percent['Regions'] != 'Central Arctic']



    var_na_sw_aer = ['PCHO$_{aer}$']#,'DCAA$_{aer}$']#
    panel_names = ['OMF_POL']#,'OMF_PRO']
    columns = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns[0].append(reg_na)
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            # print(max_abs, var_na, reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)
    col_name_omf = 'OMF \n (% ${yr^{-1}}$)'
    df_vals_piv_omf_pol, cmap_omf_pol = create_df_plot_heatmap(columns, col_name_omf, return_colorbar=True)
#    plot_heatmap(df_vals_piv_omf, col_name_omf, f'{season}_OMF_abs_max_')


###################################3

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    fig = plt.figure(figsize=(14, 10))
    # ax.flatten()
    ax1 = plt.subplot2grid((11, 4), (1, 0), colspan=2, rowspan=5)
    ax2 = plt.subplot2grid((11, 4), (1, 2), rowspan=5)
    ax3 = plt.subplot2grid((11, 4), (1, 3), rowspan=5)

    ax4 = plt.subplot2grid((11, 4), (6, 0), rowspan=5)
    ax5 = plt.subplot2grid((11, 4), (6, 1), rowspan=5)
    ax6 = plt.subplot2grid((11, 4), (6, 2), rowspan=5)
    ax7 = plt.subplot2grid((11, 4), (6, 3), rowspan=5)

    plot_each_heatmap(ax1, df_vals_piv_ocean, col_name_oc, cmap_oc)
    ax1.set_title(r'$\bf{(a)}$'+'\n ', loc='left')
    plot_each_heatmap(ax2, df_vals_piv_omf_pol, col_name_omf, cmap, no_ylabel=True,right_label=False)
    ax2.set_title(r'$\bf{(b)}$'+'\n ', loc='left')

    var_na_sw_aer = ['PL$_{aer}$',]# 'Total$_{aer}$']#
    panel_names = ['OMF_LIP',]# 'OMF_tot']
    columns = [[], [], []]
    for vidx, var_na in enumerate(panel_names):
        for reg_na in list(reg_names.keys())[1:]:
            columns[0].append(reg_na)
            max_abs = variables_info_yr[var_na]['regions_vals'][reg_na]['max_absolute']
            # print(max_abs, var_na, reg_na)
            columns[1].append(var_na_sw_aer[vidx])
            columns[2].append(max_abs)
    col_name_omf = 'OMF \n (% ${yr^{-1}}$)'
    df_vals_piv_omf,cmap_omf_pl = create_df_plot_heatmap(columns, col_name_omf, return_colorbar=True)
 #   plot_heatmap(df_vals_piv_omf, col_name_omf, f'{season}_OMF_abs_max_')

    plot_each_heatmap(ax3, df_vals_piv_omf, col_name_omf, cmap, no_ylabel=True)

    vm = 100
    scatter_plot(fig, ax4, 
            df_ocean_grid_percent, 
            col_name_grid_pos,
            'Fraction with \n icreasing trend',
            vm,
            no_left_labels=False,
            no_colorbar=True)
    ax4.set_title(r'$\bf{(c)}$'+'\n ', loc='left')

    scatter_plot(fig, ax5, 
            df_ocean_grid_percent, 
            col_name_grid_neg, 
            'Fraction with \n decreasing trend',
            vm,
            no_left_labels=True,
            no_colorbar=False)
    #ax5.set_title('Ocean concentration '+'\n ', loc='right')

    scatter_plot(fig, ax6,
            df_omf_grid_percent,
            col_name_grid_pos,
            'Fraction with \n increasing trend',
            vm,
            no_left_labels=True,
            no_colorbar=True)
    ax6.set_title(r'$\bf{(d)}$'+'\n ', loc='left')

    scatter_plot(fig, ax7,
            df_omf_grid_percent,
            col_name_grid_neg,
            'Fraction with \n decreasing trend',
            vm,
            no_left_labels=True,
            no_colorbar=False)
    #ax7.set_title('OMF '+'\n ', loc='right')


    plt.tight_layout()
   # plt.savefig(f'{season}_Heatmap_Ocean_OMF.png',dpi = 300)    
   # plt.close()


#################################

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax.flatten()
    vm = 10
    scatter_plot(fig, ax[0],
            df_ocean_grid_percent,
            col_name_signif,
            'Ocean',
            30,
            no_left_labels=False,
            no_colorbar=False)
    ax[0].set_title(r'$\bf{(a)}$'+'\n ', loc='left')

    scatter_plot(fig, ax[1],
            df_omf_grid_percent,
            col_name_signif,
            'OMF',
            25,
            no_left_labels=True,
            no_colorbar=False)
    #ax7.set_title('OMF '+'\n ', loc='right')
    plt.tight_layout()
    #plt.savefig(f'{season}_scatter_plots_grid_fraction.png',dpi = 300)
    #plt.close()

#################################



    fig, axs = plt.subplots(2, 5, figsize=(14, 10))
    ax = axs.flatten()
    plot_each_heatmap(ax[0], df_vals_piv_ocean_pol_pl[0], col_name_oc, cmap_oc_pol_pl[0],right_label=False)
    ax[0].set_title(r'$\bf{(a)}$'+'\n ', loc='left')
    plot_each_heatmap(ax[1], df_vals_piv_ocean_pol_pl[1], col_name_oc, cmap_oc_pol_pl[1], no_ylabel=True)
    scatter_plot(fig, ax[4],
            df_ocean_grid_percent,
            col_name_signif,
            ['', 'Grid fraction with \n significant trend (%)'],
            30,
            no_left_labels=True,
            no_colorbar=False)
    ax[2].set_title(r'$\bf{(b)}$'+'\n ', loc='left')
    vm = 100
    scatter_plot(fig, ax[2],
            df_ocean_grid_percent,
            col_name_grid_pos,
            ['Fraction with \n icreasing trend',
                'Grid fraction (%)'],
            vm,
            no_left_labels=True,
            no_colorbar=True)
    ax[4].set_title(r'$\bf{(c)}$'+'\n ', loc='left')

    scatter_plot(fig, ax[3],
            df_ocean_grid_percent,
            col_name_grid_neg,
            ['Fraction with \n decreasing trend',
                'Grid fraction (%)'],
            vm,
            no_left_labels=True,
            no_colorbar=False)



    plot_each_heatmap(ax[5], df_vals_piv_omf_pol, col_name_omf, cmap_omf_pol,right_label=False)
    ax[5].set_title(r'$\bf{(d)}$'+'\n ', loc='left')
    plot_each_heatmap(ax[6], df_vals_piv_omf, col_name_omf, cmap_omf_pl, no_ylabel=True)
    scatter_plot(fig, ax[9],
            df_omf_grid_percent,
            col_name_signif,
            ['', 'Grid fraction with \n significant trend (%)'],
            25,
            no_left_labels=True,
            no_colorbar=False)
    ax[7].set_title(r'$\bf{(e)}$'+'\n ', loc='left')
    scatter_plot(fig, ax[7],
            df_omf_grid_percent,
            col_name_grid_pos,
            ['Fraction with \n increasing trend',
                'Grid fraction (%)'],
            vm,
            no_left_labels=True,
            no_colorbar=True)
    ax[9].set_title(r'$\bf{(f)}$'+'\n ', loc='left')

    scatter_plot(fig, ax[8],
            df_omf_grid_percent,
            col_name_grid_neg,
            ['Fraction with \n decreasing trend',
                'Grid fraction (%)'],
            vm,
            no_left_labels=True,
            no_colorbar=False)

    plt.tight_layout()
    plt.savefig(f'{season}_heatmap_scatter_plots_grid_fraction_paper.png',dpi = 300)
    plt.close()

















