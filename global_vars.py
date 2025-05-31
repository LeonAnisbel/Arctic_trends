root_dir= '/work/bb1005/b381361/'
main_new = '/work/bb1178/b324073/'

aer_dir_path = main_new+'ac3_arctic/ac3_arctic'
omf_dir_path = root_dir+"MOA_DATA_ECHAM/OMF_data/fesom_recon_oceanfilms_no_icemask/fesom_recon_oceanfilms_omf_*"
ocean_dir_path = root_dir+"MOA_DATA_ECHAM/regular_grid_interp/"
lat_arctic_lim = 66

factor_eim_heatmaps = 1.e6
factor_sic_heatmaps = 1.
data_type = 'orig_data'#'log_data'#
season_to_analise = 'JAS'
seasons_info = {'JFM': {'months' : [1, 2, 3],
                        'one_month': [1],
                        'long_name': 'January-February-March'},
                'AMJ': {'months' : [4, 5, 6],
                        'one_month': [6],
                        'long_name': 'April-May-June',
                        'bar_plot_lims': [[-3.5, 4.8, 1.5], [-3.5, 3.5, 1]]},
                'JAS': {'months': [7, 8, 9],
                        'one_month': [9],
                        'long_name': 'July-August-September',
                        'bar_plot_lims': [[-3.5, 4.5, 1.5], [-1, 3.5, 1]]},
                'OND': {'months' : [10, 11, 12],
                        'one_month': [10],
                        'long_name': 'October-November-December'}
                }
log_data=True
colors_arctic_reg = ['k', 'r', 'm', 'pink', 'lightgreen', 'darkblue', 'orange',
                 'brown', 'lightblue', 'y', 'gray']
