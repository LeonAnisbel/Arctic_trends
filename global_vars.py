root_dir= '/work/bb1005/b381361/'
aer_dir_path = root_dir+'my_experiments/ac3_arctic/ac3_arctic'
omf_dir_path = root_dir+"MOA_DATA_ECHAM/OMF_data/fesom_recon_oceanfilms_no_icemask/fesom_recon_oceanfilms_omf_*"
ocean_dir_path = root_dir+"MOA_DATA_ECHAM/regular_grid_interp/"
factor_eim_heatmaps = 1.e7
factor_sic_heatmaps = 1.
season_to_analise = 'AMJ'
seasons_info = {'JFM': {'months' : [1, 2, 3],
                        'one_month': [1],
                        'long_name': 'January-February-March'},
                'AMJ': {'months' : [4, 5, 6],
                        'one_month': [6],
                        'long_name': 'April-May-June'},
                'JAS': {'months': [7, 8, 9],
                        'one_month': [9],
                        'long_name': 'July-August-September'},
                'OND': {'months' : [10, 11, 12],
                        'one_month': [10],
                        'long_name': 'October-November-December'}
                }
