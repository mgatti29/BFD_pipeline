from .read_meds_utils import MOF_table
from .utilities import update_progress, save_obj, load_obj
import pyfits as pf
import numpy as np
import timeit
import pickle
import sys
import pandas as pd
import os
import math

def setup_image_sims(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    if config['stage'] == 'save_galaxy_params':
        # This stages reads PSF and galaxy models from deep fields MOF fits, and saves the parameters into pickle files,
        # such that they're ready to be used to generate galaxies at the next stage.
        MOF_deep_field = MOF_table(config['MOF_table_path'])

        # the following lines are needed to match the ID (relative to galaxy models) with the epoch_ID (PSF)
        idd = MOF_deep_field.id_array
        index_to_match = np.arange(len(MOF_deep_field.id_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = MOF_deep_field.id_array)
        df2 = pd.DataFrame(index = idd)        
        pos = np.array(df2.join(df1).loc[idd,'pos'])

        index_to_match = np.arange(len(MOF_deep_field.id_epoch_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = MOF_deep_field.id_epoch_array)
        pos_epoch = (df2.join(df1))
        pos_epoch = pos_epoch.loc[idd]


        if not os.path.exists(output_folder+'/gal_psf_params/'):
            try:
                os.mkdir(output_folder+'/gal_psf_params/')
            except:
                pass
            
        
        len_file = len(pos)
        runs = math.ceil(len_file/50000)
        run_count = 0
        comm = MPI.COMM_WORLD
        while run_count<runs:
            if run_count+comm.rank<runs:
                min_c = (run_count+comm.rank)*50000
                max_c = (run_count+comm.rank)*50000+50000
                gal_psf_dict = dict()
                print (min_c,max_c)
                for jj, pos_u  in enumerate(pos):
                    if ((jj>=min_c) & (jj<max_c)):
                
                        try:
                            ID = idd[jj]
                            try:
                                pos_epoch_u = (np.array(pos_epoch.loc[ID]).astype(np.int))[:,0]
                            except:
                                pos_epoch_u = (np.array(pos_epoch.loc[ID]).astype(np.int))[0]

                            psf_pars = MOF_deep_field.pfs_params[pos_epoch_u]


                            
                            indexes_band = [MOF_deep_field.return_band_val(band,indtostr=False) for band in config['bands']]
                            gal_psf_info = dict()
                            for ii, index_band in enumerate(indexes_band):
                                mute_band = dict()

                                gal_pars = MOF_deep_field.bdf_params[pos_u][0:6]

                                # last params is the flux in a given band ******************
                                # (7,) params for the galaxy model
                                gal_pars = np.append(gal_pars,MOF_deep_field.bdf_flux[pos_u][index_band])

                                mute_band['gal_pars'] = gal_pars
                                # psf params ***********************************************
                                # 5 (bands) x 30 
                                pos_epoch_uu = pos_epoch_u[index_band]
                                pfs_params = MOF_deep_field.pfs_params[pos_epoch_uu]
                                mute_band['pfs_params'] = pfs_params
                                gal_psf_info[config['bands'][ii]] = mute_band
                                mute_band['des_id'] = MOF_deep_field.des_id[pos_u]
                                mute_band['photoz'] = MOF_deep_field.photoz[pos_u]
                                print (MOF_deep_field.photoz[pos_u])
                                mute_band['mag_i'] = MOF_deep_field.mag_i[pos_u]
                           
                            end = timeit.default_timer()
                            gal_psf_dict[ID] = gal_psf_info
                        except:
                            pass
                        
                    
                #here save the chunk
                np.save(output_folder+'/gal_psf_params/chunk_'+str(run_count+comm.rank)+'.npy', gal_psf_dict) 
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
                
