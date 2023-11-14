import glob
import copy
import numpy as np
import meds
import ngmix
import frogress
from .image_processing_routines import CollectionOfImages,MedsStamp,GalaxyModelsTable,check_on_exposures,save_moments_targets,collapse
import bfd
from bfd.momenttable import TargetTable
from bfd.momentcalc import MomentCovariance
import timeit
import time
import gc
import astropy.io.fits as fits
import math
import multiprocessing
from functools import partial 
import os
from galsim.utilities import single_threaded
  

def run_chunk(chunk,config, tile, dictionary_runs):
    
    

    if not os.path.exists(config['output_folder']+'/targets/targets_{0}_chunk_{1}.fits'.format(tile,chunk)):

        # Initialise where we're going to store all the images loaded from the MEDS files
        Collection_of_wide_field_galaxies = CollectionOfImages()

        print ('Initialising MEDS stamps')
        meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands_meds_files']]

        start =  config['chunk_size']*chunk
        end = int(np.min([config['chunk_size']*(chunk+1),meds_array[0].size]))

        for meds_index in frogress.bar(range(meds_array[0].size)):
            Collection_of_wide_field_galaxies.add_MEDS_stamp(MedsStamp(meds_index, meds_array = meds_array, bands = config['bands_meds_files']))

        # let's check if we want to save the observed moments of the reserved stars as well
        if config['reserved_stars_list']:
            reserved_stars_id = np.load('/global/u2/m/mgatti/BFD_pipeline/data/coadd_id_reserved_stars.npy',allow_pickle=True)
            Collection_of_wide_field_galaxies.reserved_star_flag = np.in1d( np.array(Collection_of_wide_field_galaxies.coadd_IDs)  , reserved_stars_id)
        else:
            Collection_of_wide_field_galaxies.reserved_star_flag = np.array([False]*meds_array[0].size)


        # Let's load into memory galaxy models. That is needed to do the neighbours subtraction.
        models_path = glob.glob(config['path_galaxy_models']+'/'+tile+'*')
        galaxy_models =  GalaxyModelsTable(models_path[0])

        # add models 
        Collection_of_wide_field_galaxies.add_models(galaxy_models)


        # if 'debug' == True, it will fill this with the stamps images and models.
        image_storage = dict()

        # Initialise 

        # Load the list of flagged exposure and check on bad exposures
        if config['flagged_exp_list']:
            exposure_list = np.load(config['flagged_exp_list'],allow_pickle=True).item()
            mask_exposure = check_on_exposures(meds_array, exposure_list,config['bands_meds_files'])

        else:
            mask_exposure = None

        
        timers = dict()
        timers['load MEDS'] = []
        timers['make_WCS'] = []
        timers['measure_psf_HSM_moments'] = []
        timers['check_stamp_masked_frac'] = []
        timers['subtract_background'] = []
        timers['zero_padd_psf'] = []
        timers['compute_noise'] = []
        timers['render_models'] = []
        timers['deal_with_bmask'] = []
        timers['compute_moments'] = []
        timers['extra_info'] = []
        timers['count'] = 0.

        print ('Looping over targets')


        tab_targets = TargetTable(n = 4, sigma = config['filter_sigma'])


        for meds_index in frogress.bar(range(start,end)):


            # Load images, psf, wcs, etc.
            start = timeit.default_timer()
            Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].Load_MEDS(meds_index, meds_array = meds_array, mask_exposure = mask_exposure)
            t1 = timeit.default_timer()
            timers['load MEDS'].append (t1-start)



            # check masked bands and only consider objects with N bands not masked
            bands_not_masked = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].check_stamp_masked_frac(limit=config['max_frac_stamp_masked'])
            t2 = timeit.default_timer()
            timers['check_stamp_masked_frac'] .append (t2-t1)

            # if we have at least one umasked exposure on all the bands we care about, let's proceed with the moments measurement
            if sum([band in bands_not_masked for band in config['bands_meds_files']]) == len(config['bands_meds_files']):


                # Make WCS in the BFD format
                Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].make_WCS()
                t3 = timeit.default_timer()
                timers['make_WCS'].append(t3-t2)

                # let's render the galaxy models
                bands_with_a_model = Collection_of_wide_field_galaxies.render_models(MEDS_index = meds_index, render_self = False, render_others = True,use_COADD_only = False)
                t4 = timeit.default_timer()
                timers['render_models'].append(t4-t3)

                if sum([band in bands_with_a_model for band in config['bands_meds_files']]) == len(config['bands_meds_files']):

                    
                    # compute psf HSM moments  
                    Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].measure_psf_HSM_moments(config['bands_weights'],use_COADD_only = False)
                    t5 = timeit.default_timer()
                    timers['measure_psf_HSM_moments'].append(t5-t4)


                    #Zero-padd PSF
                    Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].zero_padd_psf()
                    t6 = timeit.default_timer()
                    timers['zero_padd_psf'].append (t6-t5)

                    # Compute the noise of the stamp (needed when computing the moments)
                    Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_noise()
                    t7 = timeit.default_timer()
                    timers['compute_noise'].append(t7-t6)
                
                
                    #subtract background
                    Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].subtract_background()
                    t8 = timeit.default_timer()
                    timers['subtract_background'].append(t8-t7)

                    #Interpolates ther images over masked pixels
                    Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].deal_with_bmask(use_COADD_only = False)
                    t9 = timeit.default_timer()
                    timers['deal_with_bmask'].append(t9-t8)

                    if config['debug']:
                        image_storage[meds_index] = dict()
                        for index_band in range(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].n_bands):
                            start = 1
                            end = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].ncutout[index_band]
                            image_storage[meds_index][index_band] = dict()
                            for exp in range(start, end):  
                                image_storage[meds_index][index_band][exp] = {'image':Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].imlist[index_band][exp],
                                                                         'psf':Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf[index_band][exp],
                                                                         'model_rendered':Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].model_rendered[index_band][exp],
                                                                         'model_all_rendered':Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].model_all_rendered[index_band][exp]}


                                
                    # check again if there're enough exposures after mask+exp+model checks - 
                    number_of_good_exposures_per_band,number_of_bad_exposures_per_band = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].number_of_good_exposures_per_band(use_COADD_only = False)
             
                    if len(number_of_good_exposures_per_band[number_of_good_exposures_per_band>0.5]) == len(config['bands_meds_files']):
               
                    
                        # Compute moments
                        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_moments(config['filter_sigma'], 
                                                       use_COADD_only =False, 
                                                       bands = config['bands_meds_files'],
                                                       bands_weights = config['bands_weights'], 
                                                       FFT_pad_factor=config['FFT_pad_factor'])



                        # if the stamp is a stamp of a reserved stars, let's also compute the observed moments without PSF deconvolution.

                        if Collection_of_wide_field_galaxies.reserved_star_flag[meds_index]:
                            Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_moments_observed_psf(config['filter_sigma'], 
                                                       use_COADD_only =False, 
                                                       bands = config['bands_meds_files'],
                                                       bands_weights = config['bands_weights'], 
                                                       FFT_pad_factor=config['FFT_pad_factor'])
                            Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].measure_psf_HSM_moments(config['bands_weights'],use_COADD_only = False,do_it_for_the_image = True)
                    
                            
                                    

    
        

                        t10 = timeit.default_timer()
                        timers['compute_moments'].append  (t10-t9)                                         
                        # get moments 
                        mom, meb = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments.get_moment(0.,0.,returnbands=True)

                        # re-checking this because we're paranoic
                        if len(meb) == len(config['bands_meds_files']):
                         
                            # get covariances, and get per band flux covariances,  
                            covm_even,covm_odd , covm_even_all , _ = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments.get_covariance(returnbands=True)
                            meb_ = np.array([m_.even for m_ in meb])
                            meb = meb_[:,0]
                            # get new centers
                            newcent=np.array([Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].image_ra, Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].image_dec])[:,0]+ Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].xyshift/3600.0


                            # add info to the targets table.
                            tab_targets.add(mom, xy=newcent,id=Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].coadd_ID[0],covgal=MomentCovariance(covm_even,covm_odd))



                            #Let's add quantities to the targets tab.
                            tab_targets.psf_moment.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_moments)
                            tab_targets.psf_hsm_moment.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_hsm_moments)
                            tab_targets.DESDM_coadd_y.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].DESDM_coadd_y[0])
                            tab_targets.DESDM_coadd_x.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].DESDM_coadd_x[0])
                            tab_targets.bkg.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].bkg)
                            tab_targets.pixel_used_bkg.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].pixel_used_bkg)
                            tab_targets.cov_Mf_per_band.append(covm_even_all[0][0,0])
                            tab_targets.meb.append(meb)

                            orig_row_flattened,orig_col_flattened, ccd_name_flattened = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].return_orig_coordinates()
                            tab_targets.orig_row.append(orig_row_flattened)
                            tab_targets.orig_col.append(orig_col_flattened)
                            tab_targets.ccd_name.append(ccd_name_flattened)
                            tab_targets.mfrac_per_band.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].mfrac_per_band)

                            tab_targets.bad_exposures.append(number_of_bad_exposures_per_band)
                            tab_targets.good_exposures.append(number_of_good_exposures_per_band)



                            try:
                                tab_targets.psf_moment_obs.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_moments_observed)
                            except:
                                tab_targets.psf_moment_obs.append(np.zeros(4))
                                
                            try:
                                tab_targets.psf_hsm_moments_obs.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_hsm_moments_obs)
                            except:
                                tab_targets.psf_hsm_moments_obs.append(np.zeros(3))
                  
                                
                            t11 = timeit.default_timer()
                            timers['extra_info'].append (t11-t10)  
                            timers['count'] +=1

                             # clean MEDS_stamp   
                            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].model_rendered  
                            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments       
                            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_hsm_moments  
                            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_moments      
             
                        
            # clean MEDS_stamp   
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].imlist
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].seglist
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].wtlist
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].masklist                             
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf      
   
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].DESDM_coadd_y           
            del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].DESDM_coadd_x           

            if meds_index%100 == 0:
                gc.collect()






        if chunk == 0:
            total_time = 0.
            print ('')
            print ('timing')
            for key in timers.keys():
                if key != 'count':
                    timers[key] = np.array(timers[key])
                    print (key , '{0:2.4f}s  [{1:2.4f}s]'.format(np.mean(timers[key] ), np.std(timers[key] )))
                    total_time += np.mean(timers[key])
            print ('time 1 gal: {0:2.2f}s'.format(total_time))


        save_moments_targets(tab_targets,config['output_folder']+'/targets/targets_{0}_chunk_{1}.fits'.format(tile,chunk))



    if config['debug']:
        np.save(config['output_folder']+'/targets/images_for_debugging_{0}_chunk_{1}'.format(tile,chunk),image_storage)

            

def measure_moments_targets(**config,):

    
    if config['MPI']:
        from mpi4py import MPI 
    
    if 'flagged_exp_list' not in config.keys():
        config['flagged_exp_list'] = False
        print ('You did not provide any list of flagged exposures; I will set the variable flagged_exp_list to False')
        
    if 'reserved_stars_list' not in config.keys():
        config['reserved_stars_list'] = False
        print ('You did not provide any list of reserved stars; I will set the variable reserved_stars_list to False')
        
        
    # makes a dictionary of the tiles that need to be run
    dictionary_runs = dict()
    for tile in config['tiles']:
        dictionary_runs[tile] = dict()
        for band in config['bands_meds_files']:
            dictionary_runs[tile][band] = dict()
            f_ = glob.glob(config['path_MEDS']+'/'+tile+'*fits.fz')
            dictionary_runs[tile][band]['meds'] = np.array(f_)[np.array([((band+'_meds') in ff) for ff in f_])][0]

   
    tiles = list(dictionary_runs.keys())
    print ('TILES TO RUN: ',len(tiles))
    # Runs the main pipeline **********************************************************
    # We are parallelising with MPI; each process gets a full tile.
    
    run_count = 0
    if not config['MPI']:
        while run_count<len(tiles):
            measure_moments_per_tile(config, dictionary_runs, tiles[run_count])
            run_count+=1
    else:
        while run_count<len(tiles):
            comm = MPI.COMM_WORLD
      
            if (run_count+comm.rank) < len(tiles):
                measure_moments_per_tile(config, dictionary_runs , tiles[run_count+comm.rank])

            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 

     
def run_chunk_single_threaded(chunk,config, tile, dictionary_runs):
    with single_threaded(num_threads=1):
        run_chunk(chunk,config = config, tile = tile, dictionary_runs = dictionary_runs)
            
            
        
def measure_moments_per_tile(config,dictionary_runs,tile):
    

    # Read meds and psf files for a given tile ************************************
    meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands_meds_files']]

        
    
    
    chunk_size = config['chunk_size']
    runs = math.ceil(meds_array[0].size/chunk_size)
    xlist = range(runs)
    config
    print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)

    # check if the file exists, otherwise skip this.
    
    path = config['output_folder']+'/targets/targets_'+tile
    if not os.path.exists(path+config['output_label']+'.fits'):
        if config['agents'] == 1:
            for chunk in xlist:
                run_chunk(chunk, config, tile, dictionary_runs)
        else:
            
            
            pool = multiprocessing.Pool(processes=config['agents'])
            _ = pool.map(partial(run_chunk_single_threaded, config = config, tile = tile, dictionary_runs = dictionary_runs), xlist)

        
    # collapse all the files into one
    # --
        time.sleep(10) # just make sure that IO operations from previous step are done
        collapse(path, config['output_label'])
        


