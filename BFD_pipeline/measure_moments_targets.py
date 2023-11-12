import glob
import copy
import numpy as np
import meds
import ngmix
import frogress
import copy
from .image_processing_routines import CollectionOfImages,MedsStamp,GalaxyModelsTable,check_on_bad_exposures
import bfd
from bfd.momenttable import TargetTable
from bfd.momentcalc import MomentCovariance
import timeit

def measure_moments_targets(**config,):

    
    if config['MPI']:
        from mpi4py import MPI 
    
    if 'flagged_exp_list' not in config.keys():
        config['flagged_exp_list'] = False
        print ('You did not provide any list of flagged exposures; I will set the variable flagged_exp_list to False')
        

    # makes a dictionary of the tiles that need to be run
    dictionary_runs = dict()
    for tile in config['tiles']:
        print (tile)
        dictionary_runs[tile] = dict()
        for band in config['bands_meds_files']:
            dictionary_runs[tile][band] = dict()
            f_ = glob.glob(config['path_MEDS']+'/'+tile+'*fits.fz')
            dictionary_runs[tile][band]['meds'] = np.array(f_)[np.array([((band+'_meds') in ff) for ff in f_])][0]

   
    tiles = list(dictionary_runs.keys())
    
    # Runs the main pipeline **********************************************************
    # We are parallelising with MPI; each process gets a full tile.
    
    run_count = 0
    if not config['MPI']:
        while run_count<len(tiles):
            measure_moments_per_tile(config, dictionary_runs, tiles[run_count])
            run_count+=1
    else:
        while run_count<len(list_run):
            comm = MPI.COMM_WORLD
      
            if (run_count+comm.rank) < len(tiles):
                measure_moments_per_tile(config, dictionary_runs , tiles[run_count+comm.rank])

            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
            
            
        
def measure_moments_per_tile(config,dictionary_runs,tile):
    

    # Read meds and psf files for a given tile ************************************
    meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands_meds_files']]

        
    # Initialise where we're going to store all the images loaded from the MEDS files
    Collection_of_wide_field_galaxies = CollectionOfImages()
    
    for meds_index in range(meds_array[0].size):
        Collection_of_wide_field_galaxies.add_MEDS_stamp(copy.deepcopy(MedsStamp(meds_index, meds_array = meds_array, bands = config['bands_meds_files'])))
        
    
    # Let's load into memory galaxy models. That is needed to do the neighbours subtraction.
    models_path = glob.glob(config['path_galaxy_models']+'/'+tile+'*')
    galaxy_models =  GalaxyModelsTable(models_path[0])
    
    # add models 
    Collection_of_wide_field_galaxies.add_models(galaxy_models)
    

    # if 'debug' == True, it will fill this with the stamps images and models.
    image_storage = dict()
    
    # Initialise 
    tab_targets = TargetTable(n = 4, sigma = config['filter_sigma'])
    
    # Load the list of flagged exposure and check on bad exposures
    if config['flagged_exp_list']:
        bad_exposure_list = np.load(config['exp_list'],allow_pickle=True).item()
        mask_exposure = check_on_bad_exposures(meds_array, bad_exposure_list)
    
    else:
        mask_exposure = None
        
    
        
    timers = dict()
    timers['load MEDS'] = 0.
    timers['make_WCS'] = 0.
    timers['measure_psf_HSM_moments'] = 0.
    timers['check_stamp_masked_frac'] = 0.
    timers['subtract_background'] = 0.
    timers['zero_padd_psf'] = 0.
    timers['compute_noise'] = 0.
    timers['render_models'] = 0.
    timers['deal_with_bmask'] = 0.
    timers['compute_moments'] = 0.
    timers['extra_info'] = 0.
    timers['count'] = 0.
    
    for meds_index in frogress.bar(range(100,200)): #meds_array[0].size)):
        
        
        # Load images, psf, wcs, etc.
        start = timeit.default_timer()
        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].Load_MEDS(meds_index, meds_array = meds_array, mask_exposure = mask_exposure)
        t1 = timeit.default_timer()
        timers['load MEDS'] += (t1-start)
        
        # Make WCS in the BFD format
        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].make_WCS()
        t2 = timeit.default_timer()
        timers['make_WCS'] += (t2-t1)
        
        
        # compute psf HSM moments  
        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].measure_psf_HSM_moments(use_COADD_only = False)
        t3 = timeit.default_timer()
        timers['measure_psf_HSM_moments'] += (t3-t2)
        
        # check masked bands and only consider objects with N bands not masked
        bands_not_masked = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].check_stamp_masked_frac(limit=config['max_frac_stamp_masked'])
        t4 = timeit.default_timer()
        timers['check_stamp_masked_frac'] += (t4-t3)
        
        #subtract background
        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].subtract_background()
        t5 = timeit.default_timer()
        timers['subtract_background'] += (t5-t4)
        
        #Zero-padd PSF
        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].zero_padd_psf()
        t6 = timeit.default_timer()
        timers['zero_padd_psf'] += (t6-t5)
        
        # Compute the noise of the stamp (needed when computing the moments)
        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_noise()
        t7 = timeit.default_timer()
        timers['compute_noise'] += (t7-t6)
        
        # if we have at least one umasked exposure on all the bands we care about, let's proceed with the moments measurement
        if sum([band in bands_not_masked for band in config['bands_meds_files']]) == len(config['bands_meds_files']):
            
            
            # let's render the galaxy models
            Collection_of_wide_field_galaxies.render_models(MEDS_index = meds_index, render_self = False, render_others = True,use_COADD_only = False)
            t8 = timeit.default_timer()
            timers['render_models'] += (t8-t7)
        
            #Interpolates ther images over masked pixels
            Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].deal_with_bmask(use_COADD_only = False)
            t9 = timeit.default_timer()
            timers['deal_with_bmask'] += (t9-t8)
            
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
                                                                 'model_all_rendered':Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].model_all_rendered[index_band]}
                        
            
            # Compute moments
            Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_moments(config['filter_sigma'], 
                                           use_COADD_only =False, 
                                           bands = config['bands_meds_files'],
                                           bands_weights = config['bands_weights'], 
                                           FFT_pad_factor=config['FFT_pad_factor'])
                                                                 
            t10 = timeit.default_timer()
            timers['compute_moments'] += (t10-t9)                                         
            # get moments 
            mom, meb = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments.get_moment(0.,0.,returnbands=True)
            
            
            # get covariances, and get per band flux covariances,  
            covm_even,covm_odd , covm_even_all , _ = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments.get_covariance(returnbands=True)
 
            # get new centers
            newcent=np.array([Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].image_ra, Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].image_dec])[:,0]+ Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].xyshift/3600.0

            
            # add info to the targets table.
            tab_targets.add(mom, xy=newcent,id=Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].coadd_ID[0],covgal=MomentCovariance(covm_even,covm_odd))
            
                                        
  
     
                                
            #Let's add quantities to the targets tab.
            tab_targets.psf_moment.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_moments)
            tab_targets.psf_hsm_moment.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf_hsm_moments)
            tab_targets.DESDM_coadd_y.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].DESDM_coadd_x)
            tab_targets.DESDM_coadd_x.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].DESDM_coadd_x)
            tab_targets.bkg.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].bkg)
            tab_targets.pixel_used_bkg.append(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].pixel_used_bkg)
            tab_targets.cov_Mf_per_band.append(covm_even_all)

            orig_row_flattened,orig_col_flattened, ccd_name_flattened = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].return_orig_coordinates()
            tab_targets.orig_row.append(orig_row_flattened)
            tab_targets.orig_col.append(orig_col_flattened)
            tab_targets.ccd_name.append(ccd_name_flattened)

            t11 = timeit.default_timer()
            timers['extra_info'] += (t11-t10)  
            timers['count'] +=1
            
                             

                                                                 
        # clean MEDS_stamp   
        #del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].imlist
        #del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].seglist
        #del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].wtlist
        #del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].masklist                             
        #del Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf      
        
        

    if config['debug']:
        np.save(config['output_folder']+'/targets/images_for_debugging',image_storage)
                  
    total_time = 0.
    print ('')
    print ('timing')
    for key in timers.keys():
        if key != 'count':
            print (key , '{0:2.4f}s'.format(timers[key]/timers['count']))
            total_time += timers[key]/timers['count']
    print ('time 1 gal: {0:2.2f}s'.format(total_time))


                
