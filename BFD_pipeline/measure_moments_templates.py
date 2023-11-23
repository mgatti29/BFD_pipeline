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
import h5py as h5  


            

            
            
            
def measure_moments_templates(**config):
    if config['MPI']:
        from mpi4py import MPI 
    '''
    It computes the moments from des y6 deef fields tiles.
    '''

    dictionary_runs = dict()
    for field in config['fields']:
        folders = glob.glob(config['path_MEDS']+field+'/*/')

  

        for folder in folders:

            tile = (folder.split(config['path_MEDS']+field+'/')[1]).split('/')[0]
            if config['tiles'] == 'All':
                dictionary_runs[tile] = dict()
                for band in config['bands_meds_files']:
                    dictionary_runs[tile][band] = dict()
                    dictionary_runs[tile][band]['meds'] = glob.glob(folder+'/*{0}_meds*'.format(band))
            else:
                if tile in config['tiles']:
                    dictionary_runs[tile] = dict()
                    for band in config['bands_meds_files']:
                        dictionary_runs[tile][band] = dict()
                        dictionary_runs[tile][band]['meds'] = glob.glob(folder+'/*{0}_meds*'.format(band))

                
 
    tiles = list(dictionary_runs.keys())
    print ('TILES TO RUN: ',len(tiles))

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



            
def measure_moments_per_tile(config, dictionary_runs, tile) : 
# Initialise where we're going to store all the images loaded from the MEDS files

        print (config['output_folder'])
        Collection_of_wide_field_galaxies = CollectionOfImages()

        print ('Initialising MEDS stamps')
        meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds'][0]) for band in config['bands_meds_files']]
        psf_array = [ fits.open(dictionary_runs[tile][band]['meds'][0])['psf'].data for band in config['bands_meds_files']]

        print([(dictionary_runs[tile][band]['meds'][0]) for band in config['bands_meds_files']])

        for meds_index in frogress.bar(range(meds_array[0].size)):
            Collection_of_wide_field_galaxies.add_MEDS_stamp(MedsStamp(meds_index, meds_array = meds_array, bands = config['bands_meds_files']))

        
        #we don't check reserved stars here
        Collection_of_wide_field_galaxies.reserved_star_flag = np.array([False]*meds_array[0].size)


        # Let's load into memory galaxy models. That is needed to do the neighbours subtraction.
        models_path = glob.glob(config['path_galaxy_models']+'/'+tile+'*')
        galaxy_models =  GalaxyModelsTable(models_path[0])

        # add models 
        Collection_of_wide_field_galaxies.add_models(galaxy_models)


        # if 'debug' == True, it will fill this with the stamps images and models.
        image_storage = dict()

        # Initialise 

        # we don't have masked exposures in the deep fields
        mask_exposure = None

        
        # let's match the IDs with the DF catalog 
        desy3_DF = h5.File(config['desy3_DF'],'r')
        dd = np.array(desy3_DF['data']['table'])
        mask_df = (dd['MASK_FLAGS']==0) & (dd['FLAGS']==0) & (dd['FLAGS_NIR']==0)& (dd['MASK_FLAGS_NIR']==0)  & (dd['KNN_CLASS']==1)
        good_IDs = dd['ID'][mask_df]
        Collection_of_wide_field_galaxies.mask_DF = np.in1d(Collection_of_wide_field_galaxies.coadd_IDs,good_IDs)
        
        

        timers = dict()
        timers['load MEDS'] = []
        timers['make_WCS'] = []
        #timers['measure_psf_HSM_moments'] = []
        timers['check_stamp_masked_frac'] = []
        timers['subtract_background'] = []
        timers['zero_padd_psf'] = []
        timers['compute_noise'] = []
        timers['render_models'] = []
        timers['deal_with_bmask'] = []
        timers['compute_moments'] = []
        timers['extra_info'] = []
        timers['count'] = 0.

        print ('Looping over templates')
        
        template_moments_container = dict()
        template_moments_info = dict()
        

        for meds_index in frogress.bar(range(meds_array[0].size)):
            if  Collection_of_wide_field_galaxies.mask_DF[meds_index]:


            
                # Load images, psf, wcs, etc.
                start = timeit.default_timer()
                Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].Load_MEDS(meds_index, meds_array = meds_array, mask_exposure = mask_exposure, psf_array = psf_array)
                t1 = timeit.default_timer()                
                timers['load MEDS'].append (t1-start)



                # check masked bands and only consider objects with N bands not masked
                bands_not_masked = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].check_stamp_masked_frac(limit=config['max_frac_stamp_masked'],use_COADD_only=True)
                t2 = timeit.default_timer()
                timers['check_stamp_masked_frac'] .append (t2-t1)

                # if we have at least one umasked exposure on all the bands we care about, let's proceed with the moments measurement
                if sum([band in bands_not_masked for band in config['bands_meds_files']]) == len(config['bands_meds_files']):


                    # Make WCS in the BFD format
                    Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].make_WCS()
                    t3 = timeit.default_timer()
                    timers['make_WCS'].append(t3-t2)

                    # let's render the galaxy models
                    bands_with_a_model = Collection_of_wide_field_galaxies.render_models(MEDS_index = meds_index, render_self = False, render_others = True,use_COADD_only = True)
                    t4 = timeit.default_timer()
                    timers['render_models'].append(t4-t3)

                    if sum([band in bands_with_a_model for band in config['bands_meds_files']]) == len(config['bands_meds_files']):


                        # compute psf HSM moments  
                        #Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].measure_psf_HSM_moments(config['bands_weights'],use_COADD_only = True)
                        t5 = timeit.default_timer()
                        #timers['measure_psf_HSM_moments'].append(t5-t4)


                        #Zero-padd PSF
                        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].zero_padd_psf()
                        t6 = timeit.default_timer()
                        timers['zero_padd_psf'].append (t6-t5)

                        if config['debug']:
                            image_storage[meds_index] = dict()
                            for index_band in range(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].n_bands):
                                exp = 0
                                image_storage[meds_index][index_band] = dict()
                                image_storage[meds_index][index_band][exp] = {'raw_image':copy.deepcopy(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].imlist[index_band][exp]),
                                                                             'psf':Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].psf[index_band][exp]}


                                
                                
                        # Compute the noise of the stamp (needed when computing the moments)
                        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_noise()
                        t7 = timeit.default_timer()
                        timers['compute_noise'].append(t7-t6)


                        #subtract background
                        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].subtract_background(use_COADD_only = True)
                        t8 = timeit.default_timer()
                        timers['subtract_background'].append(t8-t7)

                        #Interpolates ther images over masked pixels
                        Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].deal_with_bmask(use_COADD_only = True)
                        t9 = timeit.default_timer()
                        timers['deal_with_bmask'].append(t9-t8)



                        # check again if there're enough exposures after mask+exp+model checks - 
                        number_of_good_exposures_per_band,number_of_bad_exposures_per_band = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].number_of_good_exposures_per_band(use_COADD_only = True)



                        if len(number_of_good_exposures_per_band[number_of_good_exposures_per_band>0.5]) == len(config['bands_meds_files']):



                            if config['debug']:

                                for index_band in range(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].n_bands):
                                    exp = 0
                                    
                                    image_storage[meds_index][index_band][exp]['image_after_processing'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].imlist[index_band][exp]
                                    image_storage[meds_index][index_band][exp]['model_rendered'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].model_rendered[index_band][exp]
                                    image_storage[meds_index][index_band][exp]['model_all_rendered'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].model_all_rendered[index_band][exp]
                                    image_storage[meds_index][index_band][exp]['mask'] =copy.deepcopy(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].masklist[index_band][exp])
                                    image_storage[meds_index][index_band][exp]['mfrac'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].mfrac_per_band
                                    image_storage[meds_index][index_band][exp]['bkg'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].bkg

                            # Compute moments
                            try:
                                t9 = timeit.default_timer()
                                
                                Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].compute_moments(config['filter_sigma'], 
                                                               use_COADD_only =True, 
                                                               bands = config['bands_meds_files'],
                                                               bands_weights = config['bands_weights'], 
                                                               FFT_pad_factor=config['FFT_pad_factor'])

                                t10 = timeit.default_timer()
                                timers['compute_moments'].append(t10-t9)
                                
                                mom, meb = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments.get_moment(0.,0.,returnbands=True)
                                # get covariances, and get per band flux covariances,  
                                covm_even,covm_odd , covm_even_all , _ = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments.get_covariance(returnbands=True)
                                meb_ = np.array([m_.even for m_ in meb])
                                meb = meb_[:,0]
                                
                                
                                
                                 # check on band fluxes:
                                if np.sum(meb/np.sqrt( covm_even_all[0,0,:]) > -4.):
                                    
                                    #print (mom.even)
                                    coadd_id = Collection_of_wide_field_galaxies.coadd_IDs[meds_index]
                                    template_moments_container[coadd_id] = dict()
                                    template_moments_container[coadd_id]['moments'] = copy.deepcopy(Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].moments)


                                    # save properties of the templates -----
                                    template_moments_info[coadd_id] = dict()
                                    template_moments_info[coadd_id]['meds_index'] = meds_index
                                    template_moments_info[coadd_id]['moments'] = mom.even
                                    template_moments_info[coadd_id]['covariance'] =  MomentCovariance(covm_even,covm_odd).pack()
                                    template_moments_info[coadd_id]['bkg'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].bkg
                                    template_moments_info[coadd_id]['pixel_used_bkg'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].pixel_used_bkg
                                    template_moments_info[coadd_id]['cov_Mf_band'] = covm_even_all[0,0,:]
                                    template_moments_info[coadd_id]['Mf_band'] = meb
                                    template_moments_info[coadd_id]['mfrac_band'] = Collection_of_wide_field_galaxies.MEDS_stamps[meds_index].mfrac_per_band



                                t11 = timeit.default_timer()
                                timers['extra_info'].append(t11-t10)             
                                

                                
                            except:
                                print ('failed ',meds_index)
                            
                            '''
                             
                            list of current cuts.
                            
                            exposures cuts:
                            - if an exposure has a masked fraction larger than 0.05, exclude the exposure.
                            - if an exposure had problems rendering one of the models of the neighbour galaxies, exclude it (some of the galaxies don't have a model
                            because the sof fitting failed)
                            - if an exposure has been flagged as problematic (either because the PSF was too large or was bad because too few stars), exclude it. Only wide field
                            - if we couldn't estimate the background of a stamp (because there were not enough pixels free from masks of neighbouring objects), exclude the exposure. 
                            
                            
                            - if a galaxy doesn't have a good exposure in all bands (griz), bacause, e.g., of masking, skip it.
                            - if, when computing the galaxy moments, the recentering fails, exclude the galaxy 
                            - if a galaxy has one of the band fluxes negative, and < 5 sigma, exclude it. Visually this is usually do to over subtraction of a neighbouring galaxy.
                            
                            
                            
                                                
                            '''



        total_time = 0.
        print ('')
        print ('timing')
        for key in timers.keys():
            if key != 'count':
                timers[key] = np.array(timers[key])
                print (key , '{0:2.4f}s  [{1:2.4f}s]'.format(np.mean(timers[key] ), np.std(timers[key] )))
                total_time += np.mean(timers[key])
        print ('time 1 gal: {0:2.2f}s'.format(total_time))
        if config['debug']:
            #image_storage['sn_array'] = sn_array
            np.save(config['output_folder']+'/templates/images_for_debugging_{0}'.format(tile),image_storage)

        np.save(config['output_folder']+'/templates/moments_templates_info_{0}'.format(tile),template_moments_info)

        np.save(config['output_folder']+'/templates/template_moments_container_{0}'.format(tile),template_moments_container)

        