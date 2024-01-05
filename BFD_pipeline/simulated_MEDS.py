import frogress
from .image_processing_routines import CollectionOfImages,MedsStamp,GalaxyModelsTable,check_on_exposures,assign_efficiency,save_moments_targets,collapse,grid_search
import ngmix
import os
import glob
import numpy as np
import meds
import astropy.io.fits as fits
import copy
import bfd
from bfd import momentcalc as mc
import multiprocessing
from functools import partial 
import os
from galsim.utilities import single_threaded
import math
import frogress

def run_chunk(chunk,config, tile, dictionary_runs,Collection_of_wide_field_galaxies,DF= False):
    meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands_meds_files']]

    start_index =  config['chunk_size']*chunk
    end_index = int(np.min([config['chunk_size']*(chunk+1),meds_array[0].size]))


    #    
    # load models ----
    if Collection_of_wide_field_galaxies is None:
        Collection_of_wide_field_galaxies = CollectionOfImages()
        for meds_index in frogress.bar(range(meds_array[0].size)):
            Collection_of_wide_field_galaxies.add_MEDS_stamp(MedsStamp(meds_index, meds_array = meds_array, bands = config['bands_meds_files']))

        # Let's load into memory galaxy models. That is needed to do the neighbours subtraction.
        models_path = glob.glob(config['path_galaxy_models']+'/'+tile+'*')
        galaxy_models =  GalaxyModelsTable(models_path[0])

        # add models 
        Collection_of_wide_field_galaxies.add_models(galaxy_models)



    
    w_array = dict()
    img_array = dict()
    img_array_m =  dict()
    psf_array = dict()
    seg_array = dict()
    len_images = dict()
    len_psf = dict()
    
    for b,band in enumerate(config['bands_meds_files']):
        len_images[b] = 0
        len_psf[b] = 0
        
    for index in frogress.bar(range(start_index,end_index)):
        for b,band in enumerate(config['bands_meds_files']) :
            for exp in range((meds_array[b]['ncutout'][index])):
                len_images[b] += meds_array[b][index]['box_size']**2
                len_psf[b] +=  meds_array[b]['psf_col_size'][index][exp]**2
                
    
    for b,band in enumerate(config['bands_meds_files']) :    
        img_array[band] = np.zeros(len_images[b])
        img_array_m[band] =  np.zeros(len_images[b])
        psf_array[band] = np.zeros(len_psf[b])
        seg_array[band] = np.zeros(len_images[b]).astype(np.int)
        w_array[band] = np.zeros(len_images[b])
        
        
    # super test  ------------------------------------------------------------------------
    #'''
    Collection_of_wide_field_galaxies_this_tile = CollectionOfImages()
    for meds_index in frogress.bar(range(meds_array[0].size)):
        Collection_of_wide_field_galaxies_this_tile.add_MEDS_stamp(MedsStamp(meds_index, meds_array = meds_array, bands = config['bands_meds_files']))

     
    # load something already produced

    for b,band in enumerate(config['bands_meds_files']):
        len_images[b] = 0
        len_psf[b] = 0
    images_test = dict()
    for index in frogress.bar(range(start_index,end_index)):
        

        twotheta = 2.0 * np.radians(np.random.randint(0,36000)*1./100.)
        cos2angle = np.cos(twotheta)
        sin2angle = np.sin(twotheta)


        images_test[index] = dict()
        
        moments_images = []
        moments_psf = []
        moments_wcs = []
        moments_noise = []
        moments_bands = []
        
        for b,band in enumerate(config['bands_meds_files']) :
            images_test[index][band] = dict()
            size_stamp = meds_array[b][index]['box_size']
            wcs = meds_array[b].get_jacobian_list(index)
            wlist = meds_array[b].get_cutout_list(index, type='weight')
            for exp in range((meds_array[b]['ncutout'][index])):

                start = meds_array[b]['start_row'][index][exp]
                psf_start = meds_array[b]['psf_start_row'][index][exp]
                size_psf_col = meds_array[b]['psf_col_size'][index][exp]
                size_psf_row = meds_array[b]['psf_row_size'][index][exp]


                # make bfd wcs ------------------------------------------------


                # make bfd wcs
                cent=(wcs[exp]['col0'],wcs[exp]['row0'])
                origin = (0.,0.)
                duv_dxy = np.array( [ [wcs[exp]['dudcol'], wcs[exp]['dudrow']],
                                  [wcs[exp]['dvdcol'], wcs[exp]['dvdrow']] ])

                wcs_BFD = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)


                # convert it to ngmix jacobian -------------------------------

                
                jac = ngmix.jacobian.Jacobian(row=wcs_BFD.xy0[1],
                                          col=wcs_BFD.xy0[0],
                                        dudrow=wcs[exp]['dudrow'],
                                        dudcol=wcs[exp]['dudcol'],
                                        dvdrow=wcs[exp]['dvdrow'],
                                        dvdcol=wcs[exp]['dvdcol'])


                det  = np.abs(wcs_BFD.getdet()) 




                pars_ = Collection_of_wide_field_galaxies.MEDS_stamps[config['one_galaxy']].model_parameters[band]['gal_pars']

                # apply the rotation
                #mm1= pars_[2] * cos2angle +  pars_[3] * sin2angle
                #mm2= -pars_[2] * sin2angle + pars_[3] * cos2angle
                #pars_[2] = copy.deepcopy(mm1)
                #pars_[3] = copy.deepcopy(mm2)
#

                # psf
    
                psf_fwhm = config['turb'][0]+(np.random.random(1)*config['turb'][1])[0]
                Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)
                psf_pars_ = [0.0, 0.0, 0.0, 0.0, Tpsf, 1.0]
                psf_gmix = ngmix.GMixModel(pars=psf_pars_, model="turb")




                gmix_sky  = ngmix.gmix.GMixBDF(pars_)
                if DF:
                    gmix_image = gmix_sky.convolve(psf_gmix)
                    image = det*gmix_image.make_image((size_stamp, size_stamp), jacobian=jac)
            
                else:
                    #gmix_sky = gmix_sky.get_sheared(0.2, 0.)
                    gmix_image = gmix_sky.convolve(psf_gmix)
                    image = det*gmix_image.make_image((size_stamp, size_stamp), jacobian=jac)

                seg = 0*(np.zeros_like(image)).astype(np.int)
                seg[image>0.1*max(image.flatten())] = index + 1

                gmix_sky  = ngmix.gmix.GMixBDF(pars_)
                #gmix_sky = gmix_sky.get_sheared(-0.2, 0.)
                gmix_image = gmix_sky.convolve(psf_gmix)
                image_m = det*gmix_image.make_image((size_stamp, size_stamp), jacobian=jac)


                
                # Add noise based on wtlist -----------------------
                rnd_ = np.random.normal(size = (size_stamp,size_stamp))
                ave_noise = np.sqrt(np.median(wlist[exp][wlist[exp]>0.]))
                
                # if for some reasons noise can't be estimated, set to 0 for DF galaxies and infty for wide field galaxies.
                if ave_noise != ave_noise:
                    if DF:
                        ave_noise = 1e20
                    else:    
                        ave_noise = 1e-20
                if ave_noise == 0:
                    if DF:
                        ave_noise = 1e20
                    else:
                        ave_noise = 1e-20
                rnd_ =  rnd_/ave_noise
      

                #image_m += rnd_
                #image  += rnd_

                
                # Add background ----------------------------------


                # if masked, set the pixels to + infty ------------


                # ----------------------------

               # print ('')
               # size_psf_col = copy.deepcopy(size_stamp)
                image_psf = det*psf_gmix.make_image((size_stamp,size_stamp), jacobian=jac)

                if (size_stamp>size_psf_col):
                    image_psf = det*psf_gmix.make_image((size_stamp,size_stamp), jacobian=jac)
                    init = (size_stamp-size_psf_col)//2
                    image_psf = image_psf[1+init:-init,1+init:-init]
                else:
                    image_psf = det*psf_gmix.make_image((size_psf_col,size_psf_col), jacobian=jac)

                # ----------------------------------------------------------------------------------------------------------------------
   
         
    
    
    
                img_array[band][(len_images[b]):(len_images[b]+size_stamp*size_stamp)]    = image.flatten()
                img_array_m[band][(len_images[b]):(len_images[b]+size_stamp*size_stamp)]  = image_m.flatten()
                psf_array[band][(len_psf[b]):(len_psf[b]+size_psf_col*size_psf_col)]      = image_psf.flatten()
                seg_array[band][(len_images[b]):(len_images[b]+size_stamp*size_stamp)]    = seg.flatten()
                w_array[band][(len_images[b]):(len_images[b]+size_stamp*size_stamp)]      = ave_noise**2*np.ones_like(image.flatten())
                
                len_images[b] += size_stamp**2
                len_psf[b] +=  size_psf_col**2
                
                
                
                
            
                # measure moments on the fly
                size_x = image.shape[0]
                mute = np.zeros((size_x,size_x))
                size_psf_x = image_psf.shape[0]
                if size_psf_x>size_x:
                    dx = -np.int((size_x-size_psf_x)/2)
                    image_psf = image_psf[dx:dx+size_x,:][:,dx:dx+size_x]            
                # if the PiFF stamp is smaller than the image
                elif size_psf_x<size_x:
                    dx = np.int((size_x-size_psf_x)//2)+1
                    mute[dx:dx+size_psf_x,:][:,dx:dx+size_psf_x] = image_psf
                    image_psf = mute
 
            
                if DF:
                    if exp == 0:
                        #print (band,1./ave_noise,exp)
                        moments_images.append(image)
                        moments_psf.append(image_psf)
                        moments_wcs.append(wcs_BFD)
                        moments_noise.append(1./ave_noise)
                        moments_bands.append(band)
                else:
                    if exp >0:
                        #print (band,1./ave_noise,exp)
                        moments_images.append(image)
                        moments_psf.append(image_psf)
                        moments_wcs.append(wcs_BFD)
                        moments_noise.append(1./ave_noise)
                        moments_bands.append(band)
                        
                        
        # moments on the fly (1) -------------------------------------------------------
        kds, psf_shifts = bfd.multiImage(moments_images, (0,0), moments_psf, moments_wcs, 
                             pixel_noiselist = moments_noise, bandlist = moments_bands,
                             pad_factor=2, psf_recenter_sigma = 2.)
        

        BFD_filter = mc.KBlackmanHarris(sigma = config['filter_sigma']) 
        bandinfo = {'bands':config['bands_meds_files'], 'weights':config['bands_weights'],'index': np.arange(len(config['bands_meds_files']))} 
        multi_moment = bfd.MultiMomentCalculator(kds, BFD_filter, bandinfo = bandinfo)
        _= multi_moment.recenter()
        print ('moments: ', multi_moment.get_moment(0.,0.).even)
        covm_even,covm_odd , covm_even_all , _ = multi_moment.get_covariance(returnbands=True)
        print ('SN: ', multi_moment.get_moment(0.,0.).even[0]/np.sqrt(covm_even[0][0]))
       
          
        
  
            
            
    for b,band in enumerate(config['bands_meds_files']) :
        if tile == 'DF':
            path_save = config['output_folder']+'/simulated_MEDS/SN-E2/SN-E2_C01_r3688p02/SN-E2_C01_r3688p02_'+band+'_chunk_'+str(chunk)
        else:
            path_save = config['output_folder']+'/simulated_MEDS/'+tile+'_'+band+'_chunk_'+str(chunk)
        np.save(path_save,[img_array[band],img_array_m[band],psf_array[band],seg_array[band],w_array[band]])   

    
    
def run_chunk_single_threaded(chunk,config, tile, dictionary_runs,Collection_of_wide_field_galaxies, DF):
    with single_threaded(num_threads=1):
        run_chunk(chunk,config = config, tile = tile, dictionary_runs = dictionary_runs, Collection_of_wide_field_galaxies = Collection_of_wide_field_galaxies, DF = DF)
            
            
    
def simulated_MEDS(**config,):
    
    
    # makes a dictionary of the tiles that need to be run
    dictionary_runs = dict()
    for tile in config['tiles']:
        dictionary_runs[tile] = dict()
        for band in config['bands_meds_files']:
            dictionary_runs[tile][band] = dict()
            f_ = glob.glob(config['path_MEDS']+'/'+tile+'*fits.fz')
            dictionary_runs[tile][band]['meds'] = np.array(f_)[np.array([((band+'_meds') in ff) for ff in f_])][0]


    
    tiles_ = list(dictionary_runs.keys())
    tiles = []
    for tile in tiles_:
        #path = config['output_folder']+'/simulated_MEDS/simulated_MEDS_'+tile
        #if not os.path.exists(path+'_m.fits'):
            tiles.append(tile)
    tiles = np.array(tiles)

    
    # Wide field galaxies ------------------------------------------
    for tile in tiles:
        # load meds
        meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands_meds_files']]
    
    
        # load models ----
        Collection_of_wide_field_galaxies = CollectionOfImages()
        for meds_index in frogress.bar(range(meds_array[0].size)):
            Collection_of_wide_field_galaxies.add_MEDS_stamp(MedsStamp(meds_index, meds_array = meds_array, bands = config['bands_meds_files']))

        # Let's load into memory galaxy models. That is needed to do the neighbours subtraction.
        models_path = glob.glob(config['path_galaxy_models']+'/'+tile+'*')
        galaxy_models =  GalaxyModelsTable(models_path[0])
        # add models 
        Collection_of_wide_field_galaxies.add_models(galaxy_models)
  
        # start the multiprocessing
        chunk_size = config['chunk_size']
        runs = math.ceil(meds_array[0].size/chunk_size)
        xlist = range(runs)
        
        #'''
        if config['agents'] == 1:
            for chunk in xlist:
                run_chunk(chunk, config, tile, dictionary_runs,Collection_of_wide_field_galaxies = None, DF = False)
        else:
            pool = multiprocessing.Pool(processes=config['agents'])
            _ = pool.map(partial(run_chunk_single_threaded, config = config, tile = tile, dictionary_runs = dictionary_runs, Collection_of_wide_field_galaxies = None, DF = False), xlist)
        #'''

  

    if not os.path.exists(config['output_folder']+'/simulated_MEDS/SN-E2/'):
        os.mkdir(config['output_folder']+'/simulated_MEDS/SN-E2/')
    if not os.path.exists(config['output_folder']+'/simulated_MEDS/SN-E2/SN-E2_C01_r3688p02/'):
        os.mkdir(config['output_folder']+'/simulated_MEDS/SN-E2/SN-E2_C01_r3688p02/')

        #'''
    # Deep field galaxies ------------------------------------------
    dictionary_runs_df = dict()
    dictionary_runs_df['DF'] = dict()
    for band in config['bands_meds_files']:
        f =  glob.glob(config['path_df_meds']+'*_{0}_*'.format(band))
        dictionary_runs_df['DF'][band] = {'meds':f[0]}
        
    meds_array = [meds.MEDS(dictionary_runs_df['DF'][band]['meds']) for band in config['bands_meds_files']]
    chunk_size = config['chunk_size']
    runs = math.ceil(meds_array[0].size/chunk_size)
    xlist = range(runs)
    print ('DEEP FIELDS RUNS ',(runs))
    print ('length DF ',meds_array[0].size)
    
    if config['agents'] == 1:
        for chunk in xlist:
            run_chunk(chunk, config, 'DF', dictionary_runs_df, Collection_of_wide_field_galaxies = Collection_of_wide_field_galaxies, DF = True)
            
    else:
        pool = multiprocessing.Pool(processes=config['agents'])
        _ = pool.map(partial(run_chunk_single_threaded, config = config, tile = 'DF', dictionary_runs = dictionary_runs_df, Collection_of_wide_field_galaxies = Collection_of_wide_field_galaxies, DF = True), xlist)
        
        

     
    # save the DF ------------------------------------------------------------------------------------------------

        
    dictionary_runs_df = dict()
    dictionary_runs_df['DF'] = dict()
    for band in config['bands_meds_files']:
        f =  glob.glob(config['path_df_meds']+'*_{0}_*'.format(band))
        dictionary_runs_df['DF'][band] = {'meds':f[0]}
    meds_array = [meds.MEDS(dictionary_runs_df['DF'][band]['meds']) for band in config['bands_meds_files']]
    chunk_size = config['chunk_size']
    runs = math.ceil(meds_array[0].size/chunk_size)



    for band in config['bands_meds_files']:
        df = fits.open(dictionary_runs_df['DF'][band]['meds'])

        images = []
        images_ = []
        psf = []
        seg_map = []
        w_map = []

        for chunk in range(runs):
            
            m = np.load(config['output_folder']+'/simulated_MEDS/SN-E2/SN-E2_C01_r3688p02/SN-E2_C01_r3688p02_'+band+'_chunk_'+str(chunk)+'.npy',allow_pickle=True)
            images = np.hstack([images,m[0]])
            images_ = np.hstack([images_,m[1]])
            psf = np.hstack([psf,m[2]])
            seg_map = np.hstack([seg_map,m[3]])
            w_map = np.hstack([w_map,m[4]])


       

        new_hdulist = fits.HDUList()
        for i, hdu in enumerate(df):
            if i>1:
                if hdu.header['EXTNAME'] == 'image_cutouts':
                    hdu.data = copy.deepcopy(images)
                if hdu.header['EXTNAME'] == 'psf':
                    hdu.data = copy.deepcopy(psf)  
                if hdu.header['EXTNAME'] == 'weight_cutouts':
                    hdu.data = copy.deepcopy(w_map)                    
                if hdu.header['EXTNAME'] == 'seg_cutouts':
                    hdu.data = copy.deepcopy(seg_map.astype(np.int32))            
                if hdu.header['EXTNAME'] == 'bmask_cutouts':
                    hdu.data = np.zeros_like(seg_map).astype(np.int32)        
            new_hdulist.append(hdu)
            # check if it segmap/mask/image/psf and add new array

        path_save = config['output_folder']+'/simulated_MEDS/SN-E2/SN-E2_C01_r3688p02/SN-E2_C01_r3688p02_'+band+'_meds-Y3A2_DEEP_PIFF.fits'
        new_hdulist.writeto(path_save,overwrite=True)
        os.system('rm {0}_chunk*'.format(config['output_folder']+'/simulated_MEDS/SN-E2/SN-E2_C01_r3688p02/SN-E2_C01_r3688p02_'+band))

 
            
              
            
   
    # save the new meds files -------------------------------------------

    # Wide field galaxies ------------------------------------------
    for tile in tiles:
        # load meds
        meds_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands_meds_files']]
        
        # start the multiprocessing
        chunk_size = config['chunk_size']
        runs = math.ceil(meds_array[0].size/chunk_size)
        xlist = range(runs)
        
        for band in config['bands_meds_files']:
            df = fits.open(dictionary_runs[tile][band]['meds'])

            images = []
            images_ = []
            psf = []
            seg_map = []
            w_map = []
            for chunk in frogress.bar(range(runs)):
                m = np.load(config['output_folder']+'/simulated_MEDS/'+tile+'_'+band+'_chunk_'+str(chunk)+'.npy',allow_pickle=True)
                images = np.hstack([images,m[0]])
                images_ = np.hstack([images_,m[1]])
                psf = np.hstack([psf,m[2]])
                seg_map = np.hstack([seg_map,m[3]])
                w_map = np.hstack([w_map,m[4]])


            
            
            # start the multiprocessingfor b,band in enumerate(config['bands_meds_files']) :
            print ('saving + file')
            new_hdulist = fits.HDUList()
            for i, hdu in enumerate(df):
                if i>1:

                    if hdu.header['EXTNAME'] == 'image_cutouts':
                        hdu.data = copy.deepcopy(images[:].astype(np.float32))
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'image_cutouts'
                        hdu.header['BITPIX'] = '-32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1

                    if hdu.header['EXTNAME'] == 'psf':
                        hdu.data = copy.deepcopy(psf[:].astype(np.float32))
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'psf'
                        hdu.header['BITPIX'] = '-32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                        
                    if hdu.header['EXTNAME'] == 'weight_cutouts':
                        hdu.data = copy.deepcopy(w_map[:].astype(np.float32))
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'weight_cutouts'
                        hdu.header['BITPIX'] = '-32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                        

                    if hdu.header['EXTNAME'] == 'seg_cutouts':
                        hdu.data = copy.deepcopy(seg_map.astype(np.int32)[:])
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'seg_cutouts'
                        hdu.header['BITPIX'] = '32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1    
                    if hdu.header['EXTNAME'] == 'bmask_cutouts':
                        hdu.data = np.zeros_like(seg_map[:]).astype(np.int32)
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'bmask_cutouts'      
                        hdu.header['BITPIX'] = '32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                new_hdulist.append(hdu)
                # check if it segmap/mask/image/psf and add new array

            path_save = config['output_folder']+'/simulated_MEDS/'+tile+'_shearp_'+band+'_meds-Y6A2_MEDS_V3.fits.fz'
            new_hdulist.writeto(path_save,overwrite=True)
            print ('done')



            df = fits.open(dictionary_runs[tile][band]['meds'])
            print ('saving - file')
            new_hdulist = fits.HDUList()
            for i, hdu in enumerate(df):
                if i>1:
                    if hdu.header['EXTNAME'] == 'image_cutouts':
                        hdu.data = copy.deepcopy(images_[:].astype(np.float32))
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'image_cutouts'
                        hdu.header['BITPIX'] = '-32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                    if hdu.header['EXTNAME'] == 'psf':
                        hdu.data = copy.deepcopy(psf[:].astype(np.float32))
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'psf'
                        hdu.header['BITPIX'] = '-32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                    if hdu.header['EXTNAME'] == 'weight_cutouts':
                        hdu.data = copy.deepcopy(w_map[:].astype(np.float32))
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'weight_cutouts'
                        hdu.header['BITPIX'] = '-32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                        
                    if hdu.header['EXTNAME'] == 'seg_cutouts':
                        hdu.data =copy.deepcopy( seg_map.astype(np.int32)[:])
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'seg_cutouts'
                        hdu.header['BITPIX'] = '32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                    if hdu.header['EXTNAME'] == 'bmask_cutouts':
                        hdu.data = np.zeros_like(seg_map[:]).astype(np.int32)
                        header = fits.Header()
                        hdu.header = header
                        hdu.header['EXTNAME'] = 'bmask_cutouts'
                        hdu.header['BITPIX'] = '32'
                        hdu.header['NAXIS']  =        1                 
                        hdu.header['NAXIS1']  = len(images)             
                        hdu.header['PCOUNT']  =        0                 
                        hdu.header['GCOUNT']  =        1
                new_hdulist.append(hdu)

            path_save = config['output_folder']+'/simulated_MEDS/'+tile+'_shearm_'+band+'_meds-Y6A2_MEDS_V3.fits.fz'
            new_hdulist.writeto(path_save,overwrite=True)
            print ('done')
        
            os.system('rm {0}_chunk*'.format(config['output_folder']+'/simulated_MEDS/'+tile+'_'+band))

        

 
  