import numpy as np
import pandas as pd
import astropy.io.fits as fits
import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.keywords import *
from bfd.momentcalc import MomentCovariance
from bfd.momenttable import TemplateTable, TargetTable
from .read_meds_utils import Image, MOF_table, DetectionsTable,BandInfo
from .utilities import save_obj, load_obj, save_moments_targets, collapse
import frogress
import glob
import timeit
import pickle
import os
import argparse
import gc
import math
import time, sys
import multiprocessing
from functools import partial       
import copy
from astropy import version 
import astropy.io.fits as fits
import galsim
import joblib
import ngmix       
import ngmix.gmix as gmix
from ngmix.jacobian.jacobian_nb import jacobian_get_vu, jacobian_get_area
 
def measure_moments_targets(output_folder,**config):
    '''
    It computes the moments from des y6 tiles.
    '''
    # Read the config file
    print ('Executing the measure_target_moments stage')
    if config['MPI']:
        from mpi4py import MPI 

    # this checks how many tiles can be used. ****************************************
    for i, b in enumerate(config['bands']):
        files = glob.glob(config['path_data']+'/*fz') 
        if i == 0:
            tiles = ['DES'+f.split('DES')[1].split('_')[0] for f in files]
        else:
            tiles1 = ['DES'+f.split('DES')[1].split('_')[0] for f in files]
            tiles = np.hstack([tiles,tiles1])
        tiles_available = np.unique(tiles)

    print ('Number of tiles available: ',len(tiles_available))

    # this selects the tiles to be used based on the config entries *******************
    print ('Input target tiles: ',config['tiles'])
    if config['tiles'] == 'All':
        tiles_to_be_used = tiles_available
    elif type(config['tiles']) is list:
        mask = np.in1d(tiles_available,config['tiles'])
        tiles_to_be_used = tiles_available[mask]
    elif type(config['tiles']) is int:
        # randomly select X tiles
        index = np.random.randint(0,len(tiles_available),config['tiles'])
        tiles_to_be_used = tiles_available[index]
    print ('Selected ',len(tiles_to_be_used),' tile(s).')
    

    # makes a dictionary of the tiles *************************************************
    dictionary_runs = dict()
    for tile in tiles_to_be_used:
        dictionary_runs[tile] = dict()
        for band in config['bands']:
            dictionary_runs[tile][band] = dict()
            f_ = glob.glob(config['path_data']+'/'+tile+'*fits.fz')
            dictionary_runs[tile][band]['meds'] = np.array(f_)[np.array([((band+'_meds') in ff) for ff in f_])][0]

     
    # list of the runs to do. ********************************************************
    list_run = []
    for count, tile in enumerate(dictionary_runs.keys()):
        if not os.path.exists(output_folder+'/targets/'):
            try:
                os.mkdir(output_folder+'/targets/')
            except:
                pass
        list_run.append(count)
    print ('Runs to do: ',len(list_run))
    config['output_folder'] = output_folder
    run_count = 0

    
    # Runs the main pipeline **********************************************************
    if ((config['MPI_per_tile']) or (not config['MPI'])):
        while run_count<len(list_run):
            pipeline(config, dictionary_runs, list_run[run_count])
            run_count+=1
    else:
        while run_count<len(list_run):
            comm = MPI.COMM_WORLD
      
            if (run_count+comm.rank) < len(list_run):
                pipeline(config, dictionary_runs, list_run[run_count+comm.rank])

            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
        
            
def pipeline(config, dictionary_runs, count):
        if config['MPI']:
            from mpi4py import MPI 
        tile = list(dictionary_runs.keys())[count]
        print ('TILE running: ',tile)
        
        if 'external' not in config.keys():
            config['external'] = False


        if config['setup_image_sims']:
            path = config['output_folder']+'/targets/ISp_targets_'+tile
        else:
            path = config['output_folder']+'/targets/targets_'+tile
            
        params_template = dict()
        params_template['n'] = config['n'] # 4 default. KSigmaWeight function index   
        params_template['sigma'] = config['sigma'] # sigma KSigmaWeight         
        
        mute_b = dict()
        for i in range(len(config['band_dict'])):
            mute_b[config['band_dict'][i][0]] = config['band_dict'][i][1]
        #params_template['band_dict'] = mute
        params_template['bands'] = config['bands']
        params_template['band_dict'] = dict()
        params_template['band_dict']['bands'] = list(config['bands'])
        w = []
        for b in config['bands']:
            w.append(mute_b[b])
        params_template['band_dict']['weights'] = list(w)
        params_template['band_dict']['index'] = list(np.arange(len(w)))

        tab_detections = DetectionsTable(params_template)

        start = timeit.default_timer()

        # Read meds and psf files for a given tile ************************************
        m_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands']]

       
        print ('number of images: ', (m_array[0].size))
        for index in range(m_array[0].size):
            # Initialise the detection 
            Wide_g = Image(index, meds = m_array, bands = config['bands'])
            tab_detections.add_image(copy.deepcopy(Wide_g))
        del m_array
        gc.collect()

        # loads MOF **********
        files = []
        if not config['shredder']:
            mute = glob.glob(config['path_data']+'mof/'+tile+'*')
        else:
            mute = glob.glob(config['path_shredder']+'/'+tile+'*')
            
        if len(mute)>0:
            for m in mute:
                files.append(m)
        print ('Loading MOF solutions')

    
        MOF_wide_field = MOF_table(files[0],config['shredder'])
        tab_detections.add_MOF_models(MOF_wide_field)
        # save the MOF models --
        if not os.path.exists(config['output_folder']+'/MOF_models/'):
            try:
                os.mkdir(config['output_folder']+'/MOF_models/')
                
            except:
                pass
                
        p_ = config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,len(tab_detections.images)-1)
        if not os.path.exists(p_):
            print ('pre-saving MOF fits')
            for ii_, index in enumerate(range(len(tab_detections.images))):
                np.save(config['output_folder']+'/MOF_models/{0}_{1}'.format(tile,index),tab_detections.images[index].MOF_models)
                del tab_detections.images[index].MOF_models
                if ii_% 1000 == 0:
                    gc.collect()

        print ('loading images and computing moments')
        
        # if it's an image sims run, load the MOF parameters for PSF and galaxies.
        params_image_sims = dict()
        if config['setup_image_sims']:
            params_image_sims = np.load(config['simulated_templates'],allow_pickle='TRUE').item()
        # division in chunks *********
        len_file = np.min([len(tab_detections.images),config['max_target_per_tile']])
        external = False
        if config['external']:
            external = np.load(config['external_path'],allow_pickle=True).item()
            len_file = len(external['ra'])
            
        chunk_size = config['chunk_size']
    
        runs = math.ceil(len_file/chunk_size)
        xlist = range(runs)
        print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)
        
        if not os.path.exists(path+config['output_label']+'.fits'):
            if config['MPI_per_tile']:
                run_count = 0
                
                while run_count<runs:
                    if config['MPI']:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<runs:
                
                            f(run_count+comm.rank, config = config, params_template = params_template,chunk_size=chunk_size, path = path+config['output_label'], tab_detections =  copy.deepcopy(tab_detections), m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims,external=external,tile=tile)
                        run_count+=comm.size
                        comm.bcast(run_count,root = 0)
                        comm.Barrier() 
                    else:
                        if run_count<runs:
                
                            f(run_count, config = config, params_template = params_template,chunk_size=chunk_size, path = path+config['output_label'], tab_detections =  copy.deepcopy(tab_detections), m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims,external=external,tile=tile)
                        run_count+=1
            else:
                if config['agents_chunk'] > 1:
                    
                    pool = multiprocessing.Pool(processes=config['agents_chunk'])

                    _ = pool.map(partial(f, config = config, params_template = params_template,chunk_size=chunk_size, path = path+config['output_label'], tab_detections =  copy.deepcopy(tab_detections), m_array = copy.deepcopy(dictionary_runs[tile]), bands = config['bands'], len_file = len_file, runs = runs,params_image_sims = params_image_sims,external=external,tile=tile), xlist)
                    
                else:
                    for x in xlist:
                        f(x, config = config, params_template = params_template,chunk_size=chunk_size, path = path+config['output_label'], tab_detections =  copy.deepcopy(tab_detections), m_array = dictionary_runs[tile], bands = config['bands'], len_file = len_file, runs = runs, params_image_sims = params_image_sims,external=external,tile=tile)


        # clean MOF and assemble
        print ('delete mof models')
        if config['MOF_subtraction']:
            for ii_, index in enumerate(range(len(tab_detections.images))):
                path_ = config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,index)
                try:
                    os.remove(path_)
                except:
                    pass

        print ('assemble')
        # assemble it back ---------
        if not os.path.exists(path+config['output_label']+'.fits'):
            if config['MPI']:
                comm = MPI.COMM_WORLD
                run_count = 0
                if comm.rank == 0:
                    collapse(path+config['output_label'])

                comm.bcast(run_count,root = 0)
                comm.Barrier() 
            else:
                collapse(path+config['output_label'])













def f(iii, config, params_template, chunk_size, path, tab_detections, m_array, bands, len_file, runs, params_image_sims,external,tile):
            
            # inititalise target table ---------
            tab_targets = TargetTable(n = params_template['n'],
                                  sigma = params_template['sigma'])

            tab_targets.ra = []
            tab_targets.dec = []
            tab_targets.psf_Mf = []
            tab_targets.psf_Mr = []
            tab_targets.psf_M1= []
            tab_targets.psf_M2= []
            tab_targets.AREA= []
            tab_targets.bkg = []
            tab_targets.len_v = []
            
           

            tab_targets.band1 = []
            tab_targets.band2 = []
            tab_targets.band3 = []
            tab_targets.p0 = []
            tab_targets.p0_PSF = []
            tab_targets.meb = []
            tab_targets.true_fluxes = []
            tab_targets.cov_odd_per_band = []
            tab_targets.cov_even_per_band = []
                    
            
            tab_targets.DESDM_coadd_x = []
            tab_targets.DESDM_coadd_y = []
            
            tab_targets.orig_row = []
            tab_targets.orig_col = []
            tab_targets.ccd_name = []
            
     
            
            
            
            
            m_array = [meds.MEDS(m_array[band]['meds']) for band in bands]
            chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
            start = timeit.default_timer()
            path_save1 = path+'_chunk_{0}'.format(iii)
            
            
            # try to read if we want to use a particular stamp for the simulations ---
            try:
                index_fixed = copy.copy(config['index_fixed'])
            except:
                index_fixed = False
                
            # set them to 0 in case it's the run on data ----
            p0 = 0
            p0_PSF = 0
            res_p = []
            res_m = []    

                
            count = 0
            if config['setup_image_sims']:
  
                    parhx= (path+'_chunk_{0}.fits'.format(iii)).replace('ISp','ISm')
            else:
                    parhx = path+'_chunk_{0}.fits'.format(iii)
         
            if not os.path.exists(parhx):
        

                if config['MOF_subtraction']:
                    print ('pre checks MOF subtraction')
                    idx_n_array = []
                    for index in frogress.bar(range(chunk_range[0],chunk_range[1])):
                        idx_n_array.append(index)
                    
                    for index in frogress.bar(range(chunk_range[0],chunk_range[1])):
                        
                        
                        
                        tab_detections.images[index].Load_MEDS_fast(index, meds = m_array)
                        for b in range(len(tab_detections.images[index].seglist)):
                            idx_n = np.unique(tab_detections.images[index].seglist[b][0].flatten())
                            idx_n = idx_n[(idx_n!=0)&(idx_n!=index+1)]#&(~np.in1d(idx_n,np.array(idx_n_array)))]
                            for i_ in idx_n:
                                idx_n_array.append(i_-1)

                        del tab_detections.images[index].jaclist
                        del tab_detections.images[index].seglist
                        del tab_detections.images[index].orig_start_rowcol
                        del tab_detections.images[index].orig_rowcol
                        if (index%50 == 0):
                            gc.collect()
                            
                    idx_n_array = np.unique(np.array(idx_n_array))                        
                    for i,index in enumerate(idx_n_array):

                
                            tab_detections.images[index].Load_MEDS_fast(index, meds = m_array,load_seglist=False)
                            tab_detections.images[index].make_WCS()
                            del tab_detections.images[index].jaclist
                            
                            tab_detections.images[index].MOF_models =  np.load(config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,index),allow_pickle=True).item()
                            if (i%100 == 0):
                                gc.collect()
          
                    print ('done pre checks')
            
            
            
            
                        
                        
                image_storage = dict()
                image_storage_ = dict()
                        
                for index_t in frogress.bar(range(chunk_range[0],chunk_range[1])):
                
                    if len(tab_detections.images) != len_file:
                        print ('error')
                        import sys
                        sys.exit()
                    #try:
                    if 1==1:
                        if index_fixed:
                            index = copy.copy(index_fixed)
                        else:
                            index = copy.copy(index_t)
                        #try:
                        if 1==1:
                            tab_detections.images[index].Load_MEDS(index, meds = m_array)
                        #except:

                        #    print (index,len(tab_detections.images),'   ')
                        # read WCS 
                        tab_detections.images[index].make_WCS()
                        #setup flag
                        tab_detections.images[index].flags = 0
                        # check masked bands and only consider objects with 2+ bands not masked
                        bands_not_masked = tab_detections.images[index].check_mfrac(limit=config['frac_limit'], use_COADD_only = config['COADD_only'])



                        #subtract background
                        bckg_flag = 0
                        
                        if config['subtract_background']:
                            bkg,len_v = tab_detections.images[index].subtract_background()
                        #computes noise based on mask and weightmap, and then discard them.
                        tab_detections.images[index].compute_noise()

                        # cut the psf stamp - for testing mostly
                        if config['cut_psf'][0]:
                            for b in range(len(tab_detections.images[index].bands)):
                                for i in range((tab_detections.images[index].ncutout[b])):
                                    NN = tab_detections.images[index].psf[b][i].shape[0]
                                    M = np.zeros((NN,NN))
                                    u = np.int((NN-config['cut_psf'][1])/2)
                                    if u == 0:
                                        M = 1.
                                    else:
                                        M[u:-u,u:-u] = 1.

                                    tab_detections.images[index].psf[b][i] *= M
                        tab_detections.images[index].zero_padd_psf()

                        
                        proceed = False
                        if type(config['minimum_number_of_bands']) == np.int:
                            if len(bands_not_masked) < config['minimum_number_of_bands']:
                                proceed = False
                        else:
                            if sum([u in bands_not_masked for u in config['minimum_number_of_bands']]) == len(config['minimum_number_of_bands']):
                                proceed = True
                    
                    
                        if not proceed:
                            #print ('too many pixels masked')
                            tab_detections.images[index].flags = 1
                        else:
                            if config['MOF_subtraction']:
                                tab_detections.render_MOF_models(index = index, render_self = False, render_others = True,use_COADD_only = config['COADD_only'])
                                
                                
                            if config['interp_masking']:
                                #Interpolates ther images over masked pixels
                                tab_detections.images[index].deal_with_bmask(use_COADD_only = config['COADD_only'])

                            mute_range = [index,index+1]

                            image_storage[index] = dict()
            
                            if config['debug']:
                                for index_band in range(3):
                                    start = 0
                                    end = tab_detections.images[index].ncutout[index_band]
                                    image_storage[index][index_band] = dict()
                                    for exp in range(start, end):  
                                        image_storage[index][index_band][exp] = [tab_detections.images[index].imlist[index_band][exp],tab_detections.images[index].MOF_model_rendered[index_band][exp],tab_detections.images[index].seglist[index_band][exp]]


                            tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = config['COADD_only'], flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'])
                            del tab_detections.images[index].MOF_model_rendered
                            del tab_detections.images[index].imlist
                            gc.collect()

                            newcent=np.array([tab_detections.images[index].image_ra,tab_detections.images[index].image_dec])[:,0]+tab_detections.images[index].xyshift/3600.0

                            mom, meb = tab_detections.images[index].moments.get_moment(0.,0.,returnbands=True)

                            
                            #*************
                            covm_even,covm_odd,covm_even_all,covm_odd_all = tab_detections.images[index].moments.get_covariance(returnbands=True)
                            covgal = covm_even,covm_odd
                            covgal_per_band = covm_even_all,covm_odd_all 


                            count+=1

                        
                            tab_targets.add(mom, xy=newcent,id=tab_detections.images[index].image_ID[0],covgal=MomentCovariance(covgal[0],covgal[1]))
                     
                            #tab_targets.p0.append(p0)
                            #tab_targets.p0_PSF.append(p0_PSF)

                            
                            tab_targets.ra.append(newcent[0])
                            tab_targets.dec.append(newcent[1])

                            meb_ = np.array([m_.even for m_ in meb])
                            mf_per_band = np.array([-999]*len(config['bands']))
                            cov_per_band = np.array([-999]*len(config['bands']))
                            
                            
                            
                            index_band =[]
                            for b_ in bands_not_masked:
                                index_band.append(np.arange(len(config['bands']))[np.array(np.in1d(config['bands'],b_))][0])
                            index_band = np.array(index_band)
                            try:
                                mf_per_band[index_band] = meb_[:,0]
                                cov_per_band[index_band] = covgal_per_band[0][0,0]
                            except:
                                print ( '\n ',(meb_[:,0],mf_per_band),index_band,'\n ')
                            
                            
                            tab_targets.meb.append(mf_per_band)
                            try:
                                tab_targets.true_fluxes.append(fluxes)
                            except:
                                pass
                            Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even
                            
                            tab_targets.AREA.append(0.)    
                   
                            tab_targets.psf_Mf.append(Mf)
                            tab_targets.psf_Mr.append(Mr)
                            tab_targets.psf_M1.append(M1)
                            tab_targets.psf_M2.append(M2)
                            tab_targets.bkg.append(bkg)
                            tab_targets.len_v.append(len_v)
                            
                             
                                
                                
                                
                            tab_targets.DESDM_coadd_x.append(tab_detections.images[index].DESDM_coadd_x)
                            tab_targets.DESDM_coadd_y.append(tab_detections.images[index].DESDM_coadd_y)

                            # let's initialise it as [12 x numb_bands] --
                            
                            orig_row_ = np.zeros(50)
                            orig_col_ = np.zeros(50)
                            ccd_name_ = -np.ones(50)
                            
                            
                            u_1 = np.array([ll[0] for l in tab_detections.images[index].orig_rowcol for ll in l[1:]])
                            u_2 = np.array([ll[1] for l in tab_detections.images[index].orig_rowcol for ll in l[1:]])
                            u_3 = np.array([ll for l in tab_detections.images[index].ccd_name for ll in l])

                            
                            #print ('')
                            #print (u_1)
                            #print (u_2)
                            #print (u_3)
                            orig_row_[:len(u_1)] = u_1 
                            orig_col_[:len(u_1)] = u_2 
                            ccd_name_[:len(u_1)] = u_3 

                            tab_targets.orig_row.append(orig_row_)
                            tab_targets.orig_col.append(orig_col_)
                            tab_targets.ccd_name.append(ccd_name_)
            
            
            
                            #tab_targets.cov_odd_per_band.append(cov_odd_save_per_band)
                            tab_targets.cov_even_per_band.append(cov_per_band)

                            nn = np.array([tab_detections.images[index].noise_rms[index_band][0] for index_band in range(tab_detections.images[index].n_bands)])

                        
                            try:
                                tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                                tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                            except:
                                pass


                  
                            if index%100==0:
                                if config['add_detection_stamp']:
                                    if proceed:
                                        #print ('adding detection stamp')
                                        
                                        # add detection stamp **************************************+++++
                                        tab_detections.images[index].Load_MEDS(index, meds = m_array)
                                        tab_detections.images[index].make_WCS()
                                        tab_detections.images[index].make_false_stamp(use_COADD_only=config['COADD_only'])
                                        tab_detections.images[index].zero_padd_psf()
                                        tab_targets.bkg.append(bkg)
                                        tab_targets.len_v.append(len_v)

                                        mute_range = [index,index+1]


                                        tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = config['COADD_only'], flags = 0, MOF_subtraction = False, band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'])

                                        tab_targets.add(mom, xy=newcent,id=tab_detections.images[index].image_ID[0],covgal=MomentCovariance(covgal[0],covgal[1]))
                                        #tab_targets.p0.append(p0)
                                       # tab_targets.p0_PSF.append(p0_PSF)


                                                #*************
                                        covm_even,covm_odd,covm_even_all,covm_odd_all = tab_detections.images[index].moments.get_covariance(returnbands=True)
                                        covgal = covm_even,covm_odd
                                        covgal_per_band = covm_even_all,covm_odd_all 
                            
                            
                            
                                        tab_targets.ra.append(newcent[0])
                                        tab_targets.dec.append(newcent[1])

                                        meb_ = np.array([m_.even for m_ in meb])
                                        
                          
                                        mf_per_band = np.array([-999]*len(config['bands']))
                                        cov_per_band = np.array([-999]*len(config['bands']))
                                
                                        index_band =[]
                                        for b_ in bands_not_masked:
                                            index_band.append(np.arange(len(config['bands']))[np.array(np.in1d(config['bands'],b_))])
                                        index_band = np.array(index_band)
                                        mf_per_band[index_band] = meb_[:,0]
                                        cov_per_band[index_band] = covgal_per_band[0][0,0]
                                        
                                        
                                        
                                        
                                        
                                        tab_targets.DESDM_coadd_x.append(0)
                                        tab_targets.DESDM_coadd_y.append(0)

                                        # let's initialise it as [12 x numb_bands] --

                                        orig_row_ = np.zeros(40)
                                        orig_col_ = np.zeros(40)
                                        ccd_name_ =  -np.ones(40)




                                        tab_targets.orig_row.append(rig_row_)
                                        tab_targets.orig_col.append(rig_col_)
                                        tab_targets.ccd_name.append(cd_name_)



                                        
                                        
                                        tab_targets.meb.append(mf_per_band)

                                        try:
                                            tab_targets.true_fluxes.append(fluxes)
                                        except:
                                            pass
                                        Mf,Mr,M1,M2,_ = tab_detections.images[index].moments_PSF.get_moment(0.,0.).even


                                        tab_targets.AREA.append(100.)    

                                        tab_targets.psf_Mf.append(Mf)
                                        tab_targets.psf_Mr.append(Mr)
                                        tab_targets.psf_M1.append(M1)
                                        tab_targets.psf_M2.append(M2)
                                        
                                        #tab_targets.cov_odd_per_band.append(cov_odd_save_per_band)
                                        tab_targets.cov_even_per_band.append(cov_per_band)

                                        try:
                                            tab_targets.des_id.append(params_image_sims[p0]['des_id'])
                                            tab_targets.photoz.append(params_image_sims[p0]['photoz'])
                                        except:
                                            pass

                                        if config['debug']:
                                            for index_band in range(3):
                                                start = 1
                                                end = tab_detections.images[index].ncutout[index_band]
                                                image_storage[index][index_band] = dict()
                                                for exp in range(start, end):  
                                                    image_storage[index][index_band][exp] = [tab_detections.images[index].imlist[index_band][exp],0,False]
                                    

                            if not (config['setup_image_sims']):
                                if  config['MOF_subtraction'] :


                                    del tab_detections.images[index]
                                    Wide_g = Image(index, meds = m_array, bands = config['bands'])
                                    tab_detections.insert_image(Wide_g,index)
                                    
                                    tab_detections.images[index].Load_MEDS_fast(index, meds = m_array)
                                    tab_detections.images[index].MOF_models =  np.load(config['output_folder']+'/MOF_models/{0}_{1}.npy'.format(tile,index),allow_pickle=True).item()
                                    if index % 5:
                                        gc.collect()

                                else:

                                    tab_detections.images[index].moments = None
                                    del tab_detections.images[index].moments  

                                    if not config['MOF_subtraction'] :
                                        tab_detections.images[index].wtlist = None
                                        tab_detections.images[index].jaclist = None
                                        tab_detections.images[index].MOF_model_rendered = None
                                        tab_detections.images[index].MOF_models = None

                                        tab_detections.images[index].rowcol = None
                                        tab_detections.images[index].orig_rowcol = None


                                        del tab_detections.images[index].wtlist 
                                        del tab_detections.images[index].jaclist 

                                        del tab_detections.images[index].rowcol 
                                        del tab_detections.images[index].orig_rowcol 



                                    tab_detections.images[index].imlist = None
                                    tab_detections.images[index].seglist = None
                                    tab_detections.images[index].masklist = None
                                    tab_detections.images[index].psf = None
                                    tab_detections.images[index].size = None
                                    tab_detections.images[index].mfrac_flag = None
                                    tab_detections.images[index].noise_rm = None




                                    del tab_detections.images[index].imlist 
                                    del tab_detections.images[index].seglist 
                                    del tab_detections.images[index].masklist 
                                    del tab_detections.images[index].psf 
                                    del tab_detections.images[index].size 
                                    del tab_detections.images[index].mfrac_flag 
                                    del tab_detections.images[index].noise_rm 
                                    del tab_detections.images[index].MOF_model_rendered


                                    if not config['MOF_subtraction'] :
                                        tab_detections.images[index] = None
                                        #del tab_detections.images[index]
                                    gc.collect()

                    #except:
                    #    print ('index failed :', index,tile,chunk_range)
                print ('\n----\n')
    
                save_moments_targets(tab_targets,path+'_chunk_{0}.fits'.format(iii),config)
                if config['debug']:
                    save_obj(path+'_image_storage_chunk_{0}.fits'.format(iii),image_storage)
                

                        
                del tab_detections
                tab_targets = None
                del tab_targets
                gc.collect()

               # except:
               #     pass



