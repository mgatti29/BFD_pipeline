import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable
from .read_meds_utils import Image, MOF_table, DetectionsTable, BandInfo
from .utilities import  save_obj, load_obj
            
from astropy import units as uu
from astropy.coordinates import SkyCoord
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmix.gmix as gmix
import ngmix            
import timeit
import pickle
import os
import argparse
import gc
import math
import time, sys
import multiprocessing
from functools import partial
import sys         
import copy
import frogress

def f(iii, config, params_template, chunk_size, tab_detections, dic, bands, len_file, runs,output_folder,tile,params_image_sims):    
    m_array = [meds.MEDS(dic[tile][band]['meds'][0]) for band in bands]
    psf_array = [psfex.PSFEx(dic[tile][band]['psf'][0])for band in bands]

    tab_detections_out = copy.copy(tab_detections)
    
    if config['setup_image_sims']:
        path = output_folder+'/templates/'+'/IS_templates_'+tile+'_chunk_'+str(iii)
        path_A = output_folder+'/templates/'+'/AIS_templates_'+tile+'_chunk_'+str(iii)
    else:
        path = output_folder+'/templates/'+'/templates_'+tile+'_chunk_'+str(iii)
        path_A = output_folder+'/templates/'+'/Atemplates_'+tile+'_chunk_'+str(iii)
        
 
    if not os.path.exists(path+'.pkl'):  
  
        start = timeit.default_timer()

        chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
    
        number_of_replicas = 1
        if config['setup_image_sims']:
            '''
            Here I should create the image_ from image sims.
            '''
            number_of_replicas = config['number_of_replicas']

        try:
            index_fixed = config['index_fixed']
        except:
            index_fixed = False
        EFFAREA = (chunk_range[1]-chunk_range[0])*number_of_replicas


        if config['MOF_subtraction']:
            for index_t in frogress.bar(len( tab_detections.images)):
                tab_detections.images[index_t].make_WCS()  
                        
        for index_run in frogress.bar(range(chunk_range[0],chunk_range[1])):
            
        #for index_run in (range(chunk_range[0],chunk_range[1])):
             
            
            if index_fixed:
                index = copy.copy(index_fixed)
            else:
                index = copy.copy(index_run)

            # load meds for image 'index'              
            tab_detections.images[index].Load_MEDS(index, meds = m_array)   
            # read WCS 
            tab_detections.images[index].make_WCS()
            # add psf model 
            tab_detections.images[index].add_PSF_model(index, psf = psf_array, psf_type = 'PSFex', bands = bands)
            tab_detections.images[index].zero_padd_psf()



            p0,p0_PSF=0,0
            # the following loop allows to use simulate images instead real galaxyes
            #print (config['setup_image_sims'],number_of_replicas)
            count_replica = 0
            for replica in range(number_of_replicas):
        
                if config['setup_image_sims']:
                    try: 
                        resize_sn = config['resize_sn']
                    except:
                        resize_sn = 1. 
                    try:
                        noise_ext = config['noise_ext']
                    except:
                        noise_ext = False
                        
                    if replica%2 == 0:
                        # generate a simulated image
                        redoit = True
                        while redoit:
                            if config['index_P0']: 

                                index_mute_ = copy.copy(config['index_P0'])
                            else:
                                index_mute_ = copy.copy(index_run)
                            sim_p,sim_m,sim_PSF,p0,p0_PSF,fluxes,jac = tab_detections.generate_simulated_images(index,config,params_image_sims,use_COADD_only=config['COADD_only'],noiseless=config['noiseless'],  maskless = config['maskless'],size_stamp='auto',index_P0= index_mute_,index_P0_PSF=config['index_P0_PSF'],noise_factor = config['noise_factor'],count_replica=count_replica ,resize_sn=resize_sn, noise_ext = noise_ext,g1=[0.,0.],g2=[0.,0.])
                            count_replica+=1
                            #print (config['index_P0'],index,p0,p0_PSF)
                            if sim_p is None:
                                pass
                            if sim_p is not None:
                                redoit = False
                        tab_detections.images[index].imlist = copy.deepcopy(sim_p)
                        tab_detections.images[index].psf = copy.deepcopy(sim_PSF)
             
                #Interpolates ther images over masked pixels
                if config['interp_masking']:
                    tab_detections.images[index].deal_with_bmask(use_COADD_only = config['COADD_only'])
                bands_not_masked = tab_detections.images[index].check_mfrac(limit=config['frac_limit'], use_COADD_only = config['COADD_only'])
            
                # remove exposures that don't pass the mask fraction.
                #tab_detections.images[index].discard_exposures_mfrac(config['frac_limit'])

                #subtract background
                tab_detections.images[index].subtract_background()
                #computes noise based on mask and weightmap, and then discard them.
          

                if config['setup_image_sims']:
                    tab_detections.images[index].compute_noise(config['noise_factor'],noise_ext)
                else:
                    tab_detections.images[index].compute_noise()

         
                # cut the psf stamp - for testing mostly
                if config['cut_psf'][0]:
                    for b in range(len(tab_detections.images[index].bands)):
                        for i in range((tab_detections.images[index].ncutout[b])):
                            NN = tab_detections.images[index].psf[b][i].shape[0]
                            M = np.zeros((NN,NN))
                            u = np.int((NN-config['cut_psf'][1])/2)
                            M[u:-u,u:-u] = 1.
                            tab_detections.images[index].psf[b][i] *= M

                
                if len(bands_not_masked) < config['minimum_number_of_bands']:
                    #print ('too many pixels masked')
                    tab_detections.images[index].flags += 1

                
                else:

                    # read WCS and compute moments
                    
                    if config['MOF_subtraction']:
                        tab_detections.render_MOF_models(index = index, render_self = False, render_others = True)
                    mute_range = [index,index+1]

                    tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = True, flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'])
                    
                    mom, meb = tab_detections.images[index].moments.get_moment(0.,0.,returnbands=True)
                    
                  
                    
                    tab_detections_out.images[index_run] = copy.deepcopy(tab_detections.images[index])
                    tab_detections_out.images[index_run].p0 = p0
                    tab_detections_out.images[index_run].p0_PSF = p0_PSF
              
            
            
                    #tab_detections.images[index].imlist = None
                    #tab_detections.images[index].seglist = None
                    #tab_detections.images[index].masklist = None
                    #tab_detections.images[index].wtlist = None
                    #tab_detections.images[index].jaclist = None
                    #tab_detections.images[index].psf = None
                    #tab_detections.images[index].MOF_model_rendered = None
                
        #for index_run in range(100):
        #    print (tab_detections_out.images[index_run].p0)
        
        gc.collect()
        #print (chunk_range)
        print ('erase >- ')
        for index in frogress.bar(range(chunk_range[0])):
        #for index in (range(chunk_range[0])):
                    
                    tab_detections_out.images[index].imlist = None
                    tab_detections_out.images[index].seglist = None
                    tab_detections_out.images[index].masklist = None
                    tab_detections_out.images[index].wtlist = None
                    tab_detections_out.images[index].jaclist = None
                    tab_detections_out.images[index].psf = None
                    tab_detections_out.images[index].MOF_model_rendered = None
                    try:
                        tab_detections_out.images[index].moments = None
                    except:
                        pass
        gc.collect()
                    
        print ('erase -< ')
        for index in frogress.bar(range(chunk_range[1],len(tab_detections_out.images))):
        #for index in (range(chunk_range[1],len(tab_detections_out.images))):
        
                    tab_detections_out.images[index].imlist = None
                    tab_detections_out.images[index].seglist = None
                    tab_detections_out.images[index].masklist = None
                    tab_detections_out.images[index].wtlist = None
                    tab_detections_out.images[index].jaclist = None
                    tab_detections_out.images[index].psf = None
                    tab_detections_out.images[index].MOF_model_rendered = None
                    try:
                        tab_detections_out.images[index].moments = None
                    except:
                        pass
        gc.collect()
        
       
        tab_detections.EFFAREA = EFFAREA
        save_obj(path_A,[EFFAREA,1,True])
        save_obj(path,tab_detections_out)
        
    

            
def pipeline(config, output_folder,params_template, bands, dictionary_runs, count, MOF_deep_field,deep_fields_catalog,params_image_sims):

    
        if config['MPI']:
            from mpi4py import MPI 

        tab = TemplateTable(n = params_template['n'],
                       sigma = params_template['sigma'],
                        sn_min = 0.,#self.params['sn_min'], 
                        sigma_xy = 0.,#self.params['sigma_xy'], 
                        sigma_flux = 0.,#self.params['sigma_flux'], 
                        sigma_step = 0.,#self.params['sigma_step'], 
                        sigma_max = 0.,#self.params['sigma_max'],
                        xy_max = 0.)#self.params['xy_max'])

        tab_detections = DetectionsTable(params_template)

        included_detections = 0
        discarded_detections_n = 0
        discarded_detections_mask = 0
        count_templates = 0


        start = timeit.default_timer()
        tile = list(dictionary_runs.keys())[count]

        # Read meds and psf files for a given tile ************************************
        m_array = [meds.MEDS(dictionary_runs[tile][band]['meds'][0]) for band in bands]
        psf_array = [psfex.PSFEx(dictionary_runs[tile][band]['psf'][0])for band in bands]

        # loop on detections **********************************************************
        print ('number of images: ', (m_array[0].size))
        for index in range(m_array[0].size):
            # Initialise the detection 
            Wide_g = Image(index, meds = m_array, bands = config['bands'])
            tab_detections.add_image(Wide_g)
            
        # loads MOF ********
        if MOF_deep_field is not None:
            print ('radius weird testing') 
            tab_detections.add_MOF_models(MOF_deep_field)
        
            # this checks if anything has been dropped from the fiducial deep field catalog
            pos = np.arange(len(np.array(tab_detections.ID_array)))
            df1 = pd.DataFrame(data = {'pos': pos} , index = tab_detections.ID_array)
            df2 = pd.DataFrame(index = deep_fields_catalog.ID,data = {'ra':np.array(deep_fields_catalog.ra),'dec':np.array(deep_fields_catalog.dec)})                
            uu = df2.join(df1)
            uu = uu.dropna()
            discard = pos[~np.in1d(pos,(uu['pos']))]
            for d in discard:
                if (tab_detections.images[d].flags == 0):
                    tab_detections.images[d].flags = 10
            print ('using {0}/{1}'.format(len(np.array(tab_detections.ID_array))-len(discard),len(np.array(tab_detections.ID_array))))
            
            # flaggin out stuff with a given radius *******
            the_same = True
            x = np.array(np.array(uu_.ra))*3600.
            y = np.array(np.array(uu_.dec))*3600.
            indexes_final = x==x
            print (len(x))
            config['radius_blends_templates'] =2.
            count_t = 0

            while the_same:
                catalog = SkyCoord(ra=x*uu.arcsec, dec=y*uu.arcsec)  
                idx, d2d, d3d = catalog.match_to_catalog_sky(catalog, nthneighbor=2) 
                dist_pix = np.sqrt((x-x[idx])**2+(y-y[idx])**2)
                # creating index pairs

                dist_t = config['radius_blends_templates']
                # unique pairs of close obj.
                vv = np.vstack([idx[dist_pix<dist_t],np.arange(len(idx))[dist_pix<dist_t]]).T
                vv_ = []
                if len(vv)>2:
                    for v in vv:
                        vv_.append(np.sort(v))
                    idx_close_pairs = np.unique(np.array((vv_))[:,0])
                else:
                    idx_close_pairs=-1
            #
                indexes_unique = ~np.in1d(np.arange(len(x)),idx_close_pairs)
            #
                indexes_final[indexes_final] = indexes_final[indexes_final] & indexes_unique
            #
                if (len(x)==len(x[indexes_unique])):
                    the_same = False
                else:
                    x = x[indexes_unique]
                    y = y[indexes_unique]
                count_t+=1
                if count_t>100:
                    print ('stuck') 




            all_indexes = np.arange(len(tab_detections.images))
            too_close = all_indexes[~np.in1d(all_indexes,np.array(uu_.pos)[indexes_final])]
            for d in too_close:
                if (tab_detections.images[d].flags == 0):
                    tab_detections.images[d].flags = 10
        
        #total_ob = np.min([config['max_target_per_tile'],m_array[0].size])
        
        # division in chunks *********
        print ('division in chunks')
        len_file = np.min([m_array[0].size,config['max_target_per_tile']])
        chunk_size = config['chunk_size']

        runs = math.ceil(len_file/chunk_size)
        xlist = range(runs)
        print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)
        
        del MOF_deep_field
        del deep_fields_catalog
        gc.collect()
            # if it's an image sims run, load the MOF parameters for PSF and galaxies.
        params_image_sims = dict()
        if config['setup_image_sims']:
            params_image_sims = np.load(config['simulated_templates'],allow_pickle='TRUE').item()
        print ('Doing simulated images, number of replicas: ',config['number_of_replicas'])  
        #f(0, config = config,params_template = params_template,chunk_size=chunk_size,tab_detections = tab_detections, dic = dictionary_runs, bands = config['bands'], len_file = len_file,runs = runs,output_folder = output_folder, tile = tile,params_image_sims=params_image_sims)

        #pool = multiprocessing.Pool(processes=config['agents_chunk'])
        
        #_ = pool.map(partial(f, config = config,params_template = params_template,chunk_size=chunk_size,tab_detections = tab_detections, dic = dictionary_runs, bands = config['bands'], len_file = len_file,runs = runs,output_folder = output_folder, tile = tile,params_image_sims=params_image_sims), xlist)
        for i in xlist:
            print (i)
            f(i, config = config,params_template = params_template,chunk_size=chunk_size,tab_detections = tab_detections, dic = dictionary_runs, bands = config['bands'], len_file = len_file,runs = runs,output_folder = output_folder, tile = tile,params_image_sims=params_image_sims)






def measure_moments_templates(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    '''
    It computes the moments from des y6 deef fields tiles.
    '''
    
    # Read the config file
    print ('Executing the measure_moments_templates stage')
    config['output_folder'] = output_folder
    '''
    This reads the fiducial deep field catalog. It is needed as we need to match the ra,dec from this catalog with the ra,dec from the 
    detections contained in the single-exposure meds files.
    '''
    
    config['g1'] =[0.,0.]
    config['g2'] =[0.,0.]
    deep_fields_catalog_u = pf.open(config['deep_fields_catalog'])
    deep_fields_catalog = pd.DataFrame.from_dict({'ra':np.array(deep_fields_catalog_u[1].data['RA']).byteswap().newbyteorder(),
                                                  'dec':np.array(deep_fields_catalog_u[1].data['DEC']).byteswap().newbyteorder(),
                                                  'TILENAME':np.array(deep_fields_catalog_u[1].data['TILENAME']).byteswap().newbyteorder(),
                                                  'ID':deep_fields_catalog_u[1].data['ID']})
    
    
    


    # config for computation of moments
    params_template = {}
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
        
        
    # read MOF deep fields tiles
    if config['MOF_subtraction']:
        MOF_deep_field = MOF_table(config['shredder'])
    else:
        MOF_deep_field = None
        
    deep_fields_coadd_path = config['path_coadd_deepfields']    
    fields = config['fields']
    bands = config['bands']

    
    dictionary_runs = dict()
    for field in fields:
        folders = glob.glob(deep_fields_coadd_path+field+'/*/')
        for folder in folders:
            tile = (folder.split(deep_fields_coadd_path+field+'/')[1]).split('/')[0]
            if (config['tiles'] == 'All') or (tile in config['tiles']):
                dictionary_runs[tile] = dict()
                for band in bands:
                    dictionary_runs[tile][band] = dict()
                    dictionary_runs[tile][band]['meds'] = glob.glob(folder+'/*{0}_meds*'.format(band))
                    dictionary_runs[tile][band]['psf'] = glob.glob(folder+'/*{0}_psfcat*'.format(band))
    

    print ('Number of tiles: ',len(dictionary_runs.keys()))
    
    #check if it has already been run:
    if not os.path.exists(output_folder+'/templates/'):
        try:
            os.mkdir(output_folder+'/templates/')
        except:
            pass
    if config['setup_image_sims']:
        mask = np.array([not os.path.exists(output_folder+'/templates/'+'/IS_templates_'+ff+'.pkl') for ff in list(dictionary_runs.keys())])
    else:
        mask = np.array([not os.path.exists(output_folder+'/templates/'+'/templates_'+ff+'.pkl') for ff in list(dictionary_runs.keys())])
        

    params_image_sims = dict()
    if config['setup_image_sims']:
        params_image_sims = np.load(config['simulated_templates'],allow_pickle='TRUE').item()
            
    run_count = 0
    list_run= np.arange(len(list(dictionary_runs.keys())))[mask]
    print ('Number of tiles to be run : ',len(list_run))
    run_count = 0
    
    

                
    print (len(list_run))
    while run_count<len(list_run):
        comm = MPI.COMM_WORLD
        if  run_count+comm.rank<len(list_run):
            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
            print (list_run[(run_count+comm.rank)])
            pipeline(config, output_folder, params_template, bands, dictionary_runs, list_run[(run_count+comm.rank)],MOF_deep_field,deep_fields_catalog,params_image_sims)
   
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 

        
        
        
        
    