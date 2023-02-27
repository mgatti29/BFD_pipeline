import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable
from .read_meds_utils import Image, MOF_table, DetectionsTable, BandInfo,select_obj
from .utilities import  save_obj, load_obj
            
from astropy import units as uu
from astropy.coordinates import SkyCoord
import glob
import numpy as np
import pandas as pd
import astropy.io.fits as fits
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

def f(iii, config, params_template, chunk_size, tab_detections, dic, bands, len_file, runs,output_folder,tile):    
    
    m_array = [meds.MEDS(dic[tile][band]['meds'][0]) for band in bands]
    psf_array = [psfex.PSFEx(dic[tile][band]['psf'][0])for band in bands]
    tab_detections_out = copy.deepcopy(tab_detections)
    fluxes_out = dict()
    path = output_folder+'/templates/'+'/templates_'+tile+'_chunk_'+str(iii)
    path_m = output_folder+'/templates/'+'/Mf_templates_'+tile+'_chunk_'+str(iii)
    path_A = output_folder+'/templates/'+'/Atemplates_'+tile+'_chunk_'+str(iii)
    path_i = output_folder+'/templates/'+'/image_storage_templates_'+tile+'_chunk_'+str(iii)
    image_storage = dict()
 
    if not os.path.exists(path+'.pkl'):  
  
        # define the chunk range ++++++++
        chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
    
        # number of objects
        EFFAREA = (chunk_range[1]-chunk_range[0])


        if config['MOF_subtraction']:
            for index_t in frogress.bar(range(len( tab_detections.images))):
                tab_detections.images[index_t].Load_MEDS_fast(index_t, meds = m_array)
                tab_detections.images[index_t].make_WCS()  
                        
        for index in frogress.bar(range(chunk_range[0],chunk_range[1])):
            
            image_storage[index] = dict()
            # load meds for image 'index'              
            tab_detections.images[index].Load_MEDS(index, meds = m_array)   
            # read WCS 
            tab_detections.images[index].make_WCS()
            # add psf model 
            tab_detections.images[index].add_PSF_model(index, psf = psf_array, psf_type = 'PSFex', bands = bands)
            tab_detections.images[index].zero_padd_psf()

             

            bands_not_masked = tab_detections.images[index].check_mfrac(limit=config['frac_limit'], use_COADD_only = config['COADD_only'])

            if config['subtract_background']:
                tab_detections.images[index].subtract_background()
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

            #print (bands_not_masked)
            proceed = False
            if type(config['minimum_number_of_bands']) == np.int:
                if len(bands_not_masked) < config['minimum_number_of_bands']:
                    proceed = False
            else:
                if sum([u in bands_not_masked for u in config['minimum_number_of_bands']]) == len(config['minimum_number_of_bands']):
                    proceed = True
            if not proceed:
                #print ('too many pixels masked')
                tab_detections.images[index].flags += 1

                if config['debug']:
                    for index_band in range(len(bands)):
                        image_storage[index][index_band] = [tab_detections.images[index].imlist[index_band],0.,tab_detections.images[index].seglist[index_band],proceed,bands_not_masked,config['minimum_number_of_bands']]
                    
            else:

                # read WCS and compute moments

                if config['MOF_subtraction']:
                    tab_detections.render_MOF_models(index = index, render_self = False, render_others = True)
                #Interpolates ther images over masked pixels
                if config['interp_masking']:
                    tab_detections.images[index].deal_with_bmask(use_COADD_only = config['COADD_only'])
                
                mute_range = [index,index+1]

                try:
                #if 1==1:
                    tab_detections.compute_moments(params_template['sigma'], bands = params_template['bands'], use_COADD_only = True, flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'],filter_ = config['filter'])
                    mom, meb = tab_detections.images[index].moments.get_moment(0.,0.,returnbands=True)
                    
            

                    if config['debug']:
                        for index_band in range(len(bands)):
                            image_storage[index][index_band] = [tab_detections.images[index].imlist[index_band],tab_detections.images[index].MOF_model_rendered[index_band],tab_detections.images[index].seglist[index_band],proceed,bands_not_masked,config['minimum_number_of_bands']]
                         
                        
                    # mf per band +++++++++++++++=
                    mf_per_band = -999*np.ones(len(config['bands']))
                    index_band =[]
                    for b_ in bands_not_masked:
                        index_band.append(np.arange(len(config['bands']))[np.array(np.in1d(config['bands'],b_))])
                    index_band = np.array(index_band)
                    index_band = index_band[:,0]
                    meb_ = np.array([m_.even for m_ in meb])

                    mf_per_band[index_band] = meb_[:,0]
                    
                    
                    tab_detections.compute_moments(params_template['sigma'], bands = config['minimum_number_of_bands'], use_COADD_only = True, flags = 0, MOF_subtraction = config['MOF_subtraction'], band_dict = params_template['band_dict'], chunk_range = mute_range,pad_factor=config['pad_factor'], filter_ = config['filter'])
                    mom, meb = tab_detections.images[index].moments.get_moment(0.,0.,returnbands=True)
                    
                    
                   # print (mf_per_band)
                    #tab_detections.images[index].mf_per_band = mf_per_band
                    tab_detections_out.images[index] = copy.deepcopy(tab_detections.images[index])
                    fluxes_out[index] = dict()
                    fluxes_out[index]['mf_per_band'] = mf_per_band
                    fluxes_out[index]['index'] =       copy.deepcopy(tab_detections.images[index].image_ID[0])
                    fluxes_out[index]['MOF_index'] =   copy.deepcopy(tab_detections.images[index].MOF_index)
                    fluxes_out[index]['MAG_I'] =       copy.deepcopy(tab_detections.images[index].MAG_I)
                    fluxes_out[index]['TILENAME'] =    copy.deepcopy(tab_detections.images[index].TILENAME)
                    
                    
                   # tab_detections.images[index].mf_per_band = None
                    tab_detections.images[index].MOF_index  = None
                    tab_detections.images[index].MAG_I  = None
                    tab_detections.images[index].TILENAME = None
                    
                   # tab_detections_out.images[index].mf_per_band = None
                    tab_detections_out.images[index].MOF_index  = None
                    tab_detections_out.images[index].MAG_I  = None
                    tab_detections_out.images[index].TILENAME = None
                except:
                    pass
                    
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
                    tab_detections_out.images[index].DESDM_coadd_x = None
                    tab_detections_out.images[index].DESDM_coadd_y = None
                    tab_detections_out.images[index].ccd_name = None
                    
    
            
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
                    tab_detections_out.images[index].DESDM_coadd_x = None
                    tab_detections_out.images[index].DESDM_coadd_y = None
                    tab_detections_out.images[index].ccd_name = None
                    
    
        for index in frogress.bar(range(len(tab_detections_out.images))):
            try:
                del tab_detections_out.images[index].MOF_models    
            except:
                pass
                
        gc.collect()
        
       
        tab_detections.EFFAREA = EFFAREA
        save_obj(path_A,[EFFAREA,1,True])
        
        save_obj(path,tab_detections_out)
        save_obj(path_m,fluxes_out)
        
        
        if config['debug']:
            save_obj(path_i,image_storage)
            
            

    del tab_detections_out
    #del tab_detections
    gc.collect()

            
def pipeline(config, output_folder,params_template, bands, dictionary_runs, count, MOF_deep_field,deep_fields_catalog):

    
        if config['MPI']:
            from mpi4py import MPI 

        tab = TemplateTable(n = params_template['n'],
                       sigma = params_template['sigma'],
                        sn_min = 0.,
                        sigma_xy = 0.,
                        sigma_flux = 0.,
                        sigma_step = 0.,
                        sigma_max = 0.,
                        xy_max = 0.)

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
            using = len(np.array(tab_detections.ID_array))-len(discard)
            print ('using {0}/{1}'.format(using,len(np.array(tab_detections.ID_array))))
            
            if using != 0:        
                # flaggin out stuff with a given radius *******
                the_same = True
                x = np.array(np.array(uu['ra']))*3600.
                y = np.array(np.array(uu['dec']))*3600.     
 
                mask_too_close = select_obj(x,y,config['radius_blends_templates'])



                #all_indexes = np.arange(len(tab_detections.images))



                all_indexes = np.arange(len(tab_detections.images))
                too_close = all_indexes[~np.in1d(all_indexes,np.array(uu.pos)[mask_too_close])]
                for d in too_close:
                    if (tab_detections.images[d].flags == 0):
                        tab_detections.images[d].flags = 10
                        
            del df1
            del df2
            del uu
            gc.collect()
        

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
        if not os.path.exists(output_folder+'/templates/'+'/templates_'+tile+'.pkl'):

            if config['agents_chunk']>1:
                pool = multiprocessing.Pool(processes=config['agents_chunk'])

                _ = pool.map(partial(f, config = config,params_template = params_template,chunk_size=chunk_size,tab_detections = copy.deepcopy(tab_detections), dic = dictionary_runs, bands = config['bands'], len_file = len_file,runs = runs,output_folder = output_folder, tile = tile), xlist)


            else:
                for i in xlist:
                    f(i, config = config,params_template = params_template,chunk_size=chunk_size,tab_detections = copy.deepcopy(tab_detections), dic = dictionary_runs, bands = config['bands'], len_file = len_file,runs = runs,output_folder = output_folder, tile = tile)
                    
                    
            for ii_, index in enumerate(range(len(tab_detections.images))):
                try:
                    del tab_detections.images[index].MOF_models    
                except:
                    pass
            del tab_detections
            gc.collect()

                    
            
            # clean up and merge ----
            files = glob.glob(output_folder+'/templates/'+'/templates_'+tile+'*')

            save__ = dict()
            #save__m = dict()
            count = 0
    

            for file in files:

                tub = load_obj(file.split('.pkl')[0])
            
                for index in range(len(tub.images)):  
                

                    cc = False
                    try:
                    #if 1==1:
                        mom = tub.images[index].moments.get_moment(0.,0.)
                        if (mom.even ==  mom.even)[0]:
                            cc = True
                    except:
                        #print ('not a valid measurement')
                        pass
                    if cc:

                        count +=1
                        save__[index] = dict()
                        save__[index]['moments'] = tub.images[index].moments
                        #save__m[index]['mf_per_band'] = tub.images[index].mf_per_band
                        #save__m[index]['index'] = tub.images[index].image_ID[0]
                        save__[index]['index'] = tub.images[index].image_ID[0]
                        #try:
                        #    save__m[index]['MOF_index'] = tub.images[index].MOF_index
                        #    save__m[index]['MAG_I'] = tub.images[index].MAG_I
                        #    save__m[index]['tilename'] = tub.images[index].TILENAME                
                        #except:
                        #    pass
                        try:
                            save__[index]['ra'] = tub.images[index].image_ra[0]
                            save__[index]['dec'] = tub.images[index].image_dec[0]
                        except:
                            pass
                del tub
                gc.collect()
                os.remove(file)
            #print (output_folder+'/templates/'+'/templates_'+tile)
        
            save_obj(output_folder+'/templates/'+'/templates_'+tile,save__)
            
            
            
        
            # clean up and merge ----
            files = glob.glob(output_folder+'/templates/'+'/Mf_templates_'+tile+'*')

            save__ = dict()
            #save__m = dict()
            count = 0
    

            for file in files:

                tub = load_obj(file.split('.pkl')[0])
            
                for index in tub.keys():
                

                    try:
 
                        count +=1
                        save__[index] = dict()
                        #save__[index]['moments'] = tub.images[index].moments
                        save__[index]['mf_per_band'] = tub[index]['mf_per_band']
                        #save__m[index]['index'] = tub.images[index].image_ID[0]
                        save__[index]['index'] = tub[index]['index']
                        save__[index]['MOF_index'] = tub[index]['MOF_index']
                        save__[index]['MAG_I'] = tub[index]['MAG_I']
                        save__[index]['TILENAME'] = tub[index]['TILENAME']
                    except:
                        pass
              
                del tub
                gc.collect()
                os.remove(file)
            #print (output_folder+'/templates/'+'/templates_'+tile)
        
            save_obj(output_folder+'/templates/'+'/Mf_templates_'+tile,save__)
            
            
            #save_obj(output_folder+'/templates/'+'/Mf_templates_'+tile,save__m)
            del save__
            gc.collect()

            files = glob.glob(output_folder+'/templates/'+'/Atemplates_'+tile+'*')
            count = 0
            for ff in files:
                m = load_obj(ff.split('.pkl')[0])
                count += m[0]

                os.remove(ff)

            #print (output_folder+'/templates/'+'/Atemplates_'+tile)
            save_obj(output_folder+'/templates/'+'/Atemplates_'+tile,[count,m[1] ,m[2]])  
            print ('done')



def measure_moments_templates(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    '''
    It computes the moments from des y6 deef fields tiles.
    '''
    
    
    if 'filter' not in config.keys():
        config['filter'] = 'KBlackmanHarris'

    # Read the config file
    print ('Executing the measure_moments_templates stage')
    config['output_folder'] = output_folder
    '''
    This reads the fiducial deep field catalog. It is needed as we need to match the ra,dec from this catalog with the ra,dec from the 
    detections contained in the single-exposure meds files.
    '''
    

    deep_fields_catalog_u = fits.open(config['deep_fields_catalog'])
    deep_fields_catalog = pd.DataFrame.from_dict({'ra':np.array(deep_fields_catalog_u[1].data['RA']).byteswap().newbyteorder(),
                                                  'dec':np.array(deep_fields_catalog_u[1].data['DEC']).byteswap().newbyteorder(),
                                                  'TILENAME':np.array(deep_fields_catalog_u[1].data['TILENAME']).byteswap().newbyteorder(),
                                                  'ID':deep_fields_catalog_u[1].data['ID']})
    
    
    


    # config for computation of moments
    params_template = {}
    params_template['n'] = config['n'] 
    params_template['sigma'] = config['sigma'] 

    
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
        MOF_deep_field = MOF_table(config['MOF_models'])
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
        
        
    mask = np.array([not os.path.exists(output_folder+'/templates/'+'/templates_'+ff+'.pkl') for ff in list(dictionary_runs.keys())])
        

    run_count = 0
    list_run= np.arange(len(list(dictionary_runs.keys())))[mask]
    print ('Number of tiles to be run : ',len(list_run))
    run_count = 0
    
    

                
    if config['MPI']:
        while run_count<len(list_run):
            comm = MPI.COMM_WORLD
            if  run_count+comm.rank<len(list_run):
                try:
                    pipeline(config, output_folder, params_template, bands, dictionary_runs, list_run[(run_count+comm.rank)],MOF_deep_field,deep_fields_catalog)
                except:
                    pass
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
    else:
        while run_count<len(list_run):
      
            if  run_count<len(list_run):
                #try:
                    pipeline(config, output_folder, params_template, bands, dictionary_runs, list_run[(run_count)],MOF_deep_field,deep_fields_catalog)
                #except:
                #    pass
            run_count+=1
       


    del deep_fields_catalog_u
    del deep_fields_catalog
    del MOF_deep_field
    gc.collect()
        
        
        
        
    
