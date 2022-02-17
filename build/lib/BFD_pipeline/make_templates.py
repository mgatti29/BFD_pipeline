import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable
from .read_meds_utils import Image, MOF_table, DetectionsTable, save_template
from .utilities import update_progress, save_obj, load_obj
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmix.gmix as gmix            
import timeit
import pickle
import time
import glob
import argparse
import os
import multiprocessing
from functools import partial
from astropy import version 
import gc
import astropy.io.fits as fits
import timeit
import copy






def pipeline(output_folder,config, deep_files, targets_properties, runs, index):
   
        t_index, d_index = runs[index]

        config['sigma_xy'] =   targets_properties[t_index]['sigma_xy']
        config['sigma_flux'] = targets_properties[t_index]['sigma_flux']
        
        tab_templates = TemplateTable(n = config['n'],
                        sigma =  config['sigma'],
                        sn_min = config['sn_min'], 
                        sigma_xy = config['sigma_xy'], 
                        sigma_flux = config['sigma_flux'], 
                        sigma_step = config['sigma_step'], 
                        sigma_max = config['sigma_max'],
                        xy_max = config['xy_max'])


        limits = {'SN_Mf_min': config['sn_min'],
                  'SN_Mf_max': config['sn_max'],
                  'Mf_max': config['Mf_max'],
                  'Mf_min': targets_properties[t_index]['Mf_min']}


        print ('sigma_flux: ', config['sigma_flux'])
        print ('sigma_XY: ', config['sigma_xy'])
          
            
        #try:
        if 1==1:
            try:
                tab_mute = load_obj(output_folder+'/templates/'+'templates_'+d_index)
            except:
                tab_mute = load_obj(output_folder+'/templates/'+'IS_templates_'+d_index)
            
            templates = []
            count = 0
            count_g = 0
            print ('number of entries: ', len(tab_mute.images))
            
            for i in range(len(tab_mute.images)):
                    
                    start = timeit.default_timer()
                    #update_progress(1.*i/(len(tab_mute.images)),timeit.default_timer(),start)           
                    cc = False
     
                    try:
                        mom = tab_mute.images[i].moments.get_moment(0.,0.)
                        cc = True
                    except:
                        print ('not a valid measurement')
                        pass
                    if cc:
              
                        mom = tab_mute.images[i].moments.get_moment(0.,0.)
                        count_g +=1
                        
                        Mf = mom.even[mom.M0]
                        SN_mute = Mf/np.sqrt(tab_mute.images[i].moments.get_covariance()[0][mom.M0,mom.M0])
              
                        mask_bool = True


                        if limits['SN_Mf_min'] > SN_mute: 
                            print ('not passing SN min')
                            mask_bool = False
                        if limits['SN_Mf_max'] < SN_mute: 
                            print ('not passing SN maxx')
                            mask_bool = False
                        if limits['Mf_min'] > Mf: 
                            print ('not passing Mf_min')
                            mask_bool = False
                        if limits['Mf_max'] < Mf: 
                            print ('not passing Mf_max')
                            mask_bool = False
                        if mask_bool:
                                #print (i)
                                t = tab_mute.images[i].moments.make_templates( config['sigma_xy'],sigma_flux = config['sigma_flux'], sn_min= config['sn_min'], sigma_max= config['sigma_max'],sigma_step= config['sigma_step'], xy_max= config['xy_max'])
                                ID_mute = copy.copy(tab_mute.images[i].image_ID[0])
                                gc.collect()
                                p0 = tab_mute.images[i].p0
                                p0_PSF = tab_mute.images[i].p0_PSF
              
                                if t[0] is None:
                                    continue
                                else:   
                                    for tmpl in t:
                                        count +=1
                                        tmpl.p0 = p0
                                        tmpl.p0_PSF = p0_PSF
                                        tmpl.id = ID_mute
                                        templates.append(tmpl)
                            #except:
                            #        pass
            
            print ('templates: ',count, ' from galaxies: ', count_g,t_index, d_index)
            save_obj(output_folder+'/templates/templates_junk/templates_{0}_{1}'.format(t_index,d_index),templates)
            print ('saved',t_index, d_index)
            #del template_list
            del tab_mute
            gc.collect()
        #except:
        #    print ('problems with file ','(IS)_templates_'+d_index)
        



        
def make_templates(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    # read list of targets :
    noise_tier = []
    targets_properties = dict()
    try:
        targets = pf.open(output_folder+'/targets_sample_g.fits')
        deep_files = glob.glob(output_folder+'/templates/'+'/t*.pkl')
    
    except:
        targets = pf.open(output_folder+'/ISp_targets_sample_g.fits')
        deep_files = glob.glob(output_folder+'/templates/'+'/IS*.pkl')
    
    # number of entries:
    
    #Let's open the noisetier file.
    try:
        NT_ = pf.open(output_folder+'/noisetiers.fits')
    except:
        NT_ = pf.open(output_folder+'/ISP_targets_sample_g_noisetiers.fits')
    
    
    noisetier = len(NT_)-1
    t_entries = np.arange(noisetier)

    for t_index,trt in enumerate(t_entries):
        targets_properties[trt] = dict()
        
        targets_properties[trt]['sigma_xy'] = np.float('{0:2.4f}'.format(np.sqrt(NT_[t_index+1].header['COVMXMX'])))
        targets_properties[trt]['sigma_flux'] = np.float('{0:2.4f}'.format((NT_[t_index+1].header['SIG_FLUX'])))
        targets_properties[trt]['Mf_min'] =  np.float('{0:2.4f}'.format((NT_[t_index+1].header['FLUX_MIN']))) 

    
    
    
    
    
    
    
    '''
    noisetier = targets[1].data['NOISETIER']
    t_entries = np.unique(targets[1].data['NOISETIER'])
    t_entries = t_entries[t_entries!=-1]
    
    print ('mean noise flux')
    for t_index,trt in enumerate(t_entries):
        print (t_index)
        noise_tier.append(trt)
        targets_properties[trt] = dict()
        mask = noisetier==trt

        targets_properties[trt]['sigma_xy'] = np.float('{0:2.4f}'.format(np.sqrt(np.mean((np.array(targets[1].data['cov_odd'])[mask,0]+np.array(targets[1].data['cov_odd'])[mask,2])*0.5))))
        targets_properties[trt]['sigma_flux'] = np.float('{0:2.4f}'.format(np.sqrt(np.mean(np.array((targets[1].data['cov_even'])[mask,0])))) )

    del targets
    '''
    gc.collect()
        
    
    # eff area
    Adeep_files = glob.glob(output_folder+'/templates/'+'/A*.pkl')
    NN_templates = 0
    AREA = 0
    for d in Adeep_files:
        t_,a_,STAMP_SIM = load_obj(d.split('.pkl')[0])
        NN_templates += t_
        AREA += a_
    number_of_runs = len(noise_tier)*len(deep_files)
    print ('# deep fields: ',len(deep_files),'# noise tiers: ',len(noise_tier))
    runs = []
   
    if not os.path.exists(output_folder+'/templates/templates_junk/'):
        try:
            os.mkdir(output_folder+'/templates/templates_junk/')
        except:
            pass
    for deep_file in deep_files:
        for i in (noise_tier):
            f = deep_file.split(output_folder+'/templates/')[1].split('templates_')[1].split('.pkl')[0]
            if not os.path.exists(output_folder+'/templates/templates_junk/templates_{0}_{1}.pkl'.format(i,f)):
                    runs.append([i,f])
                
                
                
    run_count = 0
    print ('runs to do [compute stage]: ',len(runs),' out of: ', len(deep_files)*len(noise_tier))
    
    if 'compute' in config['stage']:
        
    
        while run_count<len(runs):
            comm = MPI.COMM_WORLD
            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
            if (run_count+comm.rank) < len(runs):
                pipeline(output_folder,config, deep_files, targets_properties, runs, run_count+comm.rank)
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
   
        print ('done compute')
    
    if 'assembly' in config['stage']:
        run_count = 0
        while run_count <len(noise_tier):

                comm = MPI.COMM_WORLD
                print(" I'm rank %d from %d running in total..." % (comm.rank, comm.size))

                if (run_count+comm.rank) < len(noise_tier):
                    t_index = t_entries[run_count+comm.rank]



                    config['sigma_xy'] =   targets_properties[t_index]['sigma_xy']
                    config['sigma_flux'] = targets_properties[t_index]['sigma_flux']



                    tab_templates = TemplateTable(n = config['n'],
                                sigma = config['sigma'],
                                sn_min = config['sn_min'], 
                                sigma_xy = config['sigma_xy'], 
                                sigma_flux = config['sigma_flux'], 
                                sigma_step = config['sigma_step'], 
                                sigma_max = config['sigma_max'],
                                xy_max = config['xy_max'])


                    files = glob.glob(output_folder+'/templates/templates_junk/templates_{0}_*.pkl'.format(t_index))
                    count = 0
                    print ('number of templates files: ',len(files))
                    for ii,file in enumerate(files):
                        try:

                            template_list = load_obj(file.strip('.pkl'))

                            for tmpl in template_list:
                                #tmpl.id = count
                                #count += 1
                                tab_templates.add(tmpl)
                            del template_list
                            gc.collect()
                        except:
                            print ('failed ','.'+file.strip('.pkl'))




                    end = timeit.default_timer()
                    print ('saving NOISETIER #',t_index, ', # templates: ',(len(tab_templates.templates))   )
                    
                    if not STAMP_SIM:
                        EFFAREA = 1./AREA
                    else:
                        EFFAREA = 1./NN_templates
                    save_template(tab_templates,output_folder+'/templates_NOISETIER_{0}.fits'.format(t_index),NN_templates,EFFAREA)
                    #except:
                    #    pass



                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        print ('done assembly')
    