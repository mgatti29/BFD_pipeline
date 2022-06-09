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
import frogress





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
                  'Mf_min': config['Mf_min']}#targets_properties[t_index]['Mf_min']}


        print ('sigma_flux: ', config['sigma_flux'])
        print ('sigma_XY: ', config['sigma_xy'])
          
            
        #try:
        if 1==1:
            try:
                save_ = load_obj(output_folder+'/templates/'+'templates_'+d_index)
            except:
                save_ = load_obj(output_folder+'/templates/'+'IS_templates_'+d_index)
            
            templates = []
            count = 0
            count_g = 0
            print ('number of entries: ', len(save_.keys()))
            
            #for i in range(len(tab_mute.images)):
            # downsample
            
            try:
                
                downsample_factor = config['downsample_factor']
                downsample = np.int(len(save_.keys())*downsample_factor)
            except:
                downsample = len(save_.keys())
                downsample_factor=1.
            
    
            for i_ in frogress.bar(range( np.int(len(save_.keys())*downsample_factor))):
                    
                    i = list(save_.keys())[i_]

                    
                    
                    start = timeit.default_timer()
                    #update_progress(1.*i/(len(tab_mute.images)),timeit.default_timer(),start)           
                    cc = False
     
                    try:
                    #if 1==1:
                        #mom = tab_mute.images[i].moments.get_moment(0.,0.)
                        mom = save_[i]['moments'].get_moment(0.,0.)

                        cc = True
                    except:
                    #    print ('not a valid measurement')
                        pass
                    if cc:
                        #print (i)
                        #mom = tab_mute.images[i].moments.get_moment(0.,0.)
                        mom = save_[i]['moments'].get_moment(0.,0.)

                        count_g +=1
                        
                        Mf = mom.even[mom.M0]
                        SN_mute = Mf/np.sqrt(save_[i]['moments'].get_covariance()[0][mom.M0,mom.M0])
              
                        mask_bool = True


                        if limits['SN_Mf_min'] > SN_mute: 
                            #print ('not passing SN min')
                            mask_bool = False
                        if limits['SN_Mf_max'] < SN_mute: 
                            #print ('not passing SN maxx')
                            mask_bool = False
                        if limits['Mf_min'] > Mf: 
                            #print ('not passing Mf_min')
                            mask_bool = False
                        if limits['Mf_max'] < Mf: 
                            #print ('not passing Mf_max')
                            mask_bool = False
                        if mask_bool:
                                
                                #print (i)
                                t = save_[i]['moments'].make_templates( config['sigma_xy'],sigma_flux = config['sigma_flux'], sn_min= config['sn_min'], sigma_max= config['sigma_max'],sigma_step= config['sigma_step'], xy_max= config['xy_max'])
                                ID_mute = copy.copy(save_[i]['index'])
                                gc.collect()
                                try:
                                    p0 = save_[i]['p0']
                                    p0_PSF = save_[i]['p0_PSF']
                                               
                                except:
                                    
                                    p0 = 0
                                    p0_PSF = 0
              
                                if t[0] is None:
                                    continue
                                else:   
                                    for tmpl in t:
                                        count +=1
                                        tmpl.p0 = p0
                                        tmpl.p0_PSF = p0_PSF
                                        tmpl.id = ID_mute
                                        tmpl.nda *= 1./downsample_factor
                                        templates.append(tmpl)
                            #except:
                            #        pass
            
            print ('templates: ',count, ' from galaxies: ', count_g,t_index, d_index)
            save_obj(output_folder+'/templates/templates_junk/templates_{0}_{1}'.format(t_index,d_index),templates)
            print ('saved',t_index, d_index)
            #del template_list
            del save_
            gc.collect()
        #except:
        #    print ('problems with file ','(IS)_templates_'+d_index)
        



        
def make_templates(output_folder,**config):
    if config['MPI']:
        from mpi4py import MPI 
    print ('making templates')

    if ('compute' in config['stage']) or ('assembly' in config['stage']):
        
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
           # targets_properties[trt]['Mf_min'] =  np.float('{0:2.4f}'.format((NT_[t_index+1].header['FLUX_MIN']))) 


        #print (targets_properties)




        gc.collect()


        # eff area
        Adeep_files = glob.glob(output_folder+'/templates/'+'/A*.pkl')
        NN_templates = 0
        AREA = 0
        for d in Adeep_files:
            t_,a_,STAMP_SIM = load_obj(d.split('.pkl')[0])
            NN_templates += t_
            AREA += a_
        number_of_runs = noisetier*len(deep_files) #len(noise_tier)*len(deep_files)
        print ('number_of_runs: ', number_of_runs)
        print ('# deep fields: ',len(deep_files),'# noise tiers: ',noisetier)
        runs = []

        if not os.path.exists(output_folder+'/templates/templates_junk/'):
            try:
                os.mkdir(output_folder+'/templates/templates_junk/')
            except:
                pass
        for deep_file in deep_files:
            for i in range(noisetier):
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
        print ('noisetiers ',noisetier)
        while run_count <noisetier:

                comm = MPI.COMM_WORLD
                print(" I'm rank %d from %d running in total..." % (comm.rank, comm.size))

                if (run_count+comm.rank) < noisetier:
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
                    print (AREA*NN_templates)
                    print (AREA)
                    print (NN_templates)
                    
                    if not STAMP_SIM:
                        EFFAREA = AREA
                    else:
                        EFFAREA = NN_templates
                        
                    count = 0
                    for ii in frogress.bar(range(len(files))):
                        file = files[ii]
                        try:

                            template_list = load_obj(file.strip('.pkl'))
                            
                            for jj_ in (range((len(template_list)))):
                                
                                tmpl = template_list[jj_]
                                #tmpl.id = count
                                #count += 1
                                try:
                                    tmpl.nda  *= 1./EFFAREA*config['correction_factor_selection']
                                except:
                                    
                                    tmpl.nda  *= 1./EFFAREA
                                tab_templates.add(tmpl)
                                count += 1
                        except:
                            print ('failed ','.'+file.strip('.pkl'))
                    

                    
                    '''
                    
                    boia
                    '''
                    print ('total templates ',count)
                    print ('done')


                    end = timeit.default_timer()
                    print ('saving NOISETIER #',t_index, ', # templates: ',(len(tab_templates.templates))   )
                    

                    save_template(tab_templates,output_folder+'/templates_NOISETIER_{0}.fits'.format(t_index),EFFAREA,t_index)
                    #except:
                    #    pass



                run_count+=comm.size
                comm.bcast(run_count,root = 0)
                comm.Barrier() 
        print ('done assembly')

    if 'resume_templates' in config['stage']:   
       
        print ('resume templates')
        

        deep_files = glob.glob(output_folder+'/templates/'+'/t*.pkl')
        try:
            deep_files[0]
        except:
            deep_files = glob.glob(output_folder+'/templates/'+'/IS_t*.pkl')
            
        moments = []
        error = []
        indexu = []
        mof_index = []
        ra = []
        dec = []
        ra_DF = []
        dec_DF = []
        MAG_I = []
        tilename = []
        p0_ = []
        p0_PSF_ = []
        for ii in frogress.bar(range(len(deep_files))):

            df = deep_files[ii]
            try:
                save_ = load_obj(df.strip('.pkl'))
                for index in (save_.keys()):

                    mom = save_[index]['moments'].get_moment(0.,0.)

                    moments.append(mom.even)
                    error.append(np.sqrt(save_[index]['moments'].get_covariance()[0].diagonal()))
                    indexu.append(save_[index]['index'] )

                    try:
                        mof_index.append(save_[index]['MOF_index'] )
                        ra.append(save_[index]['ra'])
                        dec.append(save_[index]['dec'])
                        ra_DF.append(save_[index]['ra'])
                        dec_DF.append(save_[index]['dec'])
                        MAG_I.append(save_[index]['MAG_I'])
                        tilename.append(save_[index]['tilename'])
                    except:
                        pass
                    try:
                        p0_.append(save_[index]['p0'])
                        p0_PSF_.append(save_[index]['p0_PSF'])
                    except:
                        pass
            except:
                print (df)

        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="index",format="K",array=np.array(indexu)))
        col.append(fits.Column(name="moments",format="5E",array=np.array(moments)))
        col.append(fits.Column(name="error_moments",format="5E",array=np.array(error)))
        
        try:
            col.append(fits.Column(name="mof_index",format="K",array=np.array(mof_index)))
            col.append(fits.Column(name="ra",format="E",array=np.array(ra)))
            col.append(fits.Column(name="dec",format="E",array=np.array(dec)))
            col.append(fits.Column(name="ra_DF",format="E",array=np.array(ra_DF)))
            col.append(fits.Column(name="dec_DF",format="E",array=np.array(dec_DF)))
            col.append(fits.Column(name="MAG_I",format="E",array=np.array(MAG_I)))
            col.append(fits.Column(name="tilename",format="128A",array=np.array(tilename)))
        except:
            pass
        
        try:
            col.append(fits.Column(name="p0",format="E",array=np.array(p0_)))
            col.append(fits.Column(name="p0_PSF",format="E",array=np.array(p0_PSF_)))
            
        except:
            pass
        tbhdu = fits.BinTableHDU.from_columns(col)
        prihdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([prihdu,tbhdu])
        try:
            thdulist.writeto(output_folder+'/templates_overview.fits',overwrite=True)
        except:
            pass

    
'''
I can probably add another snippet of code that load all the files (templates) saved and create some reference file similar to the targets,
where you. have covariances, moments, id's,ra,dec,tilename.



'''
