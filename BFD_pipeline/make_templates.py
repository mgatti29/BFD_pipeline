import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable
from .read_meds_utils import Image, MOF_table, DetectionsTable
from .utilities import update_progress, save_obj, load_obj
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmixer
import ngmix.gmix as gmix
from mpi4py import MPI             
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
version_split_string = (version.version).split('.')
astropy_version = np.float(version_split_string[0] + '.' + version_split_string[1])
import copy


def save(self, fitsname, config):
        savemoments=[]
        savemoments_dg1 = []
        savemoments_dg2 = []
        savemoments_dmu = []
        savemoments_dg1_dg1 = []
        savemoments_dg1_dg2 = []
        savemoments_dg2_dg2 = []
        savemoments_dmu_dg1 = []
        savemoments_dmu_dg2 = []
        savemoments_dmu_dmu = []
        id = []
        nda = []
        jSuppression = []
        
        start = timeit.default_timer()

        for i, tmpl in enumerate(self.templates):
            update_progress(i*1./len(self.templates),timeit.default_timer(),start)
            # obtain moments and derivs
            m0 = tmpl.get_moment()
            m1_dg1 = tmpl.get_dg1()
            m1_dg2 = tmpl.get_dg2()
            m1_dmu = tmpl.get_dmu()
            m2_dg1_dg1 = tmpl.get_dg1_dg1()
            m2_dg1_dg2 = tmpl.get_dg1_dg2()
            m2_dg2_dg2 = tmpl.get_dg2_dg2()
            m2_dmu_dg1 = tmpl.get_dmu_dg1()
            m2_dmu_dg2 = tmpl.get_dmu_dg2()
            m2_dmu_dmu = tmpl.get_dmu_dmu()
            # append to each list, merging even and odd moments
            savemoments.append(np.append(m0.even,m0.odd))
            savemoments_dg1.append(np.append(m1_dg1.even,m1_dg1.odd))
            savemoments_dg2.append(np.append(m1_dg2.even,m1_dg2.odd))
            savemoments_dmu.append(np.append(m1_dmu.even,m1_dmu.odd))
            savemoments_dg1_dg1.append(np.append(m2_dg1_dg1.even,m2_dg1_dg1.odd))
            savemoments_dg1_dg2.append(np.append(m2_dg1_dg2.even,m2_dg1_dg2.odd))
            savemoments_dg2_dg2.append(np.append(m2_dg2_dg2.even,m2_dg2_dg2.odd))
            savemoments_dmu_dg1.append(np.append(m2_dmu_dg1.even,m2_dmu_dg1.odd)) 
            savemoments_dmu_dg2.append(np.append(m2_dmu_dg2.even,m2_dmu_dg2.odd)) 
            savemoments_dmu_dmu.append(np.append(m2_dmu_dmu.even,m2_dmu_dmu.odd)) 
            nda.append(tmpl.nda)
            id.append(tmpl.id)
            jSuppression.append(tmpl.jSuppression)

        # Create the primary and table HDUs
        col1 = fits.Column(name="id",format="K",array=id)
        col2 = fits.Column(name="moments",format="7E",array=savemoments)
        col3 = fits.Column(name="moments_dg1",format="7E",array=savemoments_dg1)
        col4 = fits.Column(name="moments_dg2",format="7E",array=savemoments_dg2)
        col5 = fits.Column(name="moments_dmu",format="7E",array=savemoments_dmu)
        col6 = fits.Column(name="moments_dg1_dg1",format="7E",array=savemoments_dg1_dg1)
        col7 = fits.Column(name="moments_dg1_dg2",format="7E",array=savemoments_dg1_dg2)
        col8 = fits.Column(name="moments_dg2_dg2",format="7E",array=savemoments_dg2_dg2)
        col9 = fits.Column(name="moments_dmu_dg1",format="7E",array=savemoments_dmu_dg1)
        col10 = fits.Column(name="moments_dmu_dg2",format="7E",array=savemoments_dmu_dg2)
        col11= fits.Column(name="moments_dmu_dmu",format="7E",array=savemoments_dmu_dmu)
        col12= fits.Column(name="weight",format="E",array=nda)
        col13= fits.Column(name="jSuppress",format="E",array=jSuppression)
        cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13])
        tbhdu = fits.BinTableHDU.from_columns(cols, header=self.hdr)
        prihdu = fits.PrimaryHDU()
        for i in range(len(config['band_dict'])):
            prihdu.header['band_'+str(config['band_dict'][i][0])+'_weight'] =  config['band_dict'][i][1]

        
        thdulist = fits.HDUList([prihdu,tbhdu])
        if astropy_version >= 1.3:
            thdulist.writeto(fitsname,overwrite=True)
        else:
            thdulist.writeto(fitsname,clobber=True)
        return

def make_templates_l(self, limits = {'SN_Mf_min':  0.2,'SN_Mf_max': 0, 'Mf_max': 500000000 ,'Mf_min': 500}, params = None, flags = 'All', make_only_one = False):
    count = 0
    print ('making templates')
    count_galaxies = 0
    templates = []
    start = timeit.default_timer()
    

    #len_file = len(self.images)
    #chunk_size = 5000    
    #pool = multiprocessing.Pool(processes=10)
    #runs = math.ceil(len_file/chunk_size)
    #xlist = range(runs)
    #print ('subruns for this tile: ',runs,' chunk size: ', chunk_size)
    #    
    #_ = pool.map(partial(f,), xlist)
    #
    #
    #    
    #def f(iii):
    #    chunk_range =  [int(chunk_size*iii),int(np.min([chunk_size*(iii+1),len_file]))] 
    #    for i in range(chunk_range[0],chunk_range[1]):
    print ('number of entries: ', len(self.images))
    
    for i in range(len(self.images)):
        update_progress(i*1./len(self.images),timeit.default_timer(),start)
                    
        make_templates = False
        if flags == 'All':
            make_templates = True
        else:
            if self.images[i].flags == flags:
                make_templates = True    
        if make_templates: 
            try:
                if make_only_one:
                    t = make_one_template(self.images[i].moments,params['sigma_xy'],sigma_flux= params['sigma_flux'], sn_min= params['sn_min'], sigma_max= params['sigma_max'], 
                sigma_step = params['sigma_step'], xy_max = params['xy_max'], image_id = self.images[i].image_ID) 
                    
                    if t[0] is None:
                        continue
                    else:   
                        for tmpl in t:
                            count +=1
                            tmpl.id = self.images[i].image_ID[0]
                            templates.append(tmpl)
                            
                else:
                    mom = self.images[i].moments.get_moment(0.,0.)
                    mask_bool = True
                    Mf = mom.even[mom.M0]
                    SN_mute = Mf/np.sqrt(self.images[i].moments.get_covariance()[0][mom.M0,mom.M0])
                    
                    if limits['SN_Mf_min'] > SN_mute: 
                        mask_bool = False
                    if limits['SN_Mf_max'] < SN_mute: 
                        mask_bool = False
                    if limits['Mf_min'] > Mf: 
                        mask_bool = False
                    if limits['Mf_max'] < Mf: 
                        mask_bool = False
                    if mask_bool:
                        count_galaxies +=1
                        t = self.images[i].moments.make_templates( params['sigma_xy'],sigma_flux = params['sigma_flux'], sn_min= params['sn_min'], sigma_max= params['sigma_max'],sigma_step= params['sigma_step'], xy_max= params['xy_max'])
                        ID_mute = copy.copy(self.images[i].image_ID[0])
                        gc.collect()
                    if t[0] is None:
                        continue
                    else:   
                        for tmpl in t:
                            count +=1
                            tmpl.id = ID_mute
                            templates.append(tmpl)
             

            except:
                pass
            
    print ('number of templates: ',count)
    return templates



def pipeline(output_folder,config, deep_files, targets_properties, runs, index):
        t_index, d_index = runs[index]
        
 
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



        limits = {'SN_Mf_min': config['sn_min'],
                  'SN_Mf_max': config['sn_max'],
                  'Mf_max': config['Mf_max'],
                  'Mf_min': config['Mf_min']}


        print ('sigma_flux: ', config['sigma_flux'])
        print ('sigma_XY: ', config['sigma_xy'])
          
            
            

        #multiprocess ****************************
        #pool = multiprocessing.Pool(processes=5)
        #runs = len(files)
        #xlist = range(runs)
        #_ = pool.map(partial(f, files=files, limits = limits, params_template = params_template, target_i = target_i), xlist)
        
        if 1==1:
        #try:
            tab_mute = load_obj(deep_files[d_index].strip('.pkl'))
            template_list = make_templates_l(tab_mute, limits = limits, flags = 0, params = config)
            save_obj(output_folder+'/templates/templates_junk/templates_{0}_{1}'.format(t_index,d_index),template_list)
            del template_list
            del tab_mute
            gc.collect()
        #except:
        #    print ('something went wrong with file: ',deep_files[d_index].strip('.pkl'))
        #    try:
        #        del template_list
        #    except:
        #        pass
        #    try:
        #        del tab_mute
        #    except:
        #        pass
        #    gc.collect()
    #


        


def make_templates(output_folder,**config):

    # read list of targets :
    noise_tier = []
    targets_properties = dict()
    print (output_folder)
    try:

        targets = pf.open(output_folder+'/target_sample.fits')
        # number of entries:
        for t_index in range(1,len(targets)):
            noise_tier.append(t_index)
            targets_properties[t_index] = dict()
            targets_properties[t_index]['sigma_xy'] = np.float(np.int(np.sqrt(np.mean((np.array(targets[t_index].data['cov_odd'])[:,0]+np.array(targets[t_index].data['cov_odd'])[:,2])*0.5))))
            targets_properties[t_index]['sigma_flux'] = np.float(np.int(np.sqrt(np.mean(np.array((targets[t_index].data['cov_even'])[:,0])))) )
        
    except:
        print ('Target file not found')
        exit()
    del targets
    gc.collect()
        
    deep_files = glob.glob(output_folder+'/templates/'+'/*.pkl')
    number_of_runs = len(noise_tier)*len(deep_files)
    print ('# deep fields: ',len(deep_files),'# noise tiers: ',len(noise_tier))
    runs = []
    
            
    if not os.path.exists(output_folder+'/templates/templates_junk/'):
        os.mkdir(output_folder+'/templates/templates_junk/')
        
    for j in range(len(deep_files)):
        for i in (noise_tier):
            if not os.path.exists(output_folder+'/templates/templates_junk/templates_{0}_{1}.pkl'.format(i,j)):
                runs.append([i,j])
                
                
                
    run_count = 0
    print ('runs: ',len(runs),' out of: ', len(deep_files)*len(noise_tier))
    print ('compute stage')
    
    #while run_count<len(runs):
    #    comm = MPI.COMM_WORLD
    #    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
    #    if (run_count+comm.rank) < len(runs):
    #        pipeline(output_folder,config, deep_files, targets_properties, runs, run_count+comm.rank)
    #    run_count+=comm.size
    #    comm.bcast(run_count,root = 0)
    #    comm.Barrier() 


        
        
        
        

    run_count = 0
    while run_count <len(noise_tier):
        
            comm = MPI.COMM_WORLD
            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

            if (run_count+comm.rank) < len(noise_tier):
                t_index = run_count+comm.rank+1



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
                for ii,file in enumerate(files):
                    try:
                    
                        template_list = load_obj(file.strip('.pkl'))

                        for tmpl in template_list:
                            #tmpl.id = count
                            #count += 1
                            tab_templates.add(tmpl)
                    except:
                        print ('failed ','.'+file.strip('.pkl'))
      



                end = timeit.default_timer()
                print ('saving NOISETIER #',t_index, ', # templates: ',(len(tab_templates.templates))   )
                try:
                    save(tab_templates,output_folder+'/templates_NOISETIER_{0}.fits'.format(t_index),config)
                except:
                    pass



            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
