import meds
import psfex
import bfd
from bfd import momentcalc as mc
from bfd.momenttable import TemplateTable, TargetTable
from bfd.read_meds_utils import Image, MOF_table, DetectionsTable
from .read_meds_utils import Image, MOF_table, DetectionsTable
from .utilities import update_progress, save_obj, load_obj
import glob
import numpy as np
import pandas as pd
import pyfits as pf
from matplotlib import pyplot as plt
import ngmixer
import ngmix.gmix as gmix
import ngmix
from mpi4py import MPI             
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
from astropy import version 
import astropy.io.fits as fits

version_split_string = (version.version).split('.')
astropy_version = np.float(version_split_string[0] + '.' + version_split_string[1])



def save(self, info, noisetier):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        col.append(fits.Column(name="number",format="K",array=self.id))
        
        col.append(fits.Column(name="ra",format="D",array=info['ra']))
        col.append(fits.Column(name="dec",format="D",array=info['dec']))

        col.append(fits.Column(name="psf_g1",format="D",array=info['psf_g1']))
        col.append(fits.Column(name="psf_g2",format="D",array=info['psf_g2']))
        col.append(fits.Column(name="psf_T",format="D",array=info['psf_T']))
        col.append(fits.Column(name="NOISETIER",format="K",array=noisetier*np.ones_like(self.id)))
        
        if len(self.num_exp) == len(self.id):
            col.append(fits.Column(name="num_exp",format="K",array=self.num_exp))
        if self.cov_even is not None:
            # saved in order M0xM0, M0xMR, M0xM1, M0xM2, M0xMC,
            # MRxMR, MRxM1, MRxM2, MRxMC, M1xM1, M1xM2, M1xMC,
            # M2xM2, M2xMC, MCxMC (total of 15)
            col.append(fits.Column(name="cov_even",format="15E",array=self.cov_even))
            # saved in order MXxMX, MXxMY, MYxMY (total of 3)
            col.append(fits.Column(name="cov_odd",format="3E",array=self.cov_odd))
        if len(self.delta_flux_moment) == len(self.id):
            col.append(fits.Column(name="delta_flux_moment",format="E",array=self.delta_flux_moment))
        if len(self.cov_delta_flux_moment) == len(self.id):
            col.append(fits.Column(name="cov_delta_flux_moment",format="E",array=self.cov_delta_flux_moment))
            

        cols=fits.ColDefs(col)

        tbhdu = fits.BinTableHDU.from_columns(cols)
        self.prihdu.header['NLOST'] = self.nlost  # Update value
        tbhdu.header['WT_N'] = self.prihdu.header['WT_N'] 
        tbhdu.header['WT_SIG'] = self.prihdu.header['WT_SIG']

        tbhdu.header['TIERNAME'] = noisetier
        return self.prihdu,tbhdu



    
    
    
def make_targets_list(self, limits = {'SN_Mf_min':  0.,'SN_Mf_max': 0, 'Mf_max': 500000000 ,'Mf_min': 0}, flags = 'All', tab_targets = None):
    psf_g1 = []
    psf_g2 = []
    psf_T = []
    ra = []
    dec = []

    
    for i in range(len(self.images)):
        make_target = False
        if flags == 'All':
            make_target = True
        else:
            if self.images[i].flags == flags:
                make_target = True    
        if make_target: 
            try:
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
                    newcent=np.array([self.images[i].image_ra,self.images[i].image_dec])[:,0]+self.images[i].xyshift/3600.0
                    tab_targets.add(self.images[i].moments.get_moment(0.,0.), xy=newcent, id=self.images[i].image_ID[0],number=1,covgal=self.images[i].moments.get_covariance())
                    
                    
                    # add psf parameters to tab_targets:
                    self.images[i].psf_params_average = {'g1': 0. ,'g2': 0., 'T': 0.}
                    band_dict =  {'r':bfd.BandInfo(0.8,0),'i':bfd.BandInfo(0.6,1),'z':bfd.BandInfo(0.2,2)}
                    bands_to_use = self.images[i].bands_not_masked
                    w = 0.
                    for bx, band in enumerate(band_dict):
                        index_band = np.arange(len(self.images[i].bands))[np.array(np.in1d(self.images[i].bands,band))][0]
                        if band in bands_to_use:
                            w+=band_dict[band].weight
                            self.images[i].psf_params_average['g1'] += band_dict[band].weight * self.images[i].psf_params[index_band][0]['g1']
                            self.images[i].psf_params_average['g2'] += band_dict[band].weight * self.images[i].psf_params[index_band][0]['g2']
                            self.images[i].psf_params_average['T']  += band_dict[band].weight * self.images[i].psf_params[index_band][0]['T'] 
            
                    self.images[i].psf_params_average['g1'] = self.images[i].psf_params_average['g1']/w
                    self.images[i].psf_params_average['g2'] = self.images[i].psf_params_average['g2']/w
                    self.images[i].psf_params_average['T']  = self.images[i].psf_params_average['T']/w
 
                    psf_g1.append( self.images[i].psf_params_average['g1'])
                    psf_g2.append( self.images[i].psf_params_average['g2'])
                    psf_T.append(self.images[i].psf_params_average['T'])
                
   
                    ra.append(self.images[i].image_ra[0])
                    dec.append(self.images[i].image_dec[0])
            except:
                pass
            
    return tab_targets,psf_g1,psf_g2,psf_T,ra,dec


def add_targets_from_other_list(list1,list2):
    '''
    list and list2 are tables of targets. It allows to add the targets from list2
    to list1
    
    '''
    for binx in range(len(list1['list'])):
        if list2['info'][binx]['n'] != 0:
            for i in range(len(list2['list'][binx].id)):
                list1['list'][binx].moment.append(list2['list'][binx].moment[i])
                list1['list'][binx].id.append(list2['list'][binx].id[i])
                list1['list'][binx].number.append(list2['list'][binx].number[i])
                list1['list'][binx].xy.append(list2['list'][binx].xy[i])
                list1['list'][binx].cov_even.append(list2['list'][binx].cov_even[i])
                list1['list'][binx].cov_odd.append(list2['list'][binx].cov_odd[i])
                try:
                    list1['list'][binx].num_exp.append(list2['list'][binx].num_exp[i])
                except:
                    pass
                try:
                    list1['list'][binx].delta_flux_moment.append(list2['list'][binx].delta_flux_moment[i])
                except:
                    pass
                try:
                    list1['list'][binx].cov_delta_flux_moment.append(list2['list'][binx].cov_delta_flux_moment[i])
                except:
                    pass
                
    for binx in range(len(list1['info'])):
        if list2['info'][binx]['n'] != 0.:
            list1['info'][binx]['n'] += list2['info'][binx]['n']
            for i in range(len(list2['list'][binx].id)):
                list1['info'][binx]['psf_g1'].append(list2['info'][binx]['psf_g1'][i])
                list1['info'][binx]['psf_g2'].append(list2['info'][binx]['psf_g2'][i])
                list1['info'][binx]['psf_T'].append(list2['info'][binx]['psf_T'][i]) 
                list1['info'][binx]['ra'].append(list2['info'][binx]['ra'][i])
                list1['info'][binx]['dec'].append(list2['info'][binx]['dec'][i]) 
    return list1


            
def run_it(config,output_folder, i, file, bins_Mf,params_template):
    list_target = []
    info_target = []
    #try:
    if not os.path.exists(output_folder+'/targets/targets_junk/targets_{0}'.format(i)+'.pkl'):
            tab_mute = load_obj(file.strip('.pkl'))
            C = []
            CXY = []
            for ix in range(len(tab_mute.images)):
                try:
                    mom = tab_mute.images[ix].moments.get_moment(0.,0.)
                    CMf = (tab_mute.images[ix].moments.get_covariance()[0][mom.M0,mom.M0])
                    CMx = (tab_mute.images[ix].moments.get_covariance()[0][mom.MX,mom.MX])
                    CMy = (tab_mute.images[ix].moments.get_covariance()[0][mom.MY,mom.MY])
                    CXY.append(0.5*(CMx+CMy))
                    C.append(CMf)
                except:
                    CXY.append(-1.)
                    C.append(-1.)
            C = np.array(C)
            CXY = np.array(CXY)
            flag_s = np.array([ tab_mute.images[i].flags for i in range(len(tab_mute.images))])

            for isM in range(config['steps']):
                for ix in range(len(tab_mute.images)):
                    tab_mute.images[ix].flags = copy.deepcopy(flag_s[ix])

                #params_template['sigma_flux'] = np.float(np.int(np.sqrt((bins_Mf[isM]+bins_Mf[isM+1])*0.5))) # sqrt cov_M0M0 of the target galaxy 

                # we're now considering a hard cut of 5000*5.
                limits = {'SN_Mf_min': config['sn_min'],
                          'SN_Mf_max': config['sn_max'],
                          'Mf_max': config['Mf_max'],
                          'Mf_min': config['Mf_min']}

                tab_targets = TargetTable(n = params_template['n'],
                                  sigma = params_template['sigma'],
                                  cov=None)
                
                list_target.append(tab_targets)
                mask = (C>bins_Mf[isM])&(C<bins_Mf[isM+1])
                mean_C = np.mean(C[mask])
                mean_CXY = np.mean(CXY[mask])

                for ix in np.array(range(len(mask)))[~mask]:
                    tab_mute.images[ix].flags = -9999.
                list_target[isM],psf_g1,psf_g2,psf_T,ra,dec = make_targets_list(tab_mute, limits = limits, flags = 0, tab_targets = list_target[isM])
                info_target.append({'n':len(list_target[isM].id),'psf_g1':psf_g1,'psf_g2':psf_g2,'psf_T':psf_T,'ra':ra,'dec':dec})

            save_obj(output_folder+'/targets/targets_junk/targets_{0}'.format(i),{'list':list_target,'info':info_target})
    #except:
    #    print ('failed ', file.strip('.pkl'))
        
def make_targets(output_folder,**config):
    
    
    params_template = {}
    params_template['n'] = config['n'] 
    params_template['sigma'] =  config['sigma']  
    #params_template['sn_min'] = config['sn_min'] 
    #params_template['sigma_step'] = config['sigma_step'] 
    #params_template['sigma_max'] = config['sigma_max'] 
    #params_template['xy_max'] = config['xy_max']


    # make the sigma_Mf bins.
    dx = config['accuracy']**2*np.linspace(config['sigmaM_min']**2,config['sigmaM_max']**2,config['steps'])
    bins_Mf = np.array([config['sigmaM_min']**2+np.sum(dx[:i]) for i in range(config['steps']+1)])
    
    
    # read all the target files 
    files = glob.glob(output_folder+'/targets/*.pkl')
    list_target = []
    
    
    print ('Number of tiles: ',len(files))
    run_count = 0
    if not os.path.exists(output_folder+'/targets/targets_junk/'):
        try:
            os.mkdir(output_folder+'/targets/targets_junk/')
        except:
            pass 
        
    print ('stage: compute')

    #run_it(config, output_folder,0,files[0], bins_Mf,params_template)
    
    while run_count<len(files):
        
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank) < len(files):
            run_it(config,output_folder,run_count+comm.rank,files[run_count+comm.rank], bins_Mf,params_template)
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier()  
        
        
    
    list_targets_files = []

    print ('stage: assembly')
    files = glob.glob(output_folder+'/targets/targets_junk/targets*')

    for i,file in enumerate(files):
        if i == 0:
            list_1 = load_obj(file.strip('.pkl'))
        else:
            list_2 = load_obj(file.strip('.pkl'))
            list_1 = add_targets_from_other_list(list_1,list_2)

    
    list_targets_files=[]
    tbhdu_list = []
    for binx in range(len(list_1['list'])):
        if (list_1['info'][binx]['n'])>0:

                prihdu,tbhdu = save(list_1['list'][binx],list_1['info'][binx],binx)
                tbhdu_list.append(tbhdu)
                
    # Add band weights 
    for i in range(len(config['band_dict'])):
        prihdu.header['band_'+str(config['band_dict'][i][0])+'_weight'] =  config['band_dict'][i][1]

        
    list_final = [prihdu]
    for t in tbhdu_list:
        list_final.append(t)
    thdulist = fits.HDUList(list_final)
    
    fitsname = output_folder+'/target_sample.fits'
     
    # sometimes when doing MPI the code protests **
    try:
        if astropy_version >= 1.3:
            thdulist.writeto(fitsname,overwrite=True)
        else:
            thdulist.writeto(fitsname,clobber=True)
    except:
        pass


