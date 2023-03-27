import meds
import copy
import glob
import numpy as np
import astropy.io.fits as fits
import os
from astropy.table import Table
from mpi4py import MPI 


def doit(config,index, meds = [],exp_list = False):
    ncutout = [m['ncutout'][index] for m in meds]
    seglist = [m.get_cutout_list(index, type='seg') for m in meds]
    masklist = [m.get_cutout_list(index, type='bmask') for m in meds]
    imlist = [m.get_cutout_list(index) for m in meds]


    id_ = m_array[0]['id'][index]

    bands = len(meds)
    bkg_tot = 0
    count = 0
    len_v = 0
    len_vc = 0
    for b in range((bands)):
        for i in range(1,(ncutout[b])):
            seg = copy.deepcopy(seglist[b][i])
            segg0 = copy.deepcopy(seglist[b][i])
            for ii in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                seg += (np.roll(segg0, ii, axis = 0) + np.roll(segg0, ii, axis = 1))

            mask__ = copy.deepcopy(masklist[b][i] )
            segg0 = copy.deepcopy(masklist[b][i] )
            for ii in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                mask__ += (np.roll(segg0, ii, axis = 0) + np.roll(segg0, ii, axis = 1))


            mask = (seg == 0) & (mask__ == 0) 
            maskarr=np.ones(np.shape(mask),dtype='int')
            maskarr[~mask] = 0
            uu = 6
            maskarr[uu:-uu,uu:-uu]=0
            v = imlist[b][i][np.where(maskarr==1)]

            if len(v)>50: 
                correction = np.median(v)
                bkg_tot += config['w'][b]*correction
                count +=  config['w'][b]
                len_v = len(v)  
                len_vc +=  config['w'][b]*len(v)  

    if count ==0:
        return id_,bkg_tot,100000000000000,100000000000000
    else:
        return id_,bkg_tot/count,len_v/count,len_vc/count

    
    
    
    
    
    

    
    
    
    
    
    
    
# makes a dictionary of the tiles *************************************************
def compute_bck(tile,config):
    dictionary_runs = dict()
    for tile in tiles:
        dictionary_runs[tile] = dict()
        for band in config['bands']:
            dictionary_runs[tile][band] = dict()
            f_ = glob.glob(config['path_data']+'/'+tile+'*fits.fz')
            dictionary_runs[tile][band]['meds'] = np.array(f_)[np.array([((band+'_meds') in ff) for ff in f_])][0]

    for tile in dictionary_runs.keys():
        path_ = output+'background_'+tile+'.fits'
        if not os.path.exists(path_):
            m_array = [meds.MEDS(dictionary_runs[tile][band]['meds']) for band in config['bands']]




            id_ = []
            bkg_tot = []
            len_vc = []
            import frogress
            for index in frogress.bar(range(len(m_array[0]['id']))):

                _id_,_bkg_tot,_len_v,_len_vc = doit(config,index, meds = m_array,exp_list = False)
                id_.append(_id_)
                bkg_tot.append(_bkg_tot)
                len_vc.append(_len_vc)

            fits_f = Table()
            fits_f['id'] = np.array(id_)
            fits_f['bkg_tot'] = np.array(bkg_tot)
            fits_f['len_vc'] =  np.array(len_vc)
            fits_f.write(path_)
            #
     
    
    
    
    
if __name__ == '__main__':
    

    config = dict() 
    config['path_data'] = '/global/cscratch1/sd/mgatti/BFD//data_y6/'
    config['bands']  = ['g','r','i','z']
    config['w']  = [0.,0.7,0.2,0.1]
    tiles =  ['DES0137-3749'] # here you need the names of all the tiles
    output = '/global/cscratch1/sd/mgatti/BFD/output_bck/'

    
    
               

    run_count=0
    while run_count<len(tiles):
        comm = MPI.COMM_WORLD
#
        if (run_count+comm.rank)<len(tiles):
            try:
                compute_bck(tiles[run_count+comm.rank],config)
            except:
                pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
