import astropy.io.fits as fits
import math
import os
import multiprocessing
from functools import partial 
from mpi4py import MPI 


#source activate des-y6-fitvd
#srun --nodes=1 --tasks-per-node=20 python run_sof.py

# multiprocessing part
def run2(chunk_i,t,len_):

    start = 1000*chunk_i
    end = min([len_-1,1000*(chunk_i+1)-1])

    if not os.path.exists('/pscratch/sd/m/mgatti/BFD/data/fitdv_df/{1}_sof_chunk{2}.fits'.format(mode,t,chunk_i,start,end)):
        command = "fitvd --start {3} --end {4} --seed 1 --fofs  '/pscratch/sd/m/mgatti/BFD/data/fof-list/{1}_shredx-fofslist.fits' --config '/global/homes/m/mgatti/fresh_BFD/BFD_pipeline/data/config_sof.yaml' --model-pars '/pscratch/sd/m/mgatti/BFD/data/shredx_df/{1}_shredx.fits' --output '/pscratch/sd/m/mgatti/BFD/data/fitdv_df/{1}_sof_chunk{2}.fits'  '/global/cfs/cdirs/des/BFD_Y6/deep_fields_coadd_PIFF/{0}/{1}/{1}_g_meds-Y3A2_DEEP_PIFF.fits' '/global/cfs/cdirs/des/BFD_Y6/deep_fields_coadd_PIFF/{0}/{1}/{1}_r_meds-Y3A2_DEEP_PIFF.fits' '/global/cfs/cdirs/des/BFD_Y6/deep_fields_coadd_PIFF/{0}/{1}/{1}_i_meds-Y3A2_DEEP_PIFF.fits' '/global/cfs/cdirs/des/BFD_Y6/deep_fields_coadd_PIFF/{0}/{1}/{1}_z_meds-Y3A2_DEEP_PIFF.fits'".format(mode,t,chunk_i,start,end)
        os.system(command)
        
def run(i,f):
    
    t = f[i].split('/')[-1]

    m = fits.open('/global/cfs/cdirs/des/BFD_Y6/deep_fields_coadd_PIFF/{0}/{1}/{1}_g_meds-Y3A2_DEEP_PIFF.fits'.format(mode,t))
    m[1].header['NAXIS2']

    len_ = m[1].header['NAXIS2']
    chunks = math.ceil(len_/1000)
    xlist = range(chunks)
    
    pool = multiprocessing.Pool(processes=chunks)
    _ = pool.map(partial(run2, t =t ,len_ = len_), xlist)

            
import glob
mode = 'SN-C3'
#mode = 'COSMOS_UltraVISTA_Willv2'
f = glob.glob('/global/cfs/cdirs/des/BFD_Y6/deep_fields_coadd_PIFF/{0}/*'.format(mode))
run_count =0
while run_count<len(f):
    comm = MPI.COMM_WORLD


    if (run_count+comm.rank) < len(f):
        run(run_count+comm.rank,f)
    run_count+=comm.size
    comm.bcast(run_count,root = 0)
    comm.Barrier() 