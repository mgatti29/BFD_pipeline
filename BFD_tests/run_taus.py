# load catalogs
import pyfits as pf
# load catalogs
import pickle
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=3)
        
def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')


            
import pickle
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=3)
        
def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')


import numpy as np
#save_obj('/global/cfs/cdirs/des/mgatti/BFD_CAT_V1',catalog)

import pyfits as pf
import gc
import os
def make_maps(III):

    '''
    print ('loading stars')
    m = pf.open('/global/cfs/projectdirs/des/schutt20/catalogs/y6a2_piff/v3_HOM_bfd/y6a2_piff_v3_HOMs_BFD_v1_collated.fits')
    ras = m[1].data['RA']
    decs = m[1].data['DEC']

    # PSF model
    p1 = m[1].data['G1_MODEL']
    p2 = m[1].data['G2_MODEL']
    # PSF model residuals
    q1 = m[1].data['G1_DATA']-m[1].data['G1_MODEL']
    q2 = m[1].data['G2_DATA']-m[1].data['G2_MODEL']
    # psf size residuals
    w1 = m[1].data['G1_MODEL']*(m[1].data['T_DATA']-m[1].data['T_MODEL'])/m[1].data['T_DATA']
    w2 = m[1].data['G2_MODEL']*(m[1].data['T_DATA']-m[1].data['T_MODEL'])/m[1].data['T_DATA']

    # 4th moment PSF
    p4_1 = m[1].data['G41_MODEL']
    p4_2 = m[1].data['G42_MODEL']

    # 4th moment PSF residuals
    q4_1 = m[1].data['G41_DATA']-m[1].data['G41_MODEL']
    q4_2 = m[1].data['G42_DATA']-m[1].data['G42_MODEL']


    # 4th moment PSF rSIZE esiduals
    w4_1 = m[1].data['G41_DATA']-m[1].data['G41_MODEL']
    w4_2 = m[1].data['G42_DATA']-m[1].data['G42_MODEL']

    print ('done stars')
    del m
    gc.collect()

    import treecorr




    Nbins = 8
    min_theta = 1/60.
    max_theta = 250./60.
    number_of_cores = 64
    bin_slope = 0.5


    conf = {'nbins': Nbins,
                'min_sep': min_theta,
                'max_sep': max_theta,
                'sep_units': 'degrees',
                'bin_slop': bin_slope,
                'nodes': number_of_cores  # parameter for treecorr
                }
    import treecorr

    cat_p =  treecorr.Catalog(ra=ras, dec=decs, g1=p1-np.mean(p1), g2=p2-np.mean(p2),ra_units='deg', dec_units='deg')
    cat_q =  treecorr.Catalog(ra=ras, dec=decs, g1=q1-np.mean(q1), g2=q2-np.mean(q2),ra_units='deg', dec_units='deg')
    cat_w =  treecorr.Catalog(ra=ras, dec=decs, g1=w1-np.mean(w1), g2=w2-np.mean(w2),ra_units='deg', dec_units='deg')
    cat_p4 = treecorr.Catalog(ra=ras, dec=decs, g1=p4_1-np.mean(p4_1), g2=p4_2-np.mean(p4_2),ra_units='deg', dec_units='deg')
    cat_q4 = treecorr.Catalog(ra=ras, dec=decs, g1=q4_1-np.mean(q4_1), g2=q4_2-np.mean(q4_2),ra_units='deg', dec_units='deg')
    
    save_obj('/pscratch/sd/m/mgatti/BFD/taus/sts',[cat_p,cat_q,cat_w,cat_p4,cat_q4])
    save_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_p',cat_p)
    save_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_q',cat_q)
    save_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_w',cat_w)
    save_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_p4',cat_p4)
    save_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_q4',cat_q4)

    '''
    Nbins = 8
    min_theta = 1/60.
    max_theta = 250./60.
    number_of_cores = 64
    bin_slope = 0.1


    conf = {'nbins': Nbins,
                'min_sep': min_theta,
                'max_sep': max_theta,
                'sep_units': 'degrees',
                'bin_slop': bin_slope,
                'nodes': number_of_cores  # parameter for treecorr
                }
    import treecorr

    print ('loading cat')
    #try:
    #    m = load_obj(base+str(III))
    #except:
    #    import os
    #    os.remove(base+str(III)+'.pkl')
    #'''
    m = load_obj(base+str(III))
    m = m['sources']
    #m = pf.open(paht_)

    cat_data = treecorr.Catalog(ra=m[0]['ra'], dec=m[0]['dec'], g1=m[0]['e1']-np.mean(m[0]['e1']), g2=m[0]['e2']-np.mean(m[0]['e2']),ra_units='deg', dec_units='deg')
    del m
    gc.collect()
    
    
    print ('done cat')

    
    
    print ('tau0')
    tau0 = treecorr.GGCorrelation(conf)
    cat_p =load_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_p')
    tau0.process(cat_data,cat_p)
    del cat_p
    gc.collect()
    print ('tau2')
    tau2 = treecorr.GGCorrelation(conf)
    cat_q = load_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_q')
    
    tau2.process(cat_data,cat_q)
    del cat_q
    gc.collect()
    print ('tau5')
    tau5 = treecorr.GGCorrelation(conf)
    cat_w = load_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_w')
    
    tau5.process(cat_data,cat_w)
    del cat_w
    gc.collect()
    print ('tau04')
    tau04 = treecorr.GGCorrelation(conf)
    cat_p4 = load_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_p4')
    
    tau04.process(cat_data,cat_p4)
    del cat_p4
    gc.collect()
    print ('tau24')
    tau24 = treecorr.GGCorrelation(conf)
    cat_q4 = load_obj('/pscratch/sd/m/mgatti/BFD/taus/sts_cat_q4')

    tau24.process(cat_data,cat_q4)
    del cat_q4
    gc.collect()

    
    
    
    
    print ('tau24')
    tauw24 = treecorr.GGCorrelation(conf)
    cat_w24 = load_obj('/global/cfs/cdirs/des/mgatti/sts_cat_w24')

    tauw24.process(cat_data,cat_w24)
    del cat_w24
    gc.collect()

    print ('tau24')
    tauw42 = treecorr.GGCorrelation(conf)
    cat_w42 = load_obj('/global/cfs/cdirs/des/mgatti/sts_cat_w42')

    tauw42.process(cat_data,cat_w42)
    del cat_w42
    gc.collect()
    
    print ('tau24')
    tauw44 = treecorr.GGCorrelation(conf)
    cat_w44 = load_obj('/global/cfs/cdirs/des/mgatti/sts_cat_w44')

    tauw44.process(cat_data,cat_w44)
    del cat_w44
    gc.collect()


    new_results = dict()
    new_results['tau0p'] = tau0.xip
    new_results['tau2p'] = tau2.xip
    new_results['tau5p'] = tau5.xip
    new_results['tau04p'] =tau04.xip
    new_results['tau24p'] =tau24.xip
    
    
    
    new_results['tau0m'] = tau0.xim
    new_results['tau2m'] = tau2.xim
    new_results['tau5m'] = tau5.xim
    new_results['tau04m'] = tau04.xim
    new_results['tau24m'] = tau24.xim

    new_results['tauw24p'] =tauw24.xip
    new_results['tauw42p'] =tauw42.xip
    new_results['tauw44p'] =tauw44.xip
        
    new_results['tauw24m'] =tauw24.xim
    new_results['tauw42m'] =tauw42.xim
    new_results['tauw44m'] =tauw44.xim
        




    save_obj('/global/cfs/cdirs/des/mgatti//taus/taus_{0}'.format(III),new_results)

    #'''
 
if __name__ == '__main__':
    runstodo=[]
    import glob
    f = glob.glob('/pscratch/sd/m/mgatti/BFD/cosmogrid_maps/*')
    base = '/pscratch/sd/m/mgatti/BFD/cosmogrid_maps/seed__fid_cosmogrid_'
    
    outp = '/global/cfs/cdirs/des/mgatti//taus/taus_'
    for f_ in f:
        
        num_ = f_.split(base)[1].split('.pkl')[0]
        if not os.path.exists(outp+str(num_)+'.pkl'):
            runstodo.append(num_)
    
    run_count=0
    #print (runstodo)
    #make_maps(runstodo[0])
    from mpi4py import MPI 
    while run_count<len(runstodo):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        try:
            make_maps(runstodo[run_count+comm.rank])
        except:
            pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        #source activate perlmutter_env
     #srun --nodes=4 --tasks-per-node=4 python run_taus.py
        
        
