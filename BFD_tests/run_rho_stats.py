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
#m = pf.open('/global/cfs/cdirs/des/schutt20/catalogs/y6a2_piff_v3_allres_v3_collated.fits')
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

del m
gc.collect()

import treecorr




Nbins = 6
min_theta = 1/60.
max_theta = 100./60.
number_of_cores = 64
bin_slope = 0.01


conf = {'nbins': Nbins,
            'min_sep': min_theta,
            'max_sep': max_theta,
            'sep_units': 'degrees',
            'bin_slop': bin_slope,
            'nodes': number_of_cores  # parameter for treecorr
            }
import treecorr

cat_p =  treecorr.Catalog(ra=ras, dec=decs, g1=p1-np.mean(p1), g2=p2-np.mean(p2),ra_units='deg', dec_units='deg',npatch=500)
cat_q =  treecorr.Catalog(ra=ras, dec=decs, g1=q1-np.mean(q1), g2=q2-np.mean(q2),ra_units='deg', dec_units='deg',patch_centers=cat_p.patch_centers)
cat_w =  treecorr.Catalog(ra=ras, dec=decs, g1=w1-np.mean(w1), g2=w2-np.mean(w2),ra_units='deg', dec_units='deg',patch_centers=cat_p.patch_centers)
cat_p4 = treecorr.Catalog(ra=ras, dec=decs, g1=p4_1-np.mean(p4_1), g2=p4_2-np.mean(p4_2),ra_units='deg', dec_units='deg',patch_centers=cat_p.patch_centers)
cat_q4 = treecorr.Catalog(ra=ras, dec=decs, g1=q4_1-np.mean(q4_1), g2=q4_2-np.mean(q4_2),ra_units='deg', dec_units='deg',patch_centers=cat_p.patch_centers)


#catalog = load_obj('/global/cfs/cdirs/des/mgatti/BFD_CAT_V1')
paht_ = '/global/cfs/cdirs/des/y6-shear-catalogs/BFD_v2/BFD_v2_6_22_23_blinded.fits'
m = pf.open(paht_)

cat_data = treecorr.Catalog(ra=m[1].data['ra'], dec=m[1].data['dec'], g1=m[1].data['e1']-np.mean(m[1].data['e1']), g2=m[1].data['e2']-np.mean(m[1].data['e2']),ra_units='deg', dec_units='deg',patch_centers=cat_p.patch_centers)
del m
gc.collect()
rho0 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho0.process(cat_p,cat_p)
rho1 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho1.process(cat_q,cat_q)
rho3 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho3.process(cat_w,cat_w)
rho2 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho2.process(cat_q,cat_p)
rho4 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho4.process(cat_q,cat_w)
rho5 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho5.process(cat_p,cat_w)

rho6 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho6.process(cat_p4,cat_p)
rho7 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho7.process(cat_q4,cat_p)

rho8 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho8.process(cat_p4,cat_q)
rho9 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho9.process(cat_q4,cat_q)

rho10 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho10.process(cat_p4,cat_w)
rho11 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho11.process(cat_q4,cat_w)



rho12 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho12.process(cat_p4,cat_p4)
rho13 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho13.process(cat_q4,cat_q4)
rho14 = treecorr.GGCorrelation(conf,var_method='jackknife')
rho14.process(cat_p4,cat_q4)




tau0 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau0.process(cat_data,cat_p)
tau2 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau2.process(cat_data,cat_q)
tau5 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau5.process(cat_data,cat_w)
tau04 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau04.process(cat_data,cat_p4)
tau24 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau24.process(cat_data,cat_q4)


results = dict()
results['rho0'] = rho0
results['rho1'] = rho1
results['rho2'] = rho2
results['rho3'] = rho3
results['rho4'] = rho4
results['rho5'] = rho5
results['rho6'] = rho6
results['rho7'] = rho7
results['rho8'] = rho8
results['rho9'] = rho9
results['rho10'] = rho10
results['rho11'] = rho11
results['rho12'] = rho12
results['rho13'] = rho13
results['rho14'] = rho14
results['tau0'] = tau0
results['tau2'] = tau2
results['tau5'] = tau5
results['tau04'] = tau04
results['tau24'] = tau24
#save_obj('/pscratch/sd/m/mgatti/BFD/rhostat',results)


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
            
#results = load_obj('/pscratch/sd/m/mgatti/BFD/rhostat')

new_results = dict()
new_results['th'] = results['rho0'].meanr*60
new_results['rho0p'] =results['rho0'].xip
new_results['rho1p'] =results['rho1'].xip
new_results['rho2p'] =results['rho2'].xip
new_results['rho3p'] =results['rho3'].xip
new_results['rho4p'] =results['rho4'].xip
new_results['rho5p'] =results['rho5'].xip
new_results['rho6p'] =results['rho6'].xip
new_results['rho7p'] =results['rho7'].xip
new_results['rho8p'] =results['rho8'].xip
new_results['rho9p'] =results['rho9'].xip
new_results['rho10p'] =results['rho10'].xip
new_results['rho11p'] =results['rho11'].xip
new_results['rho12p'] =results['rho12'].xip
new_results['rho13p'] =results['rho13'].xip
new_results['rho14p'] =results['rho14'].xip






new_results['rho0m'] =results['rho0'].xim
new_results['rho1m'] =results['rho1'].xim
new_results['rho2m'] =results['rho2'].xim
new_results['rho3m'] =results['rho3'].xim
new_results['rho4m'] =results['rho4'].xim
new_results['rho5m'] =results['rho5'].xim
new_results['rho6m'] =results['rho6'].xim
new_results['rho7m'] =results['rho7'].xim
new_results['rho8m'] =results['rho8'].xim
new_results['rho9m'] =results['rho9'].xim
new_results['rho10m'] =results['rho10'].xim
new_results['rho11m'] =results['rho11'].xim
new_results['rho12m'] =results['rho12'].xim
new_results['rho13m'] =results['rho13'].xim
new_results['rho14m'] =results['rho14'].xim

new_results['tau0p'] = results['tau0'].xip
new_results['tau2p'] = results['tau2'].xip
new_results['tau5p'] = results['tau5'].xip
new_results['tau04p'] = results['tau04'].xip
new_results['tau24p'] = results['tau24'].xip


new_results['tau0m'] = results['tau0'].xim
new_results['tau2m'] = results['tau2'].xim
new_results['tau5m'] = results['tau5'].xim
new_results['tau04m'] = results['tau04'].xim
new_results['tau24m'] = results['tau24'].xim


new_results['tau0_cov'] = results['tau0'].cov
new_results['tau2_cov'] = results['tau2'].cov
new_results['tau5_cov'] = results['tau5'].cov
new_results['tau04_cov'] = results['tau04'].cov
new_results['tau24_cov'] = results['tau24'].cov
new_results['rho0p_cov'] = results['rho0'].cov
new_results['rho1p_cov'] = results['rho1'].cov
new_results['rho2p_cov'] = results['rho2'].cov
new_results['rho3p_cov'] = results['rho3'].cov
new_results['rho4p_cov'] = results['rho4'].cov
new_results['rho5p_cov'] = results['rho5'].cov

import treecorr
cov = treecorr.estimate_multi_cov([results['tau0'],results['tau2'],results['tau5'],results['tau04'],results['tau24']], 'jackknife')
new_results['tau_cov'] = cov
save_obj('rhostat_better_6bin_50_900jk',new_results)
