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
m = pf.open('/global/cfs/cdirs/des/schutt20/catalogs/y6a2_piff_v3_allres_v3_collated.fits')
ras = m[1].data['RA']
decs = m[1].data['DEC']

p1 = m[1].data['G1_MODEL']
p2 = m[1].data['G2_MODEL']
q1 = m[1].data['G1_DATA']-m[1].data['G1_MODEL']
q2 = m[1].data['G2_DATA']-m[1].data['G2_MODEL']
w1 = m[1].data['G1_DATA']*(m[1].data['T_DATA']-m[1].data['T_MODEL'])/m[1].data['T_DATA']
w2 = m[1].data['G2_DATA']*(m[1].data['T_DATA']-m[1].data['T_MODEL'])/m[1].data['T_DATA']
del m
gc.collect()

import treecorr



Nbins = 20

min_theta = 2.5/60.
max_theta = 250./60.
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

cat_p = treecorr.Catalog(ra=ras, dec=decs, g1=p1-np.mean(p1), g2=p2-np.mean(p2),ra_units='deg', dec_units='deg',npatch=100)
cat_q = treecorr.Catalog(ra=ras, dec=decs, g1=q1-np.mean(q1), g2=q2-np.mean(q2),ra_units='deg', dec_units='deg',npatch=100)
cat_w = treecorr.Catalog(ra=ras, dec=decs, g1=w1-np.mean(w1), g2=w2-np.mean(w2),ra_units='deg', dec_units='deg',npatch=100)

catalog = load_obj('/global/cfs/cdirs/des/mgatti/BFD_CAT_V1')
cat_data = treecorr.Catalog(ra=catalog['ra'], dec=catalog['dec'], g1=catalog['e1']-np.mean(catalog['e1']), g2=catalog['e2']-np.mean(catalog['e2']),ra_units='deg', dec_units='deg',npatch=100)
del catalog
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


tau0 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau0.process(cat_data,cat_p)
tau2 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau2.process(cat_data,cat_q)
tau5 = treecorr.GGCorrelation(conf,var_method='jackknife')
tau5.process(cat_data,cat_w)



results = dict()
results['rho0'] = rho0
results['rho1'] = rho1
results['rho2'] = rho2
results['rho3'] = rho3
results['rho4'] = rho4
results['rho5'] = rho5
results['tau0'] = tau0
results['tau2'] = tau2
results['tau5'] = tau5

save_obj('rhostat',results)
#plt.errorbar(t,blinding*t*gg.xip,t*np.sqrt(gg.cov.diagonal())[:20],label = r'$\xi+$')
#plt.errorbar(t,blinding*t*gg.xim,t*np.sqrt(gg.cov.diagonal())[20:],label = r'$\xi-$')
#plt.errorbar(t,0*blinding*t*gg.xip,t*np.sqrt(gg.cov.diagonal())[:20],color='black')
#plt.xscale('log')
#
