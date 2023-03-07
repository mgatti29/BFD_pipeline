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


import numpy as np
#save_obj('/global/cfs/cdirs/des/mgatti/BFD_CAT_V1',catalog)


import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
   
#des scheme
colors = ['#601A4A', '#EE442F','#63ACBE']


m = pf.open('//global/cfs/cdirs/des/mgatti/exposure_field_centers.fits')

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

# laod BFD catalog ----
catalog = load_obj('/global/cfs/cdirs/des/mgatti/BFD_CAT_V1')
cat_data = treecorr.Catalog(ra=catalog['ra'], dec=catalog['dec'], g1=catalog['e1']-np.mean(catalog['e1']), g2=catalog['e2']-np.mean(catalog['e2']),ra_units='deg', dec_units='deg',npatch=100)

# load lenses ----
cat_a = treecorr.Catalog(ra=m[1].data['RA_CENT'], dec=m[1].data['DEC_CENT'],ra_units='deg', dec_units='deg',npatch=100)
ng = treecorr.NGCorrelation(conf,var_method='jackknife')
ng.process(cat_a,cat_data)


# random 
r = pf.open('/global/cfs/cdirs/des/y6-shear-catalogs/y6-combined-hsmap_random.fits')

cat_random = treecorr.Catalog(ra=r[1].data['ra'], dec=r[1].data['dec'],ra_units='deg', dec_units='deg',npatch=100)
rg = treecorr.NGCorrelation(conf,var_method='jackknife')
rg.process(cat_random,cat_data)




gammat,gammat_im,gammaterr=ng.calculateXi(rg) 
covj = ng.estimate_cov('jackknife')
save_obj('field_centers',[gammat,gammat_im,gammaterr,covj,ng])
