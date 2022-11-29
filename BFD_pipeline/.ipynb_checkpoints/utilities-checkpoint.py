import pickle
import glob
import frogress
import pyfits as pf
from bfd.keywords import *
import os
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




# saving utilities
import astropy.io.fits as fits
import numpy as np
def save_moments_targets(self,fitsname,config):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        try:
            if len(self.p0)>0:
                col.append(fits.Column(name="id_simulated_gal",format="K",array=self.p0))
                col.append(fits.Column(name="id_simulated_PSF",format="K",array=self.p0_PSF))
        except:
            pass
        
        col.append(fits.Column(name="Mf_per_band",format="{0}E".format(np.array(self.meb).shape[1]),array=self.meb))
        try:
            if len(self.true_fluxes)>0:
                col.append(fits.Column(name="true_fluxes",format="{0}E".format(np.array(self.meb).shape[1]),array=self.true_fluxes))
        except:
            pass
        
        col.append(fits.Column(name="moments",format="5E",array=self.moment))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        
        col.append(fits.Column(name="ra",format="D",array=self.ra))
        col.append(fits.Column(name="dec",format="D",array=self.dec))
        
        PSF_moments = np.vstack([np.array(self.psf_Mf),np.array(self.psf_Mr),np.array(self.psf_M1),np.array(self.psf_M2)]).T
        col.append(fits.Column(name="psf_moments",format="4E",array=PSF_moments))

        try:
            if len(self.band1)>0:
                col.append(fits.Column(name="w_i",format="D",array=self.band1))
                col.append(fits.Column(name="w_r",format="D",array=self.band2))
                col.append(fits.Column(name="w_z",format="D",array=self.band3))
        except:
            pass
        try:
            l = np.array(self.meb).shape[1]
            col.append(fits.Column(name="cov_Mf_per_band",format="{0}E".format(l),array=np.array(self.cov_even_per_band)[:,0,:]))
        except:
            pass
        
        try:
            if len(self.photoz)>0:
                col.append(fits.Column(name="des_id",format="D",array=self.des_id))
                col.append(fits.Column(name="photoz",format="D",array=self.photoz))
        except:
            pass

        if len(self.num_exp) == len(self.id):
            col.append(fits.Column(name="num_exp",format="K",array=self.num_exp))
        col.append(fits.Column(name="covariance",format="15E",array=np.array(self.cov).astype(np.float32)))
    
        if config['setup_image_sims']:
            self.prihdu.header['STAMPS'] = 1  # Update value
            col.append(fits.Column(name="AREA",format="K",array=np.zeros(len(self.id))))
        
        else:
            self.prihdu.header['STAMPS'] = 0
            col.append(fits.Column(name="AREA",format="K",array=self.AREA))
            



        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
        return


# cllpase

def collapse(path):
            files = glob.glob(path+'*')
            for ii in frogress.bar(range(len(files))):
                file_ = files[ii]
                try:
                    mute = pf.open(file_)
                except:
                    pass
                if ii ==0:
                    mf  = mute[1].data['moments'][:,0]
                else:
                    mf = np.hstack([mf,mute[1].data['moments'][:,0]])

            hdulist = pf.HDUList([pf.PrimaryHDU()])
            cols = []        
            cols.append(pf.Column(name="notes",format="K",array=0*np.ones_like(mf)))#noisetier[mask]*np.ones_like(noisetier[mask])))
            new_cols = pf.ColDefs(cols)
            hdu = pf.BinTableHDU.from_columns(mute[1].columns + new_cols)
            for key in (hdrkeys['weightN'],hdrkeys['weightSigma']):
                hdu.header[key] = mute[0].header[key]
            for cname in mute[1].columns.names:
                sofar = 0
                for ii, file_ in enumerate(files):
                    mute = pf.open(file_)
                    nn = len(mute[1].data['moments'][:,0])
                    hdu.data[cname][sofar:sofar+nn] = mute[1].data[cname]
                    sofar += nn  
                

                                            

            for key in (hdrkeys['weightN'],hdrkeys['weightSigma']):
                hdulist[0].header[key] = mute[0].header[key]
            hdulist[0].header['STAMPS'] = mute[0].header['STAMPS']
            hdulist.append(hdu)
            del hdu

            for file_ in files:
                try:
                    os.remove(file_)
                except:
                    pass

            try:
                hdulist.writeto(path+'.fits')
            except:
                try:
                    hdulist.writeto(path+'.fits',clobber = True)# 
                except:
                    pass
