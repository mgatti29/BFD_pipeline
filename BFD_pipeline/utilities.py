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




# saving utilities
import astropy.io.fits as fits

def save_moments_targets(self,fitsname,config):
        '''
        modified save function for moments with different sigma_Mf entries
        '''
        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        try:
            col.append(fits.Column(name="id_simulated_gal",format="K",array=self.p0))
            col.append(fits.Column(name="id_simulated_PSF",format="K",array=self.p0_PSF))
        except:
            pass
        
        col.append(fits.Column(name="Mf_per_band",format="{0}E".format(np.array(self.meb).shape[1]),array=self.meb))
        try:
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
            col.append(fits.Column(name="AREA",format="K",array=self.area))
            
        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)
        return
