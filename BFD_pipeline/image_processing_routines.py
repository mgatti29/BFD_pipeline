import astropy.io.fits as fits
import numpy as np
import pandas as pd
import galsim
import ngmix
import ngmix.gmix as gmix
import bfd
import copy
import bfd
from bfd import momentcalc as mc

            

def render_model_on_stamp(gal_pars,psf_pars, wcs, shape, g1=None, 
                          g2=None, nbrxyref=None, galaxy_model = 'BDF',psf_model = 'piff'):
    """
    Render an image of a galaxy using galaxy and PSF parameters, with optional shearing.

    Parameters:
        
        gal_pars (array-like): Parameters of the galaxy model.
        psf_pars (dict): Parameters of the PSF model. Can be a dictionary with 'turb' key for turbulent model.
        wcs (object): World Coordinate System object that contains transformation details.
        shape (int): The shape of the output image (assumed square).
        g1 (float, optional): First component of shear to apply to the galaxy image.
        g2 (float, optional): Second component of shear to apply to the galaxy image.
        return_PSF (bool, optional): Whether to return the PSF image along with the galaxy image.
        nbrxyref (tuple, optional): The reference (x, y) position of a neighbor to set the galaxy center.

    Returns:
        tuple: Depending on the return_PSF, it may return a galaxy image, PSF image, and a Jacobian object,
               or just the galaxy image and Jacobian object.

    Raises:
        Exception: If the image generation process fails.
    """
    

    if psf_model =='psf_model':
        psf_gmix = gmix.GMix(pars=psf_pars)
    elif psf_model =='turb':
        psf_gmix = ngmix.GMixModel(pars=psf_pars['pars'], model="turb")

    # Calculate the determinant of the WCS transformation matrix
    det = np.abs(wcs.getdet()) 

    # Initialize the Jacobian of the WCS transformation
    jac = ngmix.jacobian.Jacobian(row=wcs.xy0[1],
                                  col=wcs.xy0[0],
                                  dudrow=wcs.jac[0,1],
                                  dudcol=wcs.jac[0,0],
                                  dvdrow=wcs.jac[1,1],
                                  dvdcol=wcs.jac[1,0])

    if gal_model =='bdf':
        gmix_sky = gmix.GMixModel(gal_pars, model='bdf')
    elif gal_model =='shredder':
        g_ = copy.deepcopy(gal_pars)
        gmix_sky = ngmix.GMix(pars=g_.reshape(60))
        
    
    # Apply shear to the galaxy model if provided
    if (g1 is not None) and (g2 is not None):
        gmix_sky = gmix_sky.get_sheared(g1, g2)

    # Convolve the galaxy model with the PSF model
    gmix_image = gmix_sky.convolve(psf_gmix)

    # Attempt to generate the galaxy and PSF images
    try:
        # Generate the PSF image if requested
        #im_psf = psf_gmix.make_image((shape, shape), jacobian=jac) if return_PSF else None

        # Adjust the galaxy center if a neighbor reference position is provided
        if nbrxyref is not None:
            v, u = jac(nbrxyref[1], nbrxyref[0])
            gmix_image.set_cen(v, u)

        # Generate the galaxy image
        image = gmix_image.make_image((shape, shape), jacobian=jac)

    except Exception as e:
        # Handle exceptions during image generation and decide what to return based on return_PSF flag
        return None, jac

    # Return the requested images and Jacobian object
    return image, jac
    
    
def check_on_bad_exposures(meds_array, bad_exposure_list):
    # Extract image paths from the meds_array, which is assumed to contain
    # metadata about the images. The '_image_info' dictionary key is used
    # to access the paths, and the leading character of each path is removed.
    images_path = [m._image_info['image_path'][1:] for m in meds_array]

    # Loop through each band (spectral range) to process the exposures.
    for index_band in range(len(meds_array)):
        # For the first band, extract the exposure numbers from the image paths.
        # The exposure number is assumed to be at the start of the file name,
        # following a specific pattern ('red/D00'). This number is converted to an integer.
        if index_band == 0:
            exposures_MEDS = np.array([np.int((image_path).split('_')[0].strip('red/D00')) for image_path in images_path[index_band]])
        else:
            # For subsequent bands, perform the same extraction and then
            # combine (stack horizontally) these exposure numbers with those from previous bands.
            exposures_MEDS_1 = np.array([np.int((image_path).split('_')[0].strip('red/D00')) for image_path in images_path[index_band]])
            exposures_MEDS = np.hstack([exposures_MEDS, exposures_MEDS_1])
    
    # Initialize a dictionary to hold the mask for each band's bad exposures.
    mask_exposure = dict()
    # Iterate over each band in the bad_exposure_list.
    for band in bad_exposure_list.keys():
        # Determine which exposures in the current band are bad by matching them
        # against the exposure list obtained from the images.
        match_exposures = np.array(np.in1d(bad_exposure_list[band]['exp'], exposures_MEDS))

        try:
            # Attempt to create a unique identifier for each bad exposure by combining
            # the exposure number with the CCD number (multiplied by 100 for scaling).
            # This identifier helps in masking specific exposures in each band.
            mask_exposure[band] = 100*bad_exposure_list[band]['exp'][match_exposures] + bad_exposure_list[band]['ccd'][match_exposures]
        except:
            # If the above operation fails, fall back to using only the exposure numbers.
            mask_exposure[band] = bad_exposure_list[band]['exp'][match_exposures]
                    
    return mask_exposure


class GalaxyModelsTable:
    """
    A class for handling galaxy models tables stored in FITS format.

    This class provides methods to load galaxy images and associated data
    from FITS files, allowing manipulation of the data such as computing
    moments, and accessing parameters for specific galaxies.

    Attributes:
        catalog (HDUList): An Astropy HDUList object representing the opened FITS file.
        this_is_wide_field (bool): Flag indicating whether the dataset is from a wide field (default True).
        cols (list): List of column names in the FITS file's second HDU.
        id_epoch_array (ndarray): Array of IDs for each epoch in the FITS file's third HDU.
        id_array (ndarray): Array of unique galaxy IDs in the FITS file's second HDU.
        bdf_mag (ndarray): Array of magnitudes for the galaxy models.
        bdf_flux (ndarray): Array of fluxes for the galaxy models.
        bdf_params (ndarray): Array of Bulge-Disk-Fit parameters for the galaxy models.
        pfs_params (ndarray): Array of point spread function parameters.
        bdf_ra (ndarray): Array of right ascension values for the galaxy models.
        bdf_dec (ndarray): Array of declination values for the galaxy models.
        numbands (int): Number of bands in the magnitude array.
        ra (ndarray): Array of right ascension values from the first HDU.
        dec (ndarray): Array of declination values from the first HDU.
    
    Methods:
        match_epochs_by_ID(ID): Returns the indices where the given ID matches the id_epoch_array.
        return_band_val_wide_field(bandind, indtostr): Converts band index to string or vice versa for wide fields.
        return_band_val_deep_fields(bandind, indtostr): Converts band index to string or vice versa for deep fields.
        return_model(band, pos, pos_epoch, shredder): Returns model parameters for a given galaxy and band.
        select_obj_by_ID(ID): Returns the indices where the given ID matches the id_array.
    """

    def __init__(self, path, this_is_wide_field=True):
        """
        Initializes the GalaxyModelsTable object, loading data from a FITS file.

        Parameters:
            path (str): The file path to the FITS file containing galaxy model data.
            this_is_wide_field (bool, optional): Indicates if the data is from a wide field. Defaults to True.
        """
        self.catalog = fits.open(path)
        self.this_is_wide_field = this_is_wide_field
        self.cols = self.catalog[1].data.columns.names
        self.id_epoch_array = self.catalog[2].data['id']
        self.id_array = self.catalog[1].data['id']
        self.bdf_mag = self.catalog[1].data['bdf_mag']
        self.bdf_flux = self.catalog[1].data['bdf_flux']
        self.bdf_params = self.catalog[1].data['bdf_pars']
        self.pfs_params = self.catalog[2].data['psf_pars']
        self.bdf_ra = self.catalog[1].data['ra']
        self.bdf_dec = self.catalog[1].data['dec']
        self.numbands = np.shape(self.bdf_mag)[1]
        self.ra = self.catalog[1].data['ra']
        self.dec = self.catalog[1].data['dec']
        

    def match_epochs_by_ID(self, ID):
        """Returns the indices of epochs with the specified ID."""
        return np.where(self.id_epoch_array == ID)

    def return_band_val_deep_fields(self,bandind,indtostr=True):
        if indtostr:
            if self.numbands==5:
                if bandind==0: return 'u'
                if bandind==1: return 'g'
                if bandind==2: return 'r'
                if bandind==3: return 'i'
                if bandind==4: return 'z'
            if self.numbands==4:
                if bandind==0: return 'g'
                if bandind==1: return 'r'
                if bandind==2: return 'i'
                if bandind==3: return 'z'
            if self.numbands==3:
                if bandind==0: return 'J'
                if bandind==1: return 'H'
                if bandind==2: return 'Ks'
        else:
            if self.numbands==5:
                if bandind=='u': return 0
                if bandind=='g': return 1
                if bandind=='r': return 2
                if bandind=='i': return 3
                if bandind=='z': return 4
            if self.numbands==4:
                if bandind=='g': return 0
                if bandind=='r': return 1
                if bandind=='i': return 2
                if bandind=='z': return 3
            if self.numbands==3:
                if bandind=='J':  return 0
                if bandind=='H':  return 1
                if bandind=='Ks': return 2
    
    def return_band_val_wide_field(self,bandind,indtostr=True):
        if indtostr:
            if self.numbands==5:
                if bandind==0: return 'g'
                if bandind==1: return 'r'
                if bandind==2: return 'i'
                if bandind==3: return 'z'
                if bandind==4: return 'Y'
        else:
            if self.numbands==5:

                if bandind=='g': return 0
                if bandind=='r': return 1
                if bandind=='i': return 2
                if bandind=='z': return 3
                if bandind=='Y': return 4
            

    def return_model(self, band='i', pos=None, pos_epoch=None, shredder=False):
        """
        Returns model parameters for a specified galaxy and band.

        Parameters:
            band (str): The band for which model parameters are requested.
            pos (int): The position index in the bdf_params and bdf_flux arrays.
            pos_epoch (int): The position index in the pfs_params array.
            shredder (bool): Flag indicating whether to use the shredder algorithm (not yet implemented).

        Returns:
            dict: A dictionary containing 'gal_pars' and 'psf_pars' for the galaxy.
        """
        if self.this_is_wide_field:
            index_band = self.return_band_val_wide_field(band, indtostr=False)
        else:
            index_band = self.return_band_val_deep_fields(band, indtostr=False)

        gal_pars = self.bdf_params[pos][0:6]
        gal_pars = np.append(gal_pars, self.bdf_flux[pos][index_band])
        pos_epoch = pos_epoch[index_band]
        psf_pars = self.pfs_params[pos_epoch]
        mof_params = {'gal_pars': gal_pars, 'psf_pars': psf_pars}

        return mof_params
            
    def select_obj_by_ID(self, ID):
        """Returns the indices of objects with the specified ID."""
        return np.where(self.id_array == ID)

    
    


class CollectionOfImages:
    '''
    This class is designed to manage and process a collection of images from the Multi-Epoch Data Structure (MEDS). It supports loading images into memory, manipulating them, and performing operations like measuring moments and rendering models.
    '''

    def __init__(self):
        '''
        Initializes the CollectionOfImages instance.
        - MEDS_stamps: List to store MEDS image data.
        - coadd_IDs: List to store the coaddition IDs associated with each image.
        - MEDS_indexes: List to store indexes of MEDS images.
        '''
        self.MEDS_stamps = []
        self.coadd_IDs = []
        self.MEDS_indexes = []
        
        

        
    def add_MEDS_stamp(self, MEDS_stamp):
        '''
        Adds a single MEDS stamp to the collection.
        - MEDS_stamp: The MEDS image data to be added.
        Each MEDS stamp's coadd ID and MEDS index are also stored.
        '''
        self.MEDS_stamps.append(MEDS_stamp)
        self.coadd_IDs.append(MEDS_stamp.coadd_ID[0])
        self.MEDS_indexes.append(MEDS_stamp.MEDS_index)

        
       
        
        
    def add_models(self, galaxy_models_table):
        '''
        Reads model parameters from a GalaxyModelsTable and stores them into the corresponding MEDS_stamp objects.
        - galaxy_models_table: The table containing galaxy model parameters.
        This function matches the models to the MEDS stamps based on their IDs, and handles multiple entries for epochs.
        '''
        
        # generate columns of matched positions
        index_to_match = np.arange(len(galaxy_models_table.id_array))
        data_ = {'pos': index_to_match}
        df1 = pd.DataFrame(data = data_, index = galaxy_models_table.id_array)
        df2 = pd.DataFrame(index = self.coadd_IDs)   
        
        self.pos = np.array(df2.join(df1).loc[self.coadd_IDs,'pos'])
        self.model_index = np.array(df2.join(df1).loc[self.coadd_IDs].index)
   
        # generate columns of matched positions for the epochs, which are in another table
        # and have multiple entries
        index_to_match = np.arange(len(galaxy_models_table.id_epoch_array))
        df1 = pd.DataFrame(data = {'pos': index_to_match} , index = galaxy_models_table.id_epoch_array)
        self.pos_epoch = (df2.join(df1))


                                  
                                  
        for MEDS_stamp, pos, model_index in zip(self.MEDS_stamps, self.pos, self.model_index):
            
            models_parameters = dict()
            
            try:
                pos_epoch = (np.array(self.pos_epoch.loc[ MEDS_stamp.coadd_ID[0]]).astype(np.int))[:,0]
            except:
                pos_epoch = (np.array(self.pos_epoch.loc[ MEDS_stamp.coadd_ID[0]]).astype(np.int))[0]
                
            for band in MEDS_stamp.bands:
                try:
                    models_parameters[band] = galaxy_models_table.return_model(band = band, pos = pos, pos_epoch = pos_epoch)
                except:
                    print ('failed rendering model for entry {0}, band [{1}]'.format( MEDS_stamp.MEDS_index,band))
                    MEDS_stamp.flags += 2
    
                
            MEDS_stamp.model_parameters = models_parameters 
            
            



        
    def render_models(self, MEDS_index = 0, render_self = False, render_others = True, use_COADD_only = True):
        '''
        Renders models onto the MEDS stamps.
        - MEDS_index: Index of the MEDS stamp to render models on.
        - render_self: Flag to indicate if the model of the MEDS stamp itself should be rendered.
        - render_others: Flag to indicate if models of other objects should be rendered.
        - use_COADD_only: Flag to use only the coaddition image for rendering.
        This function handles the rendering of models onto the image stamps, including both the target object and others in the field.
        '''
        flag_render_model = 0
        
        model_rendered = []
        model_rendered_all = []
        model_rendered_flag = []
        
        for b, band in enumerate(self.MEDS_stamps[MEDS_index].bands):
            model_rendered_band = []
            model_rendered_all_band = []  
            model_rendered_flag_band = []
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 0
                end = self.MEDS_stamps[MEDS_index].ncutout[b]
            
            for i in range(start, end): 
                size_image = self.MEDS_stamps[MEDS_index].imlist[b][i].shape[0]    
                rendered_image = np.zeros((size_image,size_image))

                try:
                    wcs = self.MEDS_stamps[MEDS_index].make_WCS_2objects(self.MEDS_stamps[MEDS_index], self.MEDS_stamps[MEDS_index].bands[b], i)
                    self_model,jac = render_model_on_stamp(self.MEDS_stamps[MEDS_index].model_parameters[band]['gal_pars'],
                                                self.MEDS_stamps[MEDS_index].model_parameters[band]['psf_pars'],wcs,size_image)

                except:
                    flag_render_model+=100
                    self_model = np.zeros((size_image,size_image))

                    
                if render_others:

                    # find a lit
                    list_MEDS_indexes = np.unique(self.MEDS_stamps[MEDS_index].seglist[b][0].flatten())
                    list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=0]
                    list_MEDS_indexes = list_MEDS_indexes[list_MEDS_indexes!=self.MEDS_stamps[MEDS_index].MEDS_index+1]


                    for mute_index in list_MEDS_indexes:
                        MEDS_index_j = np.array(self.MEDS_indexes)[np.in1d(np.array(self.MEDS_indexes),mute_index-1)][0]
                    
                        try:
                            wcs = self.MEDS_stamps[MEDS_index].make_WCS_2objects(self.MEDS_stamps[MEDS_index_j], self.MEDS_stamps[MEDS_index].bands[b], i)
                            rendered_image_j,jac = render_model_on_stamp(self.MEDS_stamps[MEDS_index_j].MOF_models[band]['gal_pars'],
                                                              self.MEDS_stamps[MEDS_index].MOF_models[band]['psf_pars'],wcs,
                                                              self.MEDS_stamps[MEDS_index].imlist[b][i].shape[0],  g1 = 0., g2 = 0.)                    
                            rendered_image += rendered_image_j
                            model_rendered_flag_band.append(True)
                        except:
                            flag_render_model +=1
                            model_rendered_flag_band.append(False)
                            pass
                                
                model_rendered_band.append(rendered_image)
                model_rendered_all_band.append(rendered_image+self_model)
                
            model_rendered.append(model_rendered_band) 
            model_rendered_all.append(model_rendered_all_band) 
            model_rendered_flag.append(model_rendered_flag_band)
        self.MEDS_stamps[MEDS_index].model_rendered = model_rendered
        self.MEDS_stamps[MEDS_index].model_all_rendered = model_rendered_all
        self.MEDS_stamps[MEDS_index].flag_render_model = flag_render_model
        self.MEDS_stamps[MEDS_index].model_rendered_flag = model_rendered_flag
             
        

        
            
            


class MedsStamp:
    '''
    This class is designed to manage and process individual MEDS stamps, which are subsections of astronomical images. It facilitates the computation of various quantities relevant to the BFD (Balanced Fourier Descriptor) pipeline, such as moments, PSD (Power Spectral Density) parameters, etc.
    '''

    def __init__(self, MEDS_index, meds_array=[], bands=[]):
        '''
        Initializes the MedsStamp instance.
        - MEDS_index: Index of the MEDS stamp in the larger MEDS array.
        - meds_array: Array containing MEDS data.
        - bands: List of bands (spectral ranges) associated with the MEDS stamp.
        Attributes like coadd_ID, xyshift, moments, flags, and model parameters are also initialized.
        '''
        self.MEDS_index = MEDS_index
        self.coadd_ID = [m['id'][MEDS_index] for m in meds_array]
        
        self.bands = bands
        self.n_bands = len(bands)
        self.xyshift = None
        self.moments = None
        self.flags = 0
        self.model_parameters = None
        self.index_model = None

    def check_stamp_masked_frac(self, limit=0.1, use_COADD_only = False):
        '''
        Checks which bands have exposures with a fraction of pixels masked above a specified limit.
        - limit: The threshold for considering a pixel as masked.
        - use_COADD_only: Flag to consider only the coadded image.
        Returns a list of bands not significantly masked.
        '''
        self.size =  [[len(self.masklist[b][i].flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        #self.mfrac = [[1.*len(np.zeros(self.size[b][i])[(self.wtlist[b][i].flatten()==0.) | (self.masklist[b][i].flatten()!=0.)])/self.size[b][i] for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        self.mfrac_flag = [[True for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        #ed = timeit.default_timer()
        #
        
        
        # gaussian aperture ****
        psf_fwhm = 2.
        Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)

        psf_pars = [0.0, 0.0, 0.0, 0.00, Tpsf, 1.0]
        psf_gmix = ngmix.GMixModel(pars=psf_pars, model="turb")

        
        
        self.mfrac = [[np.sum((psf_gmix.make_image((self.imlist[b][i].shape), jacobian=ngmix.jacobian.Jacobian(row=self.wcslist[b][i].xy0[1],
                      col=self.wcslist[b][i].xy0[0],
                      dudrow=self.wcslist[b][i].jac[0,1],
                      dudcol=self.wcslist[b][i].jac[0,0],
                      dvdrow=self.wcslist[b][i].jac[1,1],
                      dvdcol=self.wcslist[b][i].jac[1,0]),fast_exp=True)).flatten()[((self.wtlist[b][i].flatten()==0.) | (self.masklist[b][i].flatten()!=0.))])/np.sum((psf_gmix.make_image((self.imlist[b][i].shape), jacobian=ngmix.jacobian.Jacobian(row=self.wcslist[b][i].xy0[1],
                      col=self.wcslist[b][i].xy0[0],
                      dudrow=self.wcslist[b][i].jac[0,1],
                      dudcol=self.wcslist[b][i].jac[0,0],
                      dvdrow=self.wcslist[b][i].jac[1,1],
                      dvdcol=self.wcslist[b][i].jac[1,0]),fast_exp=True)).flatten()) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
        mfrac_per_band = []
        for b in range(len(self.bands)):
            mfrac_per_band.append(np.mean(np.array(self.mfrac[b])))
        self.mfrac_per_band = mfrac_per_band
        bands_not_masked = dict()
        for b in range(len(self.bands)):
            bands_not_masked[self.bands[b]] = True
            if use_COADD_only:
                if self.mfrac[b][0] > limit:
                    bands_not_masked[self.bands[b]] = False
                    self.mfrac_flag[b][0] = False
            else:
                bands_not_masked[self.bands[b]] = False
                for i in range(1, self.ncutout[b]):  #MG *******+
                    # at least one exposure needs to make it
                    if self.mfrac[b][i] > limit:
                        self.mfrac_flag[b][i] = False
                    if self.mfrac[b][i] < limit:
                        bands_not_masked[self.bands[b]] = True
                  
        bands_not_masked_list = []
        for b in bands_not_masked.keys():
            if bands_not_masked[b]:
                 bands_not_masked_list.append(b)
        self.bands_not_masked = bands_not_masked_list
        return bands_not_masked_list


    def compute_moments(self, sigma = 3, FFT_pad_factor = 2., use_COADD_only = False, bands = ['g','r', 'i', 'z'], 
                                  bands_weights = [0.,0.7,0.2,0.1] , Detectinator_=False):
        '''
        Computes moments by combining exposures and bands.
        - sigma, FFT_pad_factor: Parameters for the BFD filter.
        - use_COADD_only: Flag to consider only the coadded image.
        - bands: List of bands to consider.
        - bands_weights: Weights for different bands.
        - Detectinator_: Flag to use a specialized detection algorithm.
        This method is key for extracting scientific information from the image.
        '''
        
        BFD_filter = mc.KBlackmanHarris(sigma = sigma) 
        
        images_array = []
        psf_array = []
        wcs_array = []
        noise_array = []
        band_array = []
        
        psf_moments = np.zeros(4)
        psf_weights = 0
        
        for  index_band,band in enumerate(bands):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[index_band]
                
            for exp in range(start, end):  
                compute = True
                # check if we can use the exposure
                if self.mfrac_flag[index_band][exp]:
                    pass
                else:
                    compute = compute & False
                if self.explist_mask is not None:
                    if self.explist_mask[index_band][exp-1]:
                        pass
                    else:
                        compute = compute & False
            
                if compute:
                    images_array.append(self.imlist[index_band][exp] - self.model_rendered[index_band][exp])
                    psf_array.append(self.psf[index_band][exp] )
                    
                    noise_rms = (1./np.sqrt(np.median(self.wtlist[index_band][exp][self.masklist[index_band][exp] == 0])))
                    noise_array.append(noise_rms)
                    
                    band_array.append(self.bands[index_band])
                    
                    
                    
                    # Detectinator ---------------------------------------------------------------------------------
                    '''     
                    I think this has to be 1 coherent shift for th galaxy exposures, and proabbly should be a weighted average of all the
                    exposures shifts, or something like that.
                    duv_dxy = np.array([[self.wcslist[index_band][exp].jac[0,0], self.wcslist[index_band][exp].jac[0,1]],
                                        [self.wcslist[index_band][exp].jac[1,0], self.wcslist[index_band][exp].jac[1,1]]])
                    
                    
                    dx,dx_init,kval,ku,kv,d2k,conjugate,kvar,mf_map,mf_cov = Detectinator(img,
                                                                          psf=self.psf[index_band][exp],
                                                                          noise_rms=noise_rms,
                                                                          sn=3,
                                                                          duv_dxy=duv_dxy,
                                                                          minsep=5)
                    wcs_exposure = copy.deepcopy(self.wcslist[index_band][exp])
    
                    try:
                        shift_ =  (dx[0][0]-dx_init[0][0])**2+(dx[0][1]-dx_init[0][1])**2
                        wcs_exposure.xy0 = dx[0]
                        wcs_array.append(wcs_exposure)
                    except:
                        wcs_array.append(self.wcslist[index_band][exp])
                    '''
                    wcs_array.append(self.wcslist[index_band][exp])
                        
                    # -----------------------------------------------
                    # Compute PSF moments, for diagnosticss
                    nominal = np.array(self.psf[index_band][exp].shape) // 2
                    psf_shift = bfd.momentcalc.xyWin(self.psf[index_band][exp], sigma=2, nominal=nominal)
                    
                    origin = (0.,0.)
                    cent = (nominal+[psf_shift[1],psf_shift[0]]) 
                    duv_dxy = np.array([[self.wcslist[index_band][exp].jac[0,0], self.wcslist[index_band][exp].jac[0,1]],
                                        [self.wcslist[index_band][exp].jac[1,0], self.wcslist[index_band][exp].jac[1,1]]])
                    wcs_psf = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)


                    delta_stamp = np.zeros_like(self.psf[index_band][exp])
                    delta_stamp[len(self.psf[index_band][exp])//2,len(self.psf[index_band][exp])//2] = 1.
                                       
                    psf_kd = bfd.multiImage([self.psf[index_band][exp]], (0,0), [delta_stamp], [wcs_psf], 
                             pixel_noiselist = [noise_rms], bandlist = ['x'],
                             pad_factor=FFT_pad_factor) #, psf_recenter_sigma = 2.)
                    
                    psf_moment = bfd.MultiMomentCalculator(psf_kd, BFD_filter, bandinfo = {'bands':['x'],'weights':[1.]})
                    

                    Mf,Mr,M1,M2,_ = psf_moment.get_moment(0.,0.).even
                    psf_moments += np.array([Mf,Mr,M1,M2])
                    psf_weights += bands_weights[index_band]
                    #'''               
                                     

        self.psf_moments  = psf_moments / psf_weights   
                        
        # compute image moments ----
        kds = bfd.multiImage(images_array, (0,0), psf_array, wcs_array, 
                             pixel_noiselist = noise_array, bandlist = band_array,
                             pad_factor=FFT_pad_factor, psf_recenter_sigma = 2.)
        
        
        
        bandinfo = {'bands':bands, 'weights':bands_weights,'index': np.arange(len(bands))} 

        multi_moment = bfd.MultiMomentCalculator(kds, BFD_filter, bandinfo = bandinfo)
        self.xyshift, error,msg = multi_moment.recenter()
        self.moments = multi_moment

        
    def compute_noise(self):
        '''
        Computes the noise level in the image based on weight maps and masks.
        '''
        self.noise_rms =  [[np.sqrt(1./np.median(self.wtlist[b][i][(self.wtlist[b][i]!= 0.) & (self.masklist[b][i] == 0)])) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
        
 
            
    
    def deal_with_bmask(self, use_COADD_only = False):
        '''
        Processes the bitmask of the image to handle masked or bad pixels.
        - use_COADD_only: Flag to consider only the coadded image.
        '''
        for b in range(len(self.bands)):
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[b]            
            for i in range(start, end):  
                    
                bmask = copy.deepcopy(self.masklist[b][i])
                bmask |= np.rot90(self.masklist[b][i])
    
                self.imlist[b][i][bmask==0] += self.model_all_rendered[b][i][bmask==0]

                s0 = self.model_all_rendered[b][i].shape[0]
                
                self.imlist[b][i][(bmask==0) & (self.wtlist[b][i]!=0.)] += (np.random.normal(size = (s0,s0))[(bmask==0) & (self.wtlist[b][i]!=0.)]/np.sqrt(self.wtlist[b][i])[(bmask==0) & (self.wtlist[b][i]!=0.)])
                                                                
        
        
    def Load_MEDS(self, index, meds_array = [], mask_exposure = None):
        '''
        Loads MEDS data into memory for a specific detection.
        It processes images, segmentation maps, weight maps, bitmasks, Jacobians, and PSFs for different exposures.
        - index: Index of the MEDS stamp.
        - meds_array: Array containing MEDS data.
        - mask_exposure: Optional mask to filter out bad exposures.
        n.b: First cutout for each band is always the COADD
        '''
        
        if meds_array != None:
            self.image_ra = [m['ra'][index] for m in meds_array]
            self.image_dec = [m['dec'][index] for m in meds_array]
            self.ncutout = [m['ncutout'][index] for m in meds_array]
            self.imlist = [m.get_cutout_list(index) for m in meds_array]
   

            self.orig_rowcol = [[ (meds_array[b]['orig_row'][index][i],meds_array[b]['orig_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
            self.orig_start_rowcol = [[ (meds_array[b]['orig_start_row'][index][i],meds_array[b]['orig_start_col'][index][i]) for i in range((self.ncutout[b]))] for b in range(len(self.bands))] 
     
            # row,col in the cutouts. Is generally close to center of the tile.
            self.rowcol = [[meds_array[b].get_cutout_rowcol(index,i) for i in range((self.ncutout[b]))] for b in range(len(self.bands))]
            self.seglist = [m.get_cutout_list(index, type='seg') for m in meds_array]
            self.wtlist = [m.get_cutout_list(index, type='weight') for m in meds_array]
            self.masklist = [m.get_cutout_list(index, type='bmask') for m in meds_array]
            self.jaclist = [m.get_jacobian_list(index) for m in meds_array]
            

            #get ccd numbers ----
            self.ccd_name =  [[ 1000*b+np.int((meds_array[b]._image_info['image_path'][meds_array[b]['file_id'][index][i]]).split('_')[2].strip('c')) for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))]  
            self.expnum =  [[ np.int((meds_array[b]._image_info['image_path'][meds_array[b]['file_id'][index][i]]).split('_')[0].strip('red/D00')) for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 
          

            # Make a mask in case any of the exposures are in the bad exposures list
           # '''
            if mask_exposure is not None:        
                self.expnum_ccd =  [[ (self.ccd_name[b][i-1]-1000*b)+100*self.expnum[b][i-1]  for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))] 
                self.explist_mask =  [[ ~((self.expnum_ccd[b][i-1] in mask_exposure[self.bands[b]]) & (self.expnum[b][i-1] in mask_exposure['all']))  for i in range(1,(self.ncutout[b]))] for b in range(len(self.bands))]
            else:
                self.explist_mask = None
            

            # save DESDM coordinates ----
            self.DESDM_coadd_x = [m['input_row'][index] for m in meds_array]
            self.DESDM_coadd_y = [m['input_col'][index] for m in meds_array]
            
            

           
            try:
                self.psf = [m.get_cutout_list(index, type='psf') for m in meds_array]
            except:
                if self.verbose > 0:
                    print ('No PSF extension in this file')
        

    def make_WCS(self):
        '''
        Generates WCS (World Coordinate System) information for the MEDS stamp.
        This is essential for astronomical image processing to understand the spatial orientation of the image.
        '''
        
        self.wcslist = []
        for index_band in range(self.n_bands):
            mute_bands =[]
            for exp in range(self.ncutout[index_band]):  
                jac = self.jaclist[index_band][exp]
                cent=(jac['col0'],jac['row0'])
                origin = (0.,0.)
                duv_dxy = np.array( [ [jac['dudcol'], jac['dudrow']],
                                      [jac['dvdcol'], jac['dvdrow']] ])
           
                mute_bands.append(bfd.WCS(duv_dxy,xyref=cent,uvref=origin))
            self.wcslist.append(mute_bands)
            
            
            
        
    def make_WCS_2objects(self, obj2, band, exp,return_shift=False):
        '''
        it makes the BFD wcs for a secon obj Image provided by the user.
        It requires to sepcify the band and the exposure.
        '''
        pos_band = np.arange(self.n_bands)[np.in1d(self.bands,band)][0]
        
        '''
        rowcol are ~ the coordinates of the center of the cutout (i.e. where 'self' is located).
        orig_rowcol are the coordinates wrt the tile, such that (self.orig_rowcol - obj.orig_rowcol) corresponds to the
        coordinate difference between the two objects' center.
        '''

        drow = self.rowcol[pos_band][exp][0] - (self.orig_rowcol[pos_band][exp][0] - obj2.orig_rowcol[pos_band][exp][0])
        dcol = self.rowcol[pos_band][exp][1] - (self.orig_rowcol[pos_band][exp][1] - obj2.orig_rowcol[pos_band][exp][1])
        
    
        nbrxyref = (dcol,drow)
        nbruvref = (0,0)
        jac = self.jaclist[pos_band][exp]
        nbrduv_dxy = np.array( [ [jac['dudcol'], jac['dudrow']],
                                 [jac['dvdcol'], jac['dvdrow']]])
        nbrwcs = bfd.WCS(nbrduv_dxy, xyref=nbrxyref, uvref=nbruvref)
        if return_shift:
            return nbrwcs,nbrxyref
        else:
            return nbrwcs
            
  
            

   
        

            
    def measure_psf_HSM_moments(self,bands_weights = [], use_COADD_only = False):
        
        psf_moments = np.zeros(4)
        for b in range(len(self.bands)): 
            psf_moments__ = np.zeros(4)
            psf_moments__w = np.zeros(4)
            
            if use_COADD_only:
                start = 0
                end = 1
            else:
                start = 1
                end = self.ncutout[b]
                
            for i in range(start,end):
                y = self.orig_rowcol[b][i][0]+1
                x = self.orig_rowcol[b][i][1]+1

                img = galsim.Image(self.psf[b][i],copy=True)

                jac = self.jaclist[b][i]
                jac = galsim.JacobianWCS(dudx=jac['dudcol'], dudy=jac['dudrow'], dvdx=jac['dvdcol'], dvdy=jac['dvdrow'])


                # calculate hsm moments -----
                try:
                    mom = galsim.hsm.FindAdaptiveMom(img)
                    sigma = mom.moments_sigma
                    shape = mom.observed_shape
                    scale, shear, theta, flip = jac.getDecomposition()
                    sigma *= scale
                    # Fix shear.  First the flip, if any.
                    if flip:
                        shape = galsim.Shear(g1 = -shape.g1, g2 = shape.g2)
                    # Next the rotation
                    shape = galsim.Shear(g = shape.g, beta = shape.beta + theta)
                    # Finally the shear
                    shape = shear + shape
                    e1 = shape.g1
                    e2 =  shape.g2

                    ssq = sigma**2 / (1-e1**2-e2**2)**0.5
                    M = [[(1+e1)*ssq, e2*ssq],[e2*ssq, (1-e1)*ssq]]


                    weight = bands_weights[b]

                    psf_moments__ += np.array([weight*ssq,weight*M[0][0],weight*M[1][1],weight*M[0][1]])
                    psf_moments__w += np.array([1,1,1,1])
                except:
                    pass
            psf_moments__ = psf_moments__/psf_moments__w
            psf_moments+=psf_moments__
            
        # compute psf ellipticities:    
        sigma_ = np.sqrt(np.sqrt(psf_moments[1]*psf_moments[2]-psf_moments[3]**2))
        e1_ = psf_moments[1]/psf_moments[0]-1
        e2_ = psf_moments[3]/psf_moments[0]
 
        self.psf_hsm_moments = [sigma_,e1_,e2_]

        
   
            
            
            
    def return_orig_coordinates(self):
        '''
        Returns the original coordinates of the image within its larger context (e.g., the full astronomical survey).
        Useful for mapping the stamp back to its source location.
        '''
        orig_row_flattened = np.zeros(70)
        orig_col_flattened = np.zeros(70)
        ccd_name_flattened = -np.ones(70)


        u_1 = np.array([ll[0] for l in self.orig_rowcol for ll in l[1:]])
        u_2 = np.array([ll[1] for l in self.orig_rowcol for ll in l[1:]])
        u_3 = np.array([ll for l in self.ccd_name for ll in l])


        orig_row_flattened[:len(u_1)] = u_1 
        orig_col_flattened[:len(u_1)] = u_2 
        ccd_name_flattened[:len(u_1)] = u_3 
        return orig_row_flattened,orig_col_flattened, ccd_name_flattened
    
            
            
            
    def subtract_background(self):
        '''
        Recomputes and subtracts the background from the image based on the outermost pixels.
        This method is used for background noise reduction.
        '''
        
        bkg_tot = 0
        count = 0
        len_v = 0
        for b in range(len(self.bands)):
            for i in range((self.ncutout[b])):
                
                
                    seg = copy.deepcopy(self.seglist[b][i])
                    segg0 = copy.deepcopy(self.seglist[b][i])
                    for ii in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                        seg += (np.roll(segg0, ii, axis = 0) + np.roll(segg0, ii, axis = 1))

                    mask__ = copy.deepcopy(self.masklist[b][i] )
                    segg0 = copy.deepcopy(self.masklist[b][i] )
                    for ii in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
                        mask__ += (np.roll(segg0, ii, axis = 0) + np.roll(segg0, ii, axis = 1))


                    mask = (seg == 0) & (mask__ == 0) 
                    maskarr=np.ones(np.shape(mask),dtype='int')
                    maskarr[~mask] = 0
                    uu = 6
                    maskarr[uu:-uu,uu:-uu]=0
                    v = self.imlist[b][i][np.where(maskarr==1)]

                    if len(v)>50: 
                        correction = np.median(v)
                        self.imlist[b][i] -= correction
                        bkg_tot += correction
                        count += 1
                        len_v += len(v)               
                        
        if count ==0:
            self.bkg = bkg_tot
            self.pixel_used_bkg = 1e10
        else:
            self.bkg = bkg_tot/count
            self.pixel_used_bkg = len_v/count

            
            
            
            
            
    def zero_padd_psf(self):
        '''
        Zero-pads the PSF solutions so they match the size of the images.
        This is important for ensuring consistency in image processing.
        '''
        
        for index_band in range(self.n_bands):
                size_x = self.imlist[index_band][0].shape[0]
                size_y = self.imlist[index_band][0].shape[1]
                for exp in range(self.ncutout[index_band])[::-1]:            
                    mute = np.zeros((size_x,size_y))
                    size_psf_x = self.psf[index_band][exp].shape[0]
                    size_psf_y = self.psf[index_band][exp].shape[1]
                    # sometimes the PiFF stamp is larger than the image
                    if size_psf_x>size_x:
                        dx = -np.int((size_x-size_psf_x)/2)
                        self.psf[index_band][exp] = self.psf[index_band][exp][dx:dx+size_x,:][:,dx:dx+size_x]
                    # if the PiFF stamp is smaller than the image
                    elif size_psf_x<size_x:
                        dx = np.int((size_x-size_psf_x)//2)+1
                        mute[dx:dx+size_psf_x,:][:,dx:dx+size_psf_x] = self.psf[index_band][exp]
                        self.psf[index_band][exp] = mute
            
            

            
            
            
            
        
        
            
            
            
