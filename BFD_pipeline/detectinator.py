import numpy as np
from scipy.optimize import root
import bfd

class WCS(object):
    '''A simple WCS, which will be an affine transformation from pixel coords
    xy to some sky coordinates uv.'''
    def __init__(self, duv_dxy, xyref=[0,0],uvref=[0,0]):
        ''' Define an affine transformation from pixel to sky coordinates
        (uv - uvref) = duv_dxy * (xy - xyref)
        -----
        parameters
        duv_dxy:   The 2x2 Jacobian matrix such that duv_dxy[i,j] = du_i / dx_j
        xyref:     2-element pixel coordinates of reference point
        uvref:     2-element sky coordinates of reference point
        '''
        self.jac=duv_dxy
        self.jacinv = np.linalg.inv(self.jac)
        self.xy0=np.array(xyref)
        self.uv0=np.array(uvref)
    
        return
        
    def getxy(self,uv):
        # Map from uv to xy.  Input can be of shape (2) or (2,N)
        return np.dot(self.jacinv, (uv - self.uv0)) + self.xy0

    def getuv(self,xy):
        # Map from xy to uv.  Input can be of shape (2) or (2,N)
        return np.dot(self.jac,(xy - self.xy0)) + self.uv0
    
    def getuv_k(self, kxy):
        # Map from k_x, k_y to k_u, k_v.  Input can be of shape (2) or (2,N)
        return np.dot(self.jacinv.T, kxy)
    
    def getdet(self):
        # Get determinant | duv / dxy |
        return np.linalg.det(self.jac)



class Detection(object):

    """
    A class representing a detection in an image.

    This class takes care of setting up the necessary parameters and 
    functions for modeling the detection in an image.

    Parameters
    ----------
    duv_dxy : numpy.ndarray, optional
        The conversion from detector pixels to sky coordinates. Default is np.array([[0.263, 0.], [0., 0.263]]).
    tile_size : numpy.ndarray, optional
        The size of the image tile. Default is np.array([60, 60]).
    pad_factor : int, optional
        The factor by which to pad the image tile. Default is 3.
    sigma : float, optional
        The sigma parameter used for reweighting. Default is 0.65.

    Attributes
    ----------
    duv_dxy : numpy.ndarray
        The conversion from detector pixels to sky coordinates.
    xyref : numpy.ndarray
        The reference point in the image tile.
    uvref : numpy.ndarray
        The reference point in sky coordinates.
    tile_size : numpy.ndarray
        The size of the image tile.
    pad_factor : int
        The factor by which to pad the image tile.
    reweighting : float
        The reweighting factor for the image.
    wcs_ : astropy.wcs.WCS
        The world coordinate system for the image.
    psf_ : numpy.ndarray
        The point spread function for the image.
    kpsf : numpy.ndarray
        The Fourier transform of the point spread function.
    ku : numpy.ndarray
        The Fourier space coordinates in the u direction.
    kv : numpy.ndarray
        The Fourier space coordinates in the v direction.
    d2k : float
        The square of the grid spacing in Fourier space.
    ksq : numpy.ndarray
        The squared magnitude of the wave vector in Fourier space.
    conjugate : numpy.ndarray
        The conjugate of the point spread function in Fourier space.
    w_blackman : numpy.ndarray
        The Blackman-Harris window function in Fourier space.
    """

    def __init__(
        self,
        duv_dxy = np.array([[0.263, 0.],
                             [0., 0.263]]),
        tile_size = np.array([60,60]),
        pad_factor = 3,
        sigma = 0.65
    ):
        """
        Initialize the detection object.

        Parameters
        ----------
        psf_pars : array-like, optional
            The parameters for the point spread function (PSF) of the image.
        duv_dxy : numpy.ndarray, optional
            The conversion from detector pixels to sky coordinates.
        tile_size : numpy.ndarray, optional
            The size of the image tile.
        pad_factor : int, optional
            The factor by which to pad the image tile.
        sigma : float, optional
            The sigma parameter used for reweighting.

        Notes
        -----
        This method sets the initial values of the attributes of the detection 
        object, including the conversion from detector pixels to sky 
        coordinates, the reference points in the image tile and sky 
        coordinates, the parameters for the PSF, the size of the image tile, 
        and the padding factor. It also calculates the reweighting factor and 
        sets up the world coordinate system for the image.
        """

        self.duv_dxy = duv_dxy
        self.xyref = tile_size/2.
        self.uvref = np.dot(duv_dxy,self.xyref)
        
        self.tile_size = tile_size
        self.pad_factor = pad_factor

        self.sigma = sigma

        self.reweighting = 4*np.pi*np.pi/np.linalg.det(duv_dxy)

        return

    def _set_wcs(self):
        """
        Set up the world coordinate system for the image.
        
        This method sets the `wcs_` attribute of the detection object using the
        `duv_dxy`, `xyref`, and `uvref` attributes.

        The `duv_dxy` attribute contains the conversion from detector pixels to 
        sky coordinates, and `xyref` and `uvref` are the reference points in the 
        image tile and sky coordinates, respectively. The `wcs_` attribute is 
        an instance of the `astropy.wcs.WCS` class, which defines the mapping 
        between pixel coordinates and sky coordinates.

        Returns
        -------
        None
        """
        self.wcs_ = WCS(self.duv_dxy,xyref=self.xyref,uvref=self.uvref)
        return

    def _set_kpsf(self):
        """
        Set the kernel PSF used in deconvolution.

        Parameters
        ----------
        kpsf : array_like or `None`
            The kernel PSF used in deconvolution. This should be an array
            of the same shape as the data array. If `None` is provided,
            the PSF is assumed to be a delta function.

        Returns
        -------
        None
        """

        psf = np.pad(self.psf_,self.tile_size,mode='constant',constant_values=0)
        
        kpsf = np.fft.rfft2(psf)

        self.kpsf = np.divide(kpsf,kpsf[0,0])

        return

    def _set_k(self):
        """
        Set the Fourier space coordinates and grid spacing.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method calculates the Fourier space coordinates and grid spacing using the `pad_factor` and `tile_size` attributes of the detection object.

        The following attributes of the detection object are set by this method:

        * `ku`: Fourier space coordinates in the u direction.
        * `kv`: Fourier space coordinates in the v direction.
        * `d2k`: Square of the grid spacing in Fourier space.
        * `ksq`: Squared magnitude of the wave vector in Fourier space.

        """

        N = self.pad_factor*self.tile_size[0]
        kx = np.fft.fftfreq(N)[:N//2+1]*2*np.pi
        kx[-1] *= -1.
        self.d2k = (kx[1]-kx[0])**2
        ky = np.fft.fftfreq(N)*2*np.pi
        kxx,kyy = np.meshgrid(kx,ky)

        kxy = np.vstack((kxx.flatten(),kyy.flatten()))
        kuv = self.wcs_.getuv_k(kxy)
        self.ku  = kuv[0].reshape(kxx.shape)
        self.kv  = kuv[1].reshape(kyy.shape)
        self.kk = (self.ku + 1j*self.kv)**2

        self.d2k /= np.abs(self.wcs_.getdet())

        self.ksq = self.ku*self.ku+self.kv*self.kv
        
        return 

    def w_blackmanharris(self):
        """
        Set the Blackman-Harris window function in Fourier space.
        
        This method calculates the Blackman-Harris window function in Fourier 
        space using the `ku` and `kv` attributes, and sets the `w_blackman` 
        attribute of the detection object.
        
        Parameters
        ----------
        sigma: float, optional
            The width factor for the window function.
        """
        
        coeffs = np.array([0.35875, 0.48829, 0.14128, 0.01168])
        
        kshape = self.ku.shape
        
        kmax = 1.07635*np.pi/self.sigma
        
        kr = np.hypot(self.ku,self.kv).flatten()
        
        with np.errstate(divide='ignore'):
            _invkr = np.where( kr==0, 1., 1/kr).flatten()
            
        _u = kr * (np.pi / kmax)
        
        mask_flat = (kr <= kmax)
        
        _u_mask = _u[mask_flat]
        
        _cos_u_mask = np.cos(_u_mask)
        _cos_2u_mask = np.cos(2*_u_mask)
        _cos_3u_mask = np.cos(3*_u_mask)
        
        w_mask = (coeffs[0]
                  + coeffs[1]*_cos_u_mask
                  + coeffs[2]*_cos_2u_mask
                  + coeffs[3]*_cos_3u_mask)
        
        w = np.zeros(kshape).flatten()
        
        w[mask_flat] = w_mask
        w = w.reshape(kshape)
        
        w[1::2,::2] *= -1. 
        w[::2,1::2] *= -1.
        
        self.w_blackman= w
        
        return 

    def _set_conjugate(self):
        """
        Double the k-values that are missing their conjugates.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method sets the `conjugate` attribute of the detection object by 
        doubling the k-values that are missing their conjugates. This is 
        necessary for the manual inverse Fourier transforms.

        """
        self.conjugate = np.ones(self.w_blackman.shape, dtype=float)
        self.conjugate[:,1:self.pad_factor*self.tile_size[0]//2] *= 2.
        return 

    def _set_kw(self):
        self.ku_w = self.ku * self.w_blackman
        self.kv_w = self.kv * self.w_blackman
        self.k_MR = self.w_blackman * self.ksq
        self.k_M1 = self.w_blackman * self.kk.real
        self.k_M2 = self.w_blackman * self.kk.imag
        self.k_MC = self.w_blackman * self.ksq * self.ksq
        return


    def _set_self(self):
        """
        Set the values of the attributes of the detection object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method sets the values of the following attributes of the detection object:

        * `ku`: Fourier space coordinates in the u direction.
        * `kv`: Fourier space coordinates in the v direction.
        * `d2k`: Square of the grid spacing in Fourier space.
        * `ksq`: Squared magnitude of the wave vector in Fourier space.
        * `conjugate`: Conjugate of the point spread function in Fourier space.
        * `w_blackman`: Blackman-Harris window function in Fourier space.
        * `psf_`: Point spread function for the image.

        """
        self._set_wcs()
        self._set_k()
        self.w_blackmanharris()
        self._set_conjugate()
        self._set_kw()
        return

    def eMoments(self,dx,kval):

        _kval = kval * self.conjugate * self.d2k
        origin = np.dot(self.duv_dxy,dx+self.tile_size)
        phase = self.ku * origin[0] + self.kv * origin[1]

        kval_shifted = np.exp(1j*phase) * _kval
        del phase
        del _kval

        M0 = np.sum(self.w_blackman * kval_shifted.real)
        MR = np.sum(self.k_MR * kval_shifted.real)
        M1 = np.sum(self.k_M1 * kval_shifted.real)
        M2 = np.sum(self.k_M2 * kval_shifted.real)
        MC = np.sum(self.k_MC * kval_shifted.real)

        return np.array([M0,MR,M1,M2,MC])

    def moment_xy(self,dx,kval):
        """
        Calculate the moments in the x and y directions.

        This method calculates the moments in the x and y directions using the
        `ku`, `kv`, `ku_w`, `kv_w`, and `conjugate` attributes of the detection
        object, and the `dx` and `kval` arguments.

        Parameters
        ----------
        dx: array
            The x and y coordinates of the center of the tile.
        kval: array
            The Fourier transform of the tile.
        
        Returns
        -------
        array
            The moments in the x and y directions.
        """

        _kval = kval * self.conjugate * self.d2k

        origin = np.dot(self.duv_dxy,dx+self.tile_size)
        phase = self.ku * origin[0] + self.kv * origin[1]

        kval_shifted = np.exp(1j*phase) * _kval
        del phase
        del _kval

        MX = -np.sum(self.ku_w * kval_shifted.imag)
        MY = -np.sum(self.kv_w * kval_shifted.imag)

        del kval_shifted

        return np.array([MX, 
                          MY])

    def xy_jacobian(self,dx,kval):
        """
        Calculate the Jacobian matrix in the x and y directions.

        This method calculates the Jacobian matrix in the x and y directions using the
        `ku`, `kv`, `ku_w`, `kv_w`, and `conjugate` attributes of the detection object,
        and the `dx` and `kval` arguments.

        Parameters
        ----------
        dx: array
            The x and y coordinates of the center of the tile.
        kval: array
            The Fourier transform of the tile.
        
        Returns
        -------
        array
            The Jacobian matrix in the x and y directions.
        """
        _kval = kval * self.conjugate * self.d2k

        origin = np.dot(self.duv_dxy,dx+self.tile_size)
        phase = self.ku * origin[0] + self.kv * origin[1]

        kval_shifted = np.exp(1j*phase) * _kval
        del phase
        del _kval
        
        MR = np.sum(self.k_MR * kval_shifted.real)
        M1 = np.sum(self.k_M1 * kval_shifted.real)
        M2 = np.sum(self.k_M2 * kval_shifted.real)

        del kval_shifted
        
        return - 0.5 * np.array([[MR+M1,M2],
                                  [M2,MR-M1]])


    def max_mapper(self,tile,psf,sn=5,noise_rms=None):
        """
        Calculate the maximum flux moment map.

        This method calculates the maximum flux moment map using the `tile` argument
        and the `moment_xy` and `xy_jacobian` methods of the detection object.

        Parameters
        ----------
        tile: array
            The tile of the image.
        sn: float
            The signal-to-noise ratio threshold for the maximum flux moment map.
        noise_rms: float
            The root mean square of the noise in the image.
        
        Returns
        -------
        array
            The maximum flux moment map.
        array
            The Fourier transform of the tile.
        array
            The variance of the transform of the tile.
        array
            The flux moment map.
        float
            The covariance of the flux moment map.
        """

        padded_tile = np.pad(tile,self.tile_size,mode='constant',constant_values=0)

        kval = np.fft.rfft2(padded_tile)
        del padded_tile
        
        self.psf_ = psf
        self._set_kpsf()
        kpsf = self.kpsf
        del psf
        
        kval = np.divide(kval,kpsf)
        
        kvar = (np.ones_like(kval,dtype=float) \
            * (noise_rms*self.pad_factor*self.tile_size[0])**2 \
            / (kpsf.real*kpsf.real))

        mf_cov = np.sum( (self.w_blackman * self.w_blackman) * kvar * self.d2k * self.d2k * self.conjugate)

        mf_map = self.reweighting * np.fft.irfft2(kval*self.w_blackman).real[self.tile_size[0]:-self.tile_size[0],
                                                                                self.tile_size[1]:-self.tile_size[1]] / np.sqrt(mf_cov)


        map_maxima = mf_map[1:-1,1:-1]
        max_map = (map_maxima-mf_map[:-2,1:-1]>0.) & \
                  (map_maxima-mf_map[1:-1,:-2]>0.) & \
                  (map_maxima-mf_map[2:,1:-1]>0.) & \
                  (map_maxima-mf_map[1:-1,2:]>0.) & \
                  (map_maxima-mf_map[:-2,:-2]>0.) & \
                  (map_maxima-mf_map[:-2,2:]>0.) & \
                  (map_maxima-mf_map[2:,:-2]>0.) & \
                  (map_maxima-mf_map[2:,2:]>0.) & (mf_map[1:-1,1:-1]>sn)
        del map_maxima
        
        return max_map,kval,kvar,mf_map,mf_cov

    def detector(self,tile,psf,noise_rms=None,sn=5):
        """
        Identify the position of the maxima flux moment map.

        This method identifies the position of the maxima flux moment map using the
        `max_mapper` method of the detection object.

        Parameters
        ----------
        tile: array
            The tile of the image.
        sn: float
            The signal-to-noise ratio threshold for the maximum flux moment map.
        noise_rms: float
            The root mean square of the noise in the image.

        Returns
        -------
        array
            The recentered x and y coordinates of the maxima in the flux moment map.
        array
            The initial x and y coordinates of the maxima in the flux moment map.
        array 
            The Fourier transform of the tile.
        array
            The variance of the transform of the tile.
        array
            The flux moment map.
        """

        max_map,kval,kvar,mf_map,mf_cov = self.max_mapper(tile,psf,noise_rms=noise_rms,sn=sn)
        maxima_idx = np.array(np.argwhere(max_map)[:,::-1]+1,dtype='float')
        return maxima_idx,kval,kvar,mf_map,mf_cov

    def multi_recenter(self,tile,psf,noise_rms=None,sn=5,minsep=5,tol=1e-3):

        dx_init,kval,kvar,mf_map,mf_cov = self.detector(tile,psf,noise_rms=noise_rms,sn=sn)
        
        dx_init = dx_init[np.sqrt(np.sum((dx_init-self.tile_size/2)**2,axis=1))<minsep]
        M0 = []
        dx = np.zeros_like(dx_init,dtype='float')
        for i in range(dx.shape[0]):
            dx[i] = root(
                fun=self.moment_xy,
                x0=dx_init[i],
                jac=self.xy_jacobian,
                args=(kval),
                tol=tol).x
            M0.append(self.eMoments(dx[i],kval)[0])
            
        mf_sort = np.array(M0).argsort()
        dx_init = dx_init[mf_sort]
        dx = dx[mf_sort]

            
            
        return dx,dx_init,kval,self.ku,self.kv,self.d2k,self.conjugate,kvar,mf_map,mf_cov



def Detectinator(
    image,
    noise_rms=1,
    sn=1,
    psf=None,
    duv_dxy = np.array([[0.263, 0.],
                         [0., 0.263]]),
    pad_factor = 3,
    sigma = 0.65,
    minsep=5
):
    """
    Perform image detection by multi-recentering and return the detected
    positions.

    Parameters
    ----------
    image : numpy.ndarray
        2D array of shape (Ny, Nx), the input image to detect sources from.
    noise_rms : float, optional
        The root-mean-square noise level in the image (default=1).
    sn : float, optional
        The signal-to-noise threshold for detection (default=1).
    psf : numpy.ndarray, optional
        2D array of shape (Ny_psf, Nx_psf), the point-spread function to use
        for detection. If not provided, Tpsf must be provided instead
        (default=None).
    Tpsf : float, optional
        The turbulence strength parameter for generating a Kolmogorov-type
        PSF using the provided duv_dxy and tile_size parameters. If psf is
        provided, this parameter is ignored (default=None).
    duv_dxy : numpy.ndarray, optional
        2D array of shape (2, 2), the spatial frequency resolution of the PSF.
        The units are cycles per pixel, and the default value corresponds to
        a pixel scale of 0.263 arcsec/pixel.
    pad_factor : float, optional
        The factor by which to pad the image before performing the Fourier
        transform (default=3).
    sigma : float, optional
        The standard deviation of the Gaussian kernel used to smooth the image
        before multi-recentering (default=0.65).

    Returns
    -------
    dx : numpy.ndarray
        2D array of shape (Nsrc, 2), the detected positions of the sources in
        the image, in pixel coordinates.
    dx_init : numpy.ndarray
        2D array of shape (Nsrc, 2), the initial estimated positions of the
        sources in the image, in pixel coordinates.
    mf_map : numpy.ndarray
        2D array of shape (Ny, Nx), the matched-filtered image used for
        detection.
    """

    
    d = Detection(
        duv_dxy = duv_dxy,
        tile_size = np.array(image.shape),
        pad_factor = pad_factor,
        sigma = sigma
    )
    
    d._set_self()
    
    if type(psf)==type(None):
        print('No PSF supplied...')
        return
    
    else:
        dx,dx_init,kval,ku,kv,d2k,conjugate,kvar,mf_map,mf_cov = d.multi_recenter(image,psf,noise_rms=noise_rms,sn=sn,minsep=minsep)
        
        return dx,dx_init,kval,ku,kv,d2k,conjugate,kvar,mf_map,mf_cov
    
    
def centerImage(img,psf,noise,duv_dxy,sn,band,minsep=5):
    dx,dx_init,kval,ku,kv,d2k,conjugate,kvar,mf_map,mf_cov = Detectinator(img,
                                                                          psf=psf,
                                                                          noise_rms=noise,
                                                                          sn=sn,
                                                                          duv_dxy=duv_dxy,
                                                                          minsep=minsep)
    
    kds = []
    
    if len(dx)>1:
        print('Beware! Here be blends!')
        
        kval[1::2,::2] *= -1. 
        kval[::2,1::2] *= -1.
        
        for dxi in dx:
            
            origin = np.dot(duv_dxy,(dxi+np.array(img.shape)).T)
            print(origin)
            phase = ku * origin[0] + kv * origin[1]
            kval_ = np.exp(1j*phase) * kval

            kd = bfd.KData(kval_,ku,kv,d2k,conjugate,kvar,band)
            
            kds.append(kd)
    
    else:
        kval[1::2,::2] *= -1. 
        kval[::2,1::2] *= -1.
        origin = np.dot(duv_dxy,(dx+np.array(img.shape)).T)
        phase = ku * origin[0] + kv * origin[1]
        kval = np.exp(1j*phase) * kval

        kds = [bfd.KData(kval,ku,kv,d2k,conjugate,kvar,band)]
        
    return kds

def multicenterImage(imgs,psfs,noises,duv_dxys,sn,band,minsep=5):
    
    kds = []
    
    for img,psf,noise,duv_dxy in zip(imgs,psfs,noises,duv_dxys): 
        kds += centerImage(img,psf,noise,duv_dxy,sn=sn,band=band,minsep=minsep)
        
    return kds
