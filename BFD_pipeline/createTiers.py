#!/usr/bin/env python
# Function and executable that will create noise tiers
# file from a list of target files that it should span

import sys
import numpy as np
import astropy.io.fits as fits
import bfd
import argparse
from bfd.keywords import *


def createTiers(targetFile=None,
                noiseStep = 0.2,
                psfStep = 0.1,
                fluxMin=None,
                fluxMax=None,
                snMin=None,
                snMax=None,
                sample=0.01,
                minTargets=10,
                **kwargs):
    '''Sample from all the input `targetTable` FITS target files
    to determine the space of covariance matrices in the survey.
    Then divide this into noise tiers, and return a NoiseTierCollection
    containing all the noise tiers and their information.
    
    Parameters:
    `targetFiles`: a list of input target tables' FITS filenames.
    `noiseStep`: spacing of tiers in log(flux variance)
    `psfStep`: spacing of tiers in space of PSF-size parameter
    `fluxMin, fluxMax`: global bounds (if any) to use on flux
    `snMin, snMax`: bounds on S/N to set in each tier, using its
          nominal covariance.  The more restrictive of flux and S/N
          bounds will be used.
    `sample`: fraction of targets to draw from each input table.
    `minTargets`: minimum number of targets that must be in a tier
          in order for it to be retained.
    '''
    # Acquire covariances from files
    covs = []
    wtN = None
    wtSigma = None
    for f in targetFile:
        print('# Acquiring target covariances from',f)
        hdu = fits.open(f)[1]
        if wtN is None:
            hdrkeys['weightSigma']
            wtN = hdu.header[hdrkeys['weightN']]
        elif hdu.header[hdrkeys['weightN']]!=wtN:
            raise ValueError('WT_N in file ' + f + ' does not match')
        if wtSigma is None:
            wtSigma = hdu.header[ hdrkeys['weightSigma']]
        elif hdu.header[ hdrkeys['weightSigma']]!=wtSigma:
            raise ValueError('WT_SIG in file ' + f + ' does not match')

        # Save a random subsample of the covariances
        keep = np.random.random(size=len(hdu.data)) < sample
        covs.append(hdu.data['covariance'][keep])
        
    if not covs:
        raise ValueError('No target covariances acquired')
    covs = np.concatenate(covs, axis=0)
    
    if wtN is None or wtSigma is None:
        raise ValueError('Input targets did not specify wtN and wtSigma')

    # Build tier set from the collected covariances
    print("# Building tiers from total of",covs.shape[0],"targets")
    print (snMin,snMax)
    tc = bfd.TierCollection(covs, wtN=wtN, wtSigma=wtSigma,
                            snMin=snMin, snMax=snMax,
                            fluxMin=fluxMin, fluxMax=fluxMax,
                            stepA = [noiseStep,psfStep],
                            minTargets=minTargets)
    return tc

if __name__=='__main__':
    # Collect arguments for function from command line

    parser = argparse.ArgumentParser(description='''Create noise tiers that span a set of targets''')
    parser.add_argument('targetFile', help='path to target catalog(s)', type=str, nargs='+')
    parser.add_argument('-o','--output', help='Output noisetiers file', type=str, default='noisetiers.fits')
    parser.add_argument('--noiseStep', help='Tier spacing in log flux variance', type=float, default=0.2)
    parser.add_argument('--psfStep', help='Tier spacing in log psf-size param', type=float, default=0.1)
    parser.add_argument('-s','--sample', help='Fraction of inputs to sample', type=float, default=0.01)
    parser.add_argument('--snMin', help='Minimum target S/N to use', type=float, default=7.)
    parser.add_argument('--snMax', help='Maximum target S/N to use', type=float)
    parser.add_argument('--fluxMin', help='Minimum target flux to use', type=float)
    parser.add_argument('--fluxMax', help='Maximum target flux to use', type=float)
    parser.add_argument('--minTargets', help='Minimum targets to retain a tier', type=int, default=10)
    args = parser.parse_args()
    
    

    tc = createTiers(**vars(args))
    tc.save(args.output)

    sys.exit()
