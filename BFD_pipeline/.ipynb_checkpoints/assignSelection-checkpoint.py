#!/usr/bin/env python
# Function and executable that will assign noise tiers
# and interpolate selection PQR's to target files.

import sys
import numpy as np
from astropy.table import Table
import bfd
import argparse


if __name__=='__main__':
    # Collect arguments for function from command line

    parser = argparse.ArgumentParser(description='''Assign noise tiers and selection PQR's to target file(s)''')
    parser.add_argument('tierFile', help='noisetiers file', type=str)
    parser.add_argument('targetFile', help='path to target catalog(s)', type=str, nargs='+')
    args = parser.parse_args()
    
    print (args.tierFile)
    tc = bfd.TierCollection.load(args.tierFile)
    for target in args.targetFile:
        print('# Working on',target)
        tab = Table.read(target)
        
        tc.assignPQRSel(tab)
        # Overwrite new version of table
        print("columns on close:\n",tab.colnames)
        tab.write(target, overwrite=True)
    sys.exit()
