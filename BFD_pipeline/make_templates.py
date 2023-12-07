# Importing necessary libraries
import numpy as np
import astropy.io.fits as fits
import frogress
import os
import glob
import multiprocessing
from functools import partial 
from galsim.utilities import single_threaded
from bfd.momenttable import TemplateTable
from astropy.table import QTable, Table, vstack

def group_files(noise_tier,config):
    if not os.path.exists(config['output_folder']+'/templates_NOISETIER_{0}.fits'.format(noise_tier)):
        templates_table_noisetier = QTable()
        files = glob.glob(config['output_folder']+'/templates/templates_shifted_temp/NT_{0}_*'.format(noise_tier))
        for i in frogress.bar(range(len(files))):
            file = files[i]
            hdul = fits.open(file)
            data = QTable(hdul[1].data)
            templates_table_noisetier = vstack([templates_table_noisetier, data])  

        hdu1 = fits.BinTableHDU(templates_table_noisetier)
        hdulist1 = fits.HDUList([fits.PrimaryHDU(), hdu1])
        hdulist1.writeto(config['output_folder'] +'/templates_NOISETIER_{0}.fits'.format(noise_tier), overwrite=True)
    
    

# Function to make shifted templates for given runs and target properties
def make_shifted_templates(run, targets_properties, config):
    # Unpacking the run tuple
    file, noise_tier = run 
    
    # Constructing a label from the noise tier and file name
    label = 'NT_{0}_FILE_{1}'.format(noise_tier, file.split('template_moments_container')[1].split('.npy')[0])
    
    # Checking if the file already exists to avoid redundancy
    if not os.path.exists(config['output_folder']+'/templates/templates_shifted_temp/'+label+'.fits'):
        tab_templates = TemplateTable(n = 4,
                    sigma = config['filter_sigma'] ,
                    sn_min = config['sn_min'], 
                    sigma_xy = targets_properties[noise_tier]['sigma_xy'], 
                    sigma_flux = targets_properties[noise_tier]['sigma_flux'] , 
                    sigma_step = config['sigma_step'], 
                    sigma_max = config['sigma_max'],
                    xy_max = config['xy_max'],            
                    tier_number = noise_tier)
        
        # Loading a numpy file which contains the data
        m = np.load(file, allow_pickle=True).item()
                

        print('shifting copies - ', label)
        for i in frogress.bar(range(len(m.keys()))):
            ID = list(m.keys())[i]
            # Processing each item to create templates
            flag, result, xy_kept, area_integral = m[ID]['moments'].make_templates(
                targets_properties[noise_tier]['sigma_xy'],
                sigma_flux=targets_properties[noise_tier]['sigma_flux'],
                sn_min=config['sn_min'], sigma_max=config['sigma_max'],
                sigma_step=config['sigma_step'], xy_max=config['xy_max'])

            # Adding templates to the list if conditions are met
            if flag and (area_integral < 1.01):
                for template in result:
                    tab_templates.add(template)

        print ('saving --')
        tab_templates.save(config['output_folder']+'/templates/templates_shifted_temp/'+label+'.fits')



# Function to make templates based on given configuration
def make_templates(**config):
    # Checking and initializing MPI if needed
    if config['MPI']:
        from mpi4py import MPI 
    
    # Creating the necessary directory if it doesn't exist
    if not os.path.exists(config['output_folder']+'/templates/templates_shifted_temp/'):
        try:
            os.mkdir(config['output_folder']+'/templates/templates_shifted_temp/')
        except:
            pass
    
    # Opening a FITS file containing noise tiers
    NT_ = fits.open(config['output_folder']+'/noisetiers.fits')
    targets_properties = dict()
    for i, noise_tier in enumerate(range(len(NT_)-1)):
        targets_properties[noise_tier] = dict()
        # Extracting sigma values from the FITS file
        targets_properties[noise_tier]['sigma_xy'] = np.float('{0:2.4f}'.format(np.sqrt(NT_[i+1].header['COVMXMX'])))
        targets_properties[noise_tier]['sigma_flux'] = np.float('{0:2.4f}'.format((NT_[i+1].header['SIG_FLUX'])))

    # Gathering files matching a pattern using glob
    files = glob.glob(config['output_folder']+'/templates/template_moments_container*')
    summary = fits.open(config['output_folder']+'/templates/summary_templates.fits')

    # Defining different size thresholds
    # this is mostly done to run files of the same size at the same time
    sizes = [3000, 5000, 6500, 9500, 20000]
    print('loading files to check their sizes')
    runs_to_do = []
    for f in files:
        tile = f.split('container_')[1].split('.npy')[0].split('_r')[0]
        number_of_objects = len(summary[1].data['tile'][summary[1].data['tile'] == tile])      
        for noise_tier in targets_properties.keys():
            label = 'NT_{0}_FILE_{1}'.format(noise_tier, f.split('template_moments_container')[1].split('.npy')[0])
            path = config['output_folder']+'/templates/templates_shifted_temp/'+label+'.fits'
            if not os.path.exists(path):
                runs_to_do.append([f, noise_tier, number_of_objects, path])

    print('Done loading')
    for max_size in sizes:
        runs_to_do_short = []
        for run in runs_to_do:
            if (run[2] < max_size) and (not os.path.exists(run[3])):
                #if (run[2]>9500):
                    runs_to_do_short.append([run[0], run[1]])

        run_count = 0
        print('RUNS TO DO (SIZE<{0}) : {1}'.format(max_size, len(runs_to_do_short)))
        if not config['MPI']:
            # Processing runs without MPI
            while run_count < len(runs_to_do_short):
                make_shifted_templates(runs_to_do_short[run_count], targets_properties, config)
                run_count += 1
        else:
            # Processing runs with MPI for distributed computing
            while run_count < len(runs_to_do):
                comm = MPI.COMM_WORLD

                if (run_count + comm.rank) < len(runs_to_do_short):
                    make_shifted_templates(runs_to_do_short[run_count + comm.rank], targets_properties, config)

                run_count += comm.size
                comm.bcast(run_count, root=0)
                comm.Barrier()

                
                
    #'''
    run_count = 0
    noise_tiers = list(targets_properties.keys())
    if not config['MPI']:
        # Processing runs without MPI
        while run_count < len(noise_tiers):
            group_files(noise_tier,config)
            run_count += 1
    else:
        # Processing runs with MPI for distributed computing
        while run_count < len(noise_tiers):
            comm = MPI.COMM_WORLD

#            if comm.rank ==0:
            if (run_count + comm.rank) < len(noise_tiers):
                 group_files(noise_tiers[run_count + comm.rank],config)
            run_count += comm.size
            comm.bcast(run_count, root=0)
            comm.Barrier()
            
    #'''         
