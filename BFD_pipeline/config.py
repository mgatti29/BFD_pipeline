import yaml
from yaml import Loader
import os
from .measure_moments_targets import measure_moments_targets


def BFD_pipeline(config):
    """
    Run the the BFD measurement pipeline, given a configuration dictionary

    Parameters
    ----------
    config : the dictionary of config files
    """
    
    # We first check if a number of entries are in the config file; if not, we use Defaults values. 
    
    if 'output_folder'  not in config['general'].keys():
        config['general']['output_folder'] = './BFD_output'
        print ('"output_folder" not specified in the config file; using "./BFD_output" instead')
      
    if 'filter_sigma'  not in config['general'].keys():
        config['general']['filter_sigma'] = 0.65
        print ('"filter_sigma" (parameter of the BFD filter) not specified in the config file; using sigma = 0.65 as Default')
        
    if 'FFT_pad_factor'  not in config['general'].keys():
        config['general']['FFT_pad_factor'] = 2
        print ('"FFT_pad_factor" (pad_factor for the FFT of the images) not specified in the config file; using FFT_pad_factor = 2 as Default')
              
    if 'bands_meds_files'  not in config['general'].keys():
        config['general']['bands_meds_files'] = ['i']
        print ('"bands_meds_files" not specified in the config file; using ["i"]as Default')
           
            
    if 'bands_weights'  not in config['general'].keys():
        config['general']['bands_weights'] = 1.0
        print ('"bands_weights"  not specified in the config file; using 1.0 as Default')
              

    if 'MPI'  not in config['general'].keys():      
        config['general']['MPI'] =  False
        print ('"MPI"  not specified in the config file; using MPI = False as Default')
        
        
    # check if the output folder exist.
    if not os.path.exists(config['general']['output_folder']):
        try:
            os.mkdir(config['general']['output_folder'])
        except:
            pass
    if not os.path.exists(config['general']['output_folder']+'/targets/'):
        try:
            os.mkdir(config['general']['output_folder']+'/targets/')
        except:
            pass 
        
    if not os.path.exists(config['general']['output_folder']+'/MOF_models/'):
        try:
            os.mkdir(config['general']['output_folder']+'/MOF_models/')
        except:
            pass
        
    #Add the general keys to all the other submodules
    for key1 in ['measure_moments_targets']:
        if key1 != 'general':
            for key2 in config['general'].keys():
                config[key1][key2] = config['general'][key2]

                
    #Let's run the individual modules.
    if config['run']!= None:
        for entry in config['run']:
            if entry == 'measure_moments_targets':  
                measure_moments_targets(**config['measure_moments_targets'])
            

            
def read_config(file_name):
    """Read a configuration dictionary from a file

    :param file_name:   yaml file name which we read
    """

    with open(file_name) as f_in:
        config = yaml.load(f_in.read(),Loader=Loader)
    config['config_path'] = file_name
    return config