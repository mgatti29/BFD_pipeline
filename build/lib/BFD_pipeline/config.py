# TODO: License

"""
.. module:: config
"""
import yaml
import numpy as np
import os
import sys
from .measure_moments_targets import measure_moments_targets
from .measure_moments_templates import measure_moments_templates
from .make_targets import make_targets
from .make_templates import make_templates
import copy

#from .dataset import dataset
#
#from .dndz import dndz
#from .compare_multiplefiles_joint import compare

def BFD_pipeline(config):
    """Run the the BFD measurement pipeline, given a configuration dictionary

    Parameters
    ----------
    config : the dictionary of config files

    """
    
    # general options ************************************************
    try:
        if not os.path.exists(config['general']['output_folder']):
            os.mkdir(config['general']['output_folder'])
    except:
        print ('Output folder missing - exit.')
        sys.exit()
        
    # add some options from general:
    
    keys_add = ['n','sigma','band_dict','bands']

    for key in keys_add:
        config['measure_moments_targets'][key] = copy.copy(config['general'][key])
        config['measure_moments_templates'][key] = copy.copy(config['general'][key])
        config['make_targets'][key] = copy.copy(config['general'][key])
        config['make_templates'][key] = copy.copy(config['general'][key])
                
        
    # modules ************************************************
    if config['run']!= None:
        for entry in config['run']:
            if entry == 'measure_moments_targets':
                measure_moments_targets(config['general']['output_folder'],**config['measure_moments_targets'])
            if entry == 'measure_moments_templates':
                measure_moments_templates(config['general']['output_folder'],**config['measure_moments_templates'])  
            if entry == 'make_targets':
                make_targets(config['general']['output_folder'],**config['make_targets'])
            if entry == 'make_templates':
                make_templates(config['general']['output_folder'],**config['make_templates'])
            
def read_config(file_name):
    """Read a configuration dictionary from a file

    :param file_name:   yaml file name which we read
    """
    import yaml
    with open(file_name) as f_in:
        config = yaml.load(f_in.read())
    return config