# TODO: License

"""
.. module:: config
"""
import yaml
import numpy as np
import os
import sys
from .measure_moments_targets import measure_moments_targets


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
        
    # modules ************************************************
    if config['run']!= None:
        for entry in config['run']:
            if entry == 'measure_target_moments':
                measure_moments_targets(config['general']['output_folder'],**config['measure_target_moments'])
            
            
def read_config(file_name):
    """Read a configuration dictionary from a file

    :param file_name:   yaml file name which we read
    """
    import yaml
    with open(file_name) as f_in:
        config = yaml.load(f_in.read())
    return config