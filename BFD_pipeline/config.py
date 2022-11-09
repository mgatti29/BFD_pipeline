# TODO: License

"""
.. module:: config
"""
import yaml
import numpy as np
import os
import sys
from .measure_moments_targets_ext import measure_moments_targets_ext
from .measure_moments_targets import measure_moments_targets
from .measure_moments_templates import measure_moments_templates
from .make_targets import make_targets
from .make_templates import make_templates
from .cpp_part_split import cpp_part
from .make_tiles_tt import make_tiles_tt
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
    if not os.path.exists(config['general']['output_folder']):
        try:
            os.mkdir(config['general']['output_folder'])
        except:
            pass
    try:
        os.system('cp {0} {1}'.format(config['config_path'],config['general']['output_folder']))
    except:
        pass
    # add some options from general:
    
    keys_add = ['n','sigma','band_dict','bands','MPI','pad_factor','classes']

    for key in keys_add:
        #try:
            try:
                config['measure_moments_targets_ext'][key] = copy.copy(config['general'][key])
            except:
                pass
            try:
                config['measure_moments_targets'][key] = copy.copy(config['general'][key])
            except:
                pass

            try:
                config['measure_moments_templates'][key] = copy.copy(config['general'][key])
            except:pass
            try:       
                config['make_targets'][key] = copy.copy(config['general'][key])
            except:pass
            try:       
                config['make_templates'][key] = copy.copy(config['general'][key])
            except:pass
            try:       
                config['cpp_part'][key] = copy.copy(config['general'][key])
            except:pass
            try:       
                config['make_tiles_tt'][key] = copy.copy(config['general'][key])
            except:pass
            #try:
            
            #except:
             #   pass
        #except:
        #    print (key+' not in the config file')
        
    # modules ************************************************
    if config['run']!= None:
        for entry in config['run']:
            if entry == 'measure_moments_targets_ext':
                measure_moments_targets_ext(config['general']['output_folder'],**config['measure_moments_targets_ext'])
            if entry == 'measure_moments_targets':
                measure_moments_targets(config['general']['output_folder'],**config['measure_moments_targets'])
            if entry == 'measure_moments_templates':
                measure_moments_templates(config['general']['output_folder'],**config['measure_moments_templates'])  
            if entry == 'make_tiles_tt':
                make_tiles_tt(config['general']['output_folder'],**config['make_tiles_tt'])
            if entry == 'make_targets':
                make_targets(config['general']['output_folder'],**config['make_targets'])
            if entry == 'make_templates':
                make_templates(config['general']['output_folder'],**config['make_templates'])
            if entry == 'setup_image_sims':
                setup_image_sims(config['general']['output_folder'],**config['setup_image_sims'])
            if entry == 'cpp_part':
                cpp_part(config['general']['output_folder'],**config['cpp_part'])

            
def read_config(file_name):
    """Read a configuration dictionary from a file

    :param file_name:   yaml file name which we read
    """
    import yaml
    with open(file_name) as f_in:
        config = yaml.load(f_in.read())
    config['config_path'] = file_name
    return config
