import time
import timeit
import pickle
import numpy as np
import sys
def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')


            
            

def update_progress(progress,elapsed_time=0,starting_time=0):
    
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))



    if progress*100>1. and elapsed_time>0 :
        remaining=((elapsed_time-starting_time)/progress)*(1.-progress)
        text = "\rPercent: [{0}] {1:.2f}% {2}  - elapsed time: {3} - estimated remaining time: {4}".format( "#"*block + "-"*(barLength-block), progress*100, status,time.strftime('%H:%M:%S',time.gmtime(elapsed_time-starting_time)),time.strftime('%H:%M:%S',time.gmtime(remaining)))
    else:
        text = "\rPercent: [{0}] {1:.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
