import numpy as np
from mpi4py import MPI     
import pyfits as pf

def cpp_part(output_folder,**config):

    add_labels = ['','ISp_','ISm_']
    for add in add_labels:
        try:
            # now,must divide by chn
            path_templates = config['output_folder']
            target = path_templates+'/{0}targets_sample_g.fits'.format(add)
            mu = pf.open(target)
            noise_tiers = np.unique(mu[1].data['NOISETIER'])
            run_count = 0

            comm = MPI.COMM_WORLD
            if comm.rank==0:
                while run_count<len(noise_tiers):

                    if (run_count+comm.rank) < len(noise_tiers):

                        a = open('config_{0}.config'.format(noise_tiers[run_count+comm.rank]),'w')
                        a.write('templateFile {0} \n'.format(path_templates+'templates_NOISETIER_{0}.fits'.format(noise_tiers[run_count+comm.rank])))
                        a.write('targetFile {0} \n'.format(target))
                        a.write('useTier {0}'.format(noise_tiers[run_count+comm.rank]))
                        a.close()
                        os.system('/global/project/projectdirs/des/BFD_Y6/bfd/bin/calculateSelection config_{0}.config   -useTier {0}'.format(noise_tiers[run_count+comm.rank]))
                        os.system('/global/project/projectdirs/des/BFD_Y6/bfd/bin/tableIntegrate config_{0}.config   -useTier {0}'.format(noise_tiers[run_count+comm.rank]))
                        os.system('/global/project/projectdirs/des/BFD_Y6/bfd/bin/meanShear -targetfile  {0} ' .format(target))


                        run_count+=1
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
        except:
            pass