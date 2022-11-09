import numpy as np

import pyfits as pf
import os
import frogress
import copy
import math
import astropy.io.fits as fits
import pyfits as pf
import frogress
import multiprocessing
from functools import partial
from bfd.keywords import *
def make_pqr(data):
    mask = data[:,0]!=0
    logP = -np.log(data[mask,0])
    Q1 = data[mask,1]/data[mask,0]
    Q2 = data[mask,2]/data[mask,0]
    Q = np.vstack([Q1,Q2]).T
    R11 = Q1**2 -data[mask,3]/data[mask,0]
    R22 = Q2**2 -data[mask,5]/data[mask,0]
    R12 = Q1*Q2 -data[mask,4]/data[mask,0]
    R = np.array([np.diag([R11[i],R22[i]]) for i in range(len(R12))])

    return logP,Q,R,mask

def f_(uu,output_folder,noise_tiers):
    os.system('cp {0}/noisetiers.fits {0}/noisetiers_{1}.fits'.format(output_folder,noise_tiers[uu]))
    str_ = '/global/project/projectdirs/des/BFD_Y6/bfd/bin/tierSelection {0}/noisetiers_{1}.fits {0}/templates_NOISETIER_{1}.fits'.format(output_folder,noise_tiers[uu])
    os.system(str_)         

def compute_g(p,q,r,mask = None):
    invcov = np.linalg.inv(np.sum(r,axis=0))
    
    g = np.matmul(invcov,np.sum(q,axis=0))
    g_err = np.sqrt(invcov.diagonal())
    return g,g_err


def find_close_neighbours(target_list,template_list):
    list_ = []
    for tu in template_list:
        try:
            distance = (np.sum((np.array([float(i) for i in tu.split('_')])-np.array([float(i) for i in target_list.split('_')]))**2))
            if distance <=1:
                list_.append(tu)
        except:
            pass
    return list_

def covariance_scalar_jck(TOTAL_PHI,jk_r, type_c = 'jackknife'):
    #  Covariance estimation
    if type_c == 'jackknife':
      fact=(jk_r-1.)/(jk_r)
    elif type_c=='bootstrap':
      fact=1./(jk_r)
    average=0.
    cov_jck=0.
    err_jck=0.
    for kk in range(jk_r):
        average+=TOTAL_PHI[kk]
    average=average/(jk_r)
    for kk in range(jk_r):
    #cov_jck+=TOTAL_PHI[kk]#*TOTAL_PHI[kk]
        cov_jck+=(-average+TOTAL_PHI[kk])*(-average+TOTAL_PHI[kk])
    err_jck=np.sqrt(cov_jck*fact)
    #average=average*(jk_r)/(jk_r-1)
    return {'cov' : cov_jck*fact,
          'err' : err_jck,
          'mean': average}

def define_jck_mask(length,n_jck):
    import math
    import random
    l_jck = math.ceil(length/n_jck)
    masks =[]
    for i in frogress.bar(range(n_jck)):
        mm = np.array([True]*length)
        mm[i*l_jck:(i+1)*l_jck] = False
        masks.append(mm)
    return masks




def cpp_part(output_folder,**config):
    config['output_folder'] = output_folder

    if config['MPI']:
        from mpi4py import MPI 
    try: 
        add_labels = config['add_labels']
    except:
        add_labels = ['','ISp_','ISm_']
    for stage in config['stage']:
        todos = [stage]
        for add_i,add in enumerate(add_labels):
            #try:
                # now,must divide by chn
            path_templates = config['output_folder']
            target = path_templates+'/{0}targets_sample_g.fits'.format(add)
            if os.path.exists(target):
                targets = pf.open(target)
                #print (target)
                try:
                    noise_tiers = np.array(config['noise_tiers'])
                except:
                    noise_tiers = np.unique(targets[1].data['NOISETIER'])
                    noise_tiers = noise_tiers[noise_tiers!=-1]
                print (noise_tiers)
                
                


                
                
                
                if 'selection' in todos:
                    chunks = len(noise_tiers)
                    pr = 3
                    subchunks  = math.ceil(chunks/pr)
                
                    run_count = 0
                    
                    while run_count<subchunks: #chunks:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<subchunks:#len(noise_tiers):
                           
                            k_ = run_count+comm.rank
                            idx_ = np.arange(pr*k_,min(chunks,pr*(k_+1)))
                            pool = multiprocessing.Pool(processes=pr)
                                
                            _ = pool.map(partial(f_,output_folder=output_folder,noise_tiers=noise_tiers),idx_)
                            
                            
                                #os.system('cp {0}/noisetiers.fits {0}/noisetiers_{1}.fits'.format(output_folder,noise_tiers[run_count+comm.rank]))
                                #str_ = '/global/project/projectdirs/des/BFD_Y6/bfd/bin/tierSelection {0}/noisetiers_{1}.fits {0}/templates_NOISETIER_{1}.fits'.format(output_folder,noise_tiers[run_count+comm.rank])
                                #os.system(str_)
                                
                        run_count+=comm.size
                        comm.bcast(run_count,root = 0)
                        comm.Barrier() 
                        
                        
                        
                        

                if 'selection_ave' in todos:
        
                        
                    run_count = 0


                    comm = MPI.COMM_WORLD
                    if comm.rank==0:
                        m = pf.open('{0}/noisetiers.fits'.format(output_folder))
                        noise_tiers = np.arange(0,len(noise_tiers))
                        for i_ in range(len(noise_tiers)):
                            mx = pf.open('{0}/noisetiers_{1}.fits'.format(output_folder,noise_tiers[i_]))
                            try:
                                m[i_+1] = mx[i_+1]
                            except:
                                m.append(mx[i_+1])


                                
                                

                        prihdu = fits.PrimaryHDU()
                        u = pf.HDUList([pf.PrimaryHDU()])
                        for i in range(len(m)-1):
                            u_ = pf.BinTableHDU.from_columns(m[i+1].columns)
                            keys_ = ['TIER_NUM','EXTNAME','COVMXMX','COVMXMY','COVMYMY','SIG_XY','SIG_FLUX','STARTA00','STEPA00','INDEXA00','STARTA01','STEPA01','INDEXA01','MONO_1_0','MONO_1_1','MONO_2_0','MONO_2_1','MONO_3_0','MONO_3_1','ELLIP_1','ELLIP_2','FLUX_MIN','FLUX_MAX','WT_N','WT_SIGMA']

                            for k in keys_:
                                try:
                                    u_.header[k] = mx[i+1].header[k]
                                except:
                                    pass
        
                            u.append(u_)
                        u.writeto(output_folder+'/noisetiers.fits',clobber = True)# 
                            
                    comm.bcast(run_count,root = 0)
                    comm.Barrier() 
                
                    run_count = 0

                    import time
                    time.sleep(5)


                    comm = MPI.COMM_WORLD
                    if comm.rank==0:
                        print ('re-assembly everything')
                                    
                        targetfile = output_folder+'/{0}targets_sample_g.fits'.format(add,run_count+comm.rank)
                        noistierfile = output_folder+'noisetiers.fits'
                        str_ = 'python ./assignSelection.py {0} {1}'.format(noistierfile,targetfile)
                        os.system(str_)
                    comm.bcast(run_count,root = 0)
                    comm.Barrier() 
                
                
                
                
                if 'split' in todos:
                    
                    '''
                    we do have to split targets & template by classes
                    
                    '''
                    # read all the templates classes from templates overview -----
                    templates_overview = pf.open(path_templates+'/templates_overview.fits')
                    class_templates = templates_overview[1].data['class']
                    types_templates = np.unique(class_templates[class_templates!='-100_-100_-100'])
                   
                    # read all the targets classes from the targets files --------
                    class_targets = targets[1].data['class']
                    types_targets = np.unique(class_targets[class_targets!='-100_-100_-100'])

                    # produce the matching scheme --------------------------------
                    list_templates = []
                    for target_ in types_targets:
                        list_templates.append(find_close_neighbours(target_,types_templates))
                    
                    
                    print ('split targets *************')
                    # we're splitting the target file into chunks and then assemble it back ***
                    len_targets = len(targets[1].data['ID'])
                    chunks = len(types_targets)
                    targets_i = copy.copy(targets)
  
                    run_count = 0
                    while run_count<chunks:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<chunks:

                            
                        
                            mask_targets = class_targets == types_targets[run_count+comm.rank]
                            n_batches = math.ceil(len(mask_targets[mask_targets])/config['bacth_size'])
                            
                            for n in range(n_batches):
                                c = targets[1].columns
                                cols = [] 
                                for k in c:
                                    try:
                                        cols.append(pf.Column(name=k.name,format=k.format,array=(targets[1].data[k.name][mask_targets,:])[n*config['bacth_size']:(n+1)*config['bacth_size'],:]))
                                    except:
                                        cols.append(pf.Column(name=k.name,format=k.format,array=(targets[1].data[k.name][mask_targets])[n*config['bacth_size']:(n+1)*config['bacth_size']]))


                                new_cols = pf.ColDefs(cols)
                                hdu = pf.BinTableHDU.from_columns(new_cols)
                                targets_i[1] = hdu
                                targets_i[1].header[hdrkeys['weightN']] = targets[1].header[hdrkeys['weightN']]
                                targets_i[1].header[hdrkeys['weightSigma']] = targets[1].header[hdrkeys['weightSigma']]


                                # Next create an image of mean covariance matrix for each tier

                                for tier in range(2,len(targets)):
                                    data = copy.copy(targets[tier].data)
                                    hdu = pf.ImageHDU(data)
                                    hdu.header[hdrkeys['weightN']] =   targets[0].header[hdrkeys['weightN']]
                                    hdu.header[hdrkeys['weightSigma']] = targets[0].header[hdrkeys['weightSigma']]
                                    hdu.header['TIERNAME'] = targets[tier].header['TIERNAME']
                                    hdu.header['TIERLOST'] = 0
                                    # Record mean covariance of odd moments in header

                                    hdu.header['COVMXMX'] = copy.copy(targets[tier].header['COVMXMX'])
                                    hdu.header['COVMXMY'] = copy.copy(targets[tier].header['COVMXMY'])
                                    hdu.header['COVMYMY'] = copy.copy(targets[tier].header['COVMYMY'])

                                    targets_i[tier] = hdu


                                # save the new target files ****
                                try:
                                    targets_i.writeto(output_folder+'/{0}targets_sample_g_{1}_{2}.fits'.format(add,run_count+comm.rank,n))
                                except:
                                    try:
                                        targets_i.writeto(output_folder+'/{0}targets_sample_g_{1}_{2}.fits'.format(add,run_count+comm.rank,n),clobber = True)# 
                                    except:
                                        pass

                        
                        run_count+=comm.size
                        comm.bcast(run_count,root = 0)
                        comm.Barrier() 
                        
                    # need to split all the templates accordingly now.....
                    run_count = 0
                    while run_count<chunks:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<chunks:

                            # loop through all the templates files
                            for u in range(len(noise_tiers)):
                                
                                templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[u])
                                templates_ = pf.open(templatesfile)
                                templates_i = copy.copy(templates_)
                                mask = [c in list_templates[run_count+comm.rank]  for c in templates_[1].data['class']]
                                
                                
                                c = templates_[1].columns
                                cols = [] 
                                for k in c:
                                    try:
                                        cols.append(pf.Column(name=k.name,format=k.format,array=templates_[1].data[k.name][mask,:]))
                                    except:
                                        cols.append(pf.Column(name=k.name,format=k.format,array=templates_[1].data[k.name][mask]))


                                new_cols = pf.ColDefs(cols)
                                hdu = pf.BinTableHDU.from_columns(new_cols)



                                templates_i[1] = hdu
                                templates_i[1].header[hdrkeys['weightN']] = templates_[1].header[hdrkeys['weightN']]
                                templates_i[1].header[hdrkeys['weightSigma']] = templates_[1].header[hdrkeys['weightSigma']]

                                templates_i[1].header['FLUX_MIN'] = templates_[1].header['FLUX_MIN']
                                templates_i[1].header['SIG_XY'] = templates_[1].header['SIG_XY']
                                templates_i[1].header['SIG_FLUX'] = templates_[1].header['SIG_FLUX']
                                templates_i[1].header['SIG_STEP'] = templates_[1].header['SIG_STEP']
                                templates_i[1].header['SIG_MAX'] = templates_[1].header['SIG_MAX']
                                templates_i[1].header['EFFAREA'] = templates_[1].header['EFFAREA']
                                templates_i[1].header['TIER_NUM'] = templates_[1].header['TIER_NUM']

                                # save the new target files ****
                                try:
                                    templates_i.writeto(output_folder+'/templates_NOISETIER_{0}_{1}_{2}.fits'.format(noise_tiers[u],add,run_count+comm.rank))
                                except:
                                    try:
                                        templates_i.writeto(output_folder+'/templates_NOISETIER_{0}_{1}_{2}.fits'.format(noise_tiers[u],add,run_count+comm.rank),clobber = True)# 
                                    except:
                                        pass

                        
                        run_count+=comm.size
                        comm.bcast(run_count,root = 0)
                        comm.Barrier() 

                if 'integrate' in todos:
                    print ('integrate *************')
                
                

                    run_count = 0
                    runs_to_do = dict()

                    
                    
                   # read all the targets classes from the targets files --------
                    class_targets = targets[1].data['class']
                    types_targets = np.unique(class_targets[class_targets!='-100_-100_-100'])
                    chunks = len(types_targets)
                    

                    
                    #for  run_count in range(chunks):
                    #    targetfile = output_folder+'/{0}targets_sample_g_{1}.fits'.format(add,run_count)
                    #    targetfile_ = pf.open(targetfile)
                    #    t_ = []
                    #    try:
                    #        for t in  np.array(targetfile_[1].header['HISTORY']):
                    #            try:
                    #                t_.append(np.int((t.split('tier')[1]).split('from')[0]) )
                    #            except:
                    #                pass
                    #    except:
                    #        pass
                    #    runs_to_do[run_count] = noise_tiers[~np.in1d(noise_tiers,np.array(t_))]


                    
                    # run them *******
                    run_count = 0
                    while run_count<chunks:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<chunks:
                            
                            #check the number of batches --------------
                            mask_targets = class_targets == types_targets[run_count+comm.rank]
                            n_batches = math.ceil(len(mask_targets[mask_targets])/config['bacth_size'])

                            # loop over batches and noise_tiers -------
                            for n in range(n_batches):
                                
                                # check what has been run so far ------------
                                targetfile_ = pf.open(output_folder+'/{0}targets_sample_g_{1}_{2}.fits'.format(add,run_count+comm.rank,n))
                                t_ = []
                                try:
                                    for t in  np.array(targetfile_[1].header['HISTORY']):
                                        try:
                                            t_.append(np.int((t.split('tier')[1]).split('from')[0]) )
                                        except:
                                            pass
                                except:
                                    pass
                                runs_to_do = noise_tiers[~np.in1d(noise_tiers,np.array(t_))]
                        
                                for u in range(len(noise_tiers)):


                                    if noise_tiers[u] in runs_to_do:
                                        targetfile = output_folder+'/{0}targets_sample_g_{1}_{2}.fits'.format(add,run_count+comm.rank,n)
                                        noistierfile = output_folder+'noisetiers.fits'
                                        templatesfile = output_folder+'/templates_NOISETIER_{0}_{1}_{2}.fits'.format(noise_tiers[u],add,run_count+comm.rank)

                                        str_ = '/global/project/projectdirs/des/BFD_Y6/bfd/bin/tableIntegrate -targetfile={0} -noisetierFile={1} -templateFile={2}'.format(targetfile,noistierfile,templatesfile)


                                        os.system(str_)
                                    #print('')
                                    #print (str_)
                                

                        run_count+=comm.size
                        comm.bcast(run_count,root = 0)
                        comm.Barrier() 



        for add in add_labels:
            path_templates = config['output_folder']
            target = path_templates+'/{0}targets_sample_g.fits'.format(add)
            if os.path.exists(target):
                run_count = 0
                comm = MPI.COMM_WORLD
                if 'assemble' in todos:

                    if comm.rank==0:
                        
                            target = path_templates+'/{0}targets_sample_g.fits'.format(add)
                            targets = pf.open(target)
                            len_targets = len(targets[1].data['ID'])
                            pqr = np.zeros((len_targets,6))
                            NUNIQUE = np.zeros(len_targets)
                            count =0
                            
                            
                            
                            
                            
                            
                         # read all the targets classes from the targets files --------
                            class_targets = targets[1].data['class']
                            types_targets = np.unique(class_targets[class_targets!='-100_-100_-100'])
                            chunks = len(types_targets)
                    

                    
                            for run_count in range(chunks):
                    
                                
                                #check the number of batches --------------
                                mask_targets = class_targets == types_targets[run_count]
                                n_batches = math.ceil(len(mask_targets[mask_targets])/config['bacth_size'])

                                # loop over batches and noise_tiers -------
                                for n in range(n_batches):

                                    
                                    
                                    targetfile = output_folder+'/{0}targets_sample_g_{1}_{2}.fits'.format(add,run_count+comm.rank,n)
                                    targets_i = pf.open(targetfile)
                                    print (targetfile)
                                        

                                    # id matching
                                    fix = np.arange(len(mask_targets))
                                    fix = fix[mask_targets]
                                    
                                    fix = fix[n*config['bacth_size']:(n+1)*config['bacth_size']]
                                    
                                    mask = np.in1d(np.arange(len(mask_targets)),fix)
                           
                                    NUNIQUE[mask] = targets_i[1].data['NUNIQUE']
                                    pqr[mask] = targets_i[1].data['PQR']
                                   
                                
                                     
                            c = targets[1].columns
                            cols = [] 
                            if 1==1:
                            #try:
                                c['PQR'].array=pqr
                            #except:
                            #    pass
                            #try:
                            #    c['NUNIQUE'].array=NUNIQUE
                            #except:
                            #    pass
                            for k in c:
                                if k.name == 'PQR':
                                    cols.append(pf.Column(name=k.name,format=k.format,array=pqr))
                                elif k.name == 'NUNIQUE':
                                    cols.append(pf.Column(name=k.name,format=k.format,array=NUNIQUE))
                                else:
                                    cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name]))

                            if sum([cc.name=='PQR' for cc in cols]) ==0:
                                cols.append(pf.Column(name='PQR',format='6E',array=pqr))

                            if sum([cc.name=='NUNIQUE' for cc in cols]) ==0:
                                cols.append(pf.Column(name='NUNIQUE',format='1J',array=NUNIQUE))



                            new_cols = pf.ColDefs(cols)
                            hdu = pf.BinTableHDU.from_columns(new_cols)
                            targets_i[1] = hdu
                            targets_i[1].header[hdrkeys['weightN']] = targets[1].header[hdrkeys['weightN']]
                            targets_i[1].header[hdrkeys['weightSigma']] = targets[1].header[hdrkeys['weightSigma']]

                            for tier in range(2,len(targets)):
                                data = copy.copy(targets[tier].data)
                                hdu = pf.ImageHDU(data)
                                
                                
                                hdu.header[hdrkeys['weightN']] =   targets[0].header[hdrkeys['weightN']]
                                hdu.header[hdrkeys['weightSigma']] = targets[0].header[hdrkeys['weightSigma']]
                                hdu.header['TIERNAME'] = targets[tier].header['TIERNAME']
                                hdu.header['TIERLOST'] = 0
                                # Record mean covariance of odd moments in header
                                hdu.header['COVMXMX'] = copy.copy(targets[tier].header['COVMXMX'])
                                hdu.header['COVMXMY'] = copy.copy(targets[tier].header['COVMXMY'])
                                hdu.header['COVMYMY'] = copy.copy(targets[tier].header['COVMYMY'])

                                targets_i[tier] = hdu


                            # save the new target files ****
                            #if 1==1:
                            try:
                                targets_i.writeto(output_folder+'/{0}targets_sample_g.fits'.format(add))
                            except:
                                try:
                                    targets_i.writeto(output_folder+'/{0}targets_sample_g.fits'.format(add),clobber = True)# 
                                except:
                                    pass
#
                comm.bcast(run_count,root = 0)
                comm.Barrier() 



        # measure mean shear *************************************

        if 'compute_m' in todos:
            comm = MPI.COMM_WORLD
            if comm.rank==0:
                if config['image_sims']:

                    add_labels = ['ISp_','ISm_']
                    results = dict()
                    for add in add_labels:
                        target = path_templates+'/{0}targets_sample_g.fits'.format(add)
                        targets = pf.open(target)
                        p, q, r, mask= make_pqr(targets[1].data['PQR'])
                        
                        results[add] = dict()
                        results[add]['p'] = np.array(p).astype(np.float64)
                        results[add]['q'] = np.array(q).astype(np.float64)
                        results[add]['r'] = np.array(r).astype(np.float64)
                        results[add]['mask'] = np.array(mask)


                    #results['ISp_'] = np.array( results['ISp_'])
                    #results['ISm_'] = np.array( results['ISm_'])
                    for i in ['p','q','r']:
                        results['ISp_'][i]  = np.array(results['ISp_'][i])[np.array(results['ISm_']['mask'])[np.array(results['ISp_']['mask'])]]
                        results['ISm_'][i]  = np.array(results['ISm_'][i])[np.array(results['ISp_']['mask'])[np.array(results['ISm_']['mask'])]]

                    p,q,r = results['ISp_']['p'],results['ISp_']['q'],results['ISp_']['r']
                    m_ = define_jck_mask(len(p),100)
                    import frogress
                    gp_ = []
                    p,q,r = results['ISp_']['p'],results['ISp_']['q'],results['ISp_']['r']
                    for m__ in frogress.bar(m_):
                        g,_ = compute_g(p[m__],q[m__,:],r[m__,:,:])
                        gp_.append(g)

                    p,q,r= results['ISp_']['p'],results['ISp_']['q'],results['ISp_']['r']
                    gp,g_err = compute_g(p,q,r)
                    print ('')
                    print ('<g1>: {0:2.5f} +- {1:2.5f},   <g2>: {2:2.5f} +- {3:2.5f}'.format(gp[0],g_err[0],gp[1],g_err[1]))

                    print ('m_p = {0:2.5f} +- {1:2.5f}'.format(gp[0]/0.02-1.,g_err[0]/0.02))

                    gm_ = []
                    p,q,r = results['ISm_']['p'],results['ISm_']['q'],results['ISm_']['r']
                    for m__ in frogress.bar(m_):
                        g,_ = compute_g(p[m__],q[m__,:],r[m__,:,:])
                        gm_.append(g)



                    d = covariance_scalar_jck((np.array(gp_)-np.array(gm_))/(2*0.02)-1.,100)



                    p,q,r=  results['ISm_']['p'],results['ISm_']['q'],results['ISm_']['r']
                    gm,g_err = compute_g(p,q,r)
                    print ('')
                    print ('<g1>: {0:2.5f} +- {1:2.5f},   <g2>: {2:2.5f} +- {3:2.5f}'.format(gm[0],g_err[0],gm[1],g_err[1]))

                    print ('m_m = {0:2.5f} +- {1:2.5f}'.format(gm[0]/0.02+1.,g_err[0]/0.02))


                    print ('m_diff = {0:2.5f} +- {1:2.5f}'.format((gp[0]-gm[0])/(2*0.02)-1.,d['err'][0]))