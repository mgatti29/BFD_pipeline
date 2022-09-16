import numpy as np

import pyfits as pf
import os
import frogress
import copy
import math
import astropy.io.fits as fits
import pyfits as pf
import frogress
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

            

def compute_g(p,q,r,mask = None):
    invcov = np.linalg.inv(np.sum(r,axis=0))
    
    g = np.matmul(invcov,np.sum(q,axis=0))
    g_err = np.sqrt(invcov.diagonal())
    return g,g_err



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
    chunks = config['chunks']
    if config['MPI']:
        from mpi4py import MPI 
    try: 
        add_labels = config['add_labels']
    except:
        add_labels = ['','ISp_','ISm_']
    for stage in config['stage']:
        todos = [stage]
        for add in add_labels:
            #try:
                # now,must divide by chn
            path_templates = config['output_folder']
            target = path_templates+'/{0}targets_sample_g.fits'.format(add)
            if os.path.exists(target):
                targets = pf.open(target)
                #print (target)
                try:
                    noise_tiers = config['noise_tiers']
                except:
                    noise_tiers = np.unique(targets[1].data['NOISETIER'])
                    noise_tiers = noise_tiers[noise_tiers!=-1]
                print (noise_tiers)
                run_count = 0

                comm = MPI.COMM_WORLD
                if comm.rank==0:
                    if 'selection' in todos:
                        print ('SELECTION *************')
                        #os.system(str_)

                        while run_count<len(noise_tiers):

                            if (run_count+comm.rank) < len(noise_tiers):
                                 
                                
                                str_ = '/global/project/projectdirs/des/BFD_Y6/bfd/bin/tierSelection {0}/noisetiers.fits {0}/templates_NOISETIER_{1}.fits'.format(output_folder,noise_tiers[run_count+comm.rank])
                                os.system(str_)
                                #print()
                                print (str_)
                                run_count+=1
                        targetfile = output_folder+'/{0}targets_sample_g.fits'.format(add,run_count+comm.rank)
                        noistierfile = output_folder+'noisetiers.fits'
                        str_ = 'python ./assignSelection.py {0} {1}'.format(noistierfile,targetfile)
                        os.system(str_)
                comm.bcast(run_count,root = 0)
                comm.Barrier() 


                if 'split' in todos:
                    print ('split *************')
                    # we're splitting the target file into chunks and then assemble it back ***
                    len_targets = len(targets[1].data['ID'])
                    chunk_size = math.ceil(len_targets/chunks)
                    targets_i = copy.copy(targets)
                    #for i in range(chunks):
                    run_count = 0
                    while run_count<chunks:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<chunks:

                            start = (run_count+comm.rank)*chunk_size
                            end = min([len_targets,(run_count+comm.rank+1)*chunk_size])

                            c = targets[1].columns
                            cols = [] 
                            for k in c:
                                try:
                                    cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name][start:end,:]))
                                except:
                                    cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name][start:end]))


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
                                targets_i.writeto(output_folder+'/{0}targets_sample_g_{1}.fits'.format(add,run_count+comm.rank))
                            except:
                                try:
                                    targets_i.writeto(output_folder+'/{0}targets_sample_g_{1}.fits'.format(add,run_count+comm.rank),clobber = True)# 
                                except:
                                    pass
                        run_count+=comm.size
                        comm.bcast(run_count,root = 0)
                        comm.Barrier() 

                if 'integrate' in todos:
                    print ('integrate *************')
                
                    
                    # figure out how many runs to do:
                    #chunks = 80
                    run_count = 0
                    runs_to_do = dict()

                    for  run_count in range(chunks):
                        targetfile = output_folder+'/{0}targets_sample_g_{1}.fits'.format(add,run_count)
                        targetfile_ = pf.open(targetfile)
                        t_ = []
                        try:
                            for t in  np.array(targetfile_[1].header['HISTORY']):
                                try:
                                    t_.append(np.int((t.split('tier')[1]).split('from')[0]) )
                                except:
                                    pass
                        except:
                            pass
                        runs_to_do[run_count] = noise_tiers[~np.in1d(noise_tiers,np.array(t_))]


                    
                    # run them *******
                    run_count = 0
                    while run_count<chunks:
                        comm = MPI.COMM_WORLD
                        if run_count+comm.rank<chunks:

                            for u in range(len(noise_tiers)):
                                
                                if noise_tiers[u] in runs_to_do[run_count+comm.rank]:
                                    targetfile = output_folder+'/{0}targets_sample_g_{1}.fits'.format(add,run_count+comm.rank)
                                    noistierfile = output_folder+'noisetiers.fits'
                                    templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[u])

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
                comm = MPI.COMM_WORLD
                if 'assemble' in todos:

                    if comm.rank==0:
                        
                            target = path_templates+'/{0}targets_sample_g.fits'.format(add)
                            targets = pf.open(target)
                            len_targets = len(targets[1].data['ID'])
                            pqr = np.zeros((len_targets,6))
                            NUNIQUE = np.zeros(len_targets)
                            count =0
                            for i in range(chunks):
                                
                                targets_i = pf.open(path_templates+'/{0}targets_sample_g_{1}.fits'.format(add,i))
                                print (path_templates+'/{0}targets_sample_g_{1}.fits'.format(add,i))
                                try:
                                    NUNIQUE[count:count+len(targets_i[1].data['ID'])] = targets_i[1].data['NUNIQUE']
                                    pqr[count:count+len(targets_i[1].data['ID']),:] = targets_i[1].data['PQR']
                                    count+=len(targets_i[1].data['ID'])
                                except:
                                    NUNIQUE[count:count+len(targets_i[1].data['ID'])] = targets_i[1].data['NUNIQUE'][count:count+len(targets_i[1].data['ID']),:]
                                    pqr[count:count+len(targets_i[1].data['ID']),:] = targets_i[1].data['PQR'][count:count+len(targets_i[1].data['ID']),:]
                                    count+=len(targets_i[1].data['ID'])           
                            c = targets[1].columns
                            cols = [] 
                            #if 1==1:
                            try:
                                c['PQR'].array=pqr
                            except:
                                pass
                            try:
                                c['NUNIQUE'].array=NUNIQUE
                            except:
                                pass
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
                            try:
                                targets_i.writeto(output_folder+'/{0}targets_sample_g.fits'.format(add,i))
                            except:
                                try:
                                    targets_i.writeto(output_folder+'/{0}targets_sample_g.fits'.format(add,i),clobber = True)# 
                                except:
                                    pass

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
