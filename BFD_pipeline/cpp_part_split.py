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



def f_(uu,output_folder,noise_tiers,path_cpp):
    os.system('cp {0}/noisetiers.fits {0}/noisetiers_{1}.fits'.format(output_folder,noise_tiers[uu]))
    str_ = '{2}tierSelection {0}/noisetiers_{1}.fits {0}/templates_NOISETIER_{1}.fits'.format(output_folder,noise_tiers[uu],path_cpp)
    os.system(str_)      


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





def cpp_part(output_folder,**config):
    config['output_folder'] = output_folder
    path_cpp = config['path_cpp']
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
                                
                            _ = pool.map(partial(f_,output_folder=output_folder,noise_tiers=noise_tiers,path_cpp=path_cpp),idx_)
                            
                            
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
                        str_ = 'python BFD_pipeline/assignSelection.py {0} {1}'.format(noistierfile,targetfile)
                        os.system(str_)
                    comm.bcast(run_count,root = 0)
                    comm.Barrier() 
                
                
                
                
                if 'split' in todos:
                    
                    '''
                    option 1)
                    default. splitted targets into chunks, and everything analyse din noisetiers
                    option 2)
                    split targtes into noisetiers
                    option 3)
                    split targtes into noisetiers + templates chunk splitting.
                    '''
                    
                    if config['option'] == 'default':
                        print ('split *************')
                        chunks = config['chunks']
                        len_targets = len(targets[1].data['ID'])
                        chunk_size = math.ceil(len_targets/chunks)
                        targets_i = copy.copy(targets)
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
                        
                        
                    elif config['option'] == 'noisetiers':
                        
                        print ('split *************')
                        chunks = len(noise_tiers)
                        # we're splitting the target file into chunks and then assemble it back ***
                        len_targets = len(targets[1].data['ID'])
        
                        targets_i = copy.copy(targets)
                        #for i in range(chunks):
                        run_count = 0
                        while run_count<chunks:
                            comm = MPI.COMM_WORLD
                            if run_count+comm.rank<chunks:

                                mask = targets[1].data['NOISETIER']  == noise_tiers[run_count+comm.rank]
                                c = targets[1].columns
                                cols = [] 
                                for k in c:
                                    try:
                                        cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name][mask,:]))
                                    except:
                                        cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name][mask]))


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
                                    targets_i.writeto(output_folder+'/nt_{0}targets_sample_g_{1}.fits'.format(add,noise_tiers[run_count+comm.rank]))
                                except:
                                    try:
                                        targets_i.writeto(output_folder+'/nt_{0}targets_sample_g_{1}.fits'.format(add,noise_tiers[run_count+comm.rank]),clobber = True)# 
                                    except:
                                        pass
                            run_count+=comm.size
                            comm.bcast(run_count,root = 0)
                            comm.Barrier() 
                        
                        
                        
                    elif config['option'] == 'noisetiers_chunks':
                        
                        
                        print ('split noisetiers_chunks *************')
                        chunks = len(noise_tiers)
                        # we're splitting the target file into chunks and then assemble it back ***
                        len_targets = len(targets[1].data['ID'])
                        print ('len targets, ', len_targets)
                        print ('chunks, ',chunks)
                        targets_i = copy.copy(targets)
                        #for i in range(chunks):
                        run_count = 0
                        while run_count<chunks:
                            comm = MPI.COMM_WORLD
                            if run_count+comm.rank<chunks:
                                
                                
                                
                                
                                # divide templates into chunks +++++++++++++++++
                                
                                
                                
                                
                                
                                templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[run_count+comm.rank])
                                templates_ = pf.open(templatesfile)
                                templates_i = copy.copy(templates_)
                                len_templ = len(templates_[1].data['ID'])
                                n_batch_templates = math.ceil(len_templ/config['bacth_size_templates'])
                                print ('batch templates ', n_batch_templates)
                                print ('len templates, ', len_templ)
                                for n in range(n_batch_templates):
                                
                                    if add != 'ISm_':
                                        c = templates_[1].columns
                                        cols = [] 
                                        for k in c:
                                            try:
                                                cols.append(pf.Column(name=k.name,format=k.format,array=templates_[1].data[k.name][n*config['bacth_size_templates']:(n+1)*config['bacth_size_templates'],:]))
                                            except:
                                                cols.append(pf.Column(name=k.name,format=k.format,array=templates_[1].data[k.name][n*config['bacth_size_templates']:(n+1)*config['bacth_size_templates']]))


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
                                            templates_i.writeto(output_folder+'/templates_NOISETIER_{0}_{1}.fits'.format(noise_tiers[run_count+comm.rank],n))
                                        except:
                                            try:
                                                templates_i.writeto(output_folder+'/templates_NOISETIER_{0}_{1}.fits'.format(noise_tiers[run_count+comm.rank],n),clobber = True)# 
                                            except:
                                                pass


                                
                                
                                    
                                    
                                

                                    mask = targets[1].data['NOISETIER']  == noise_tiers[run_count+comm.rank]
                                    n_batch_targets = math.ceil(len(mask[mask])/config['bacth_size_targets'])
                                
                                    for tt in range(n_batch_targets):
                                        start = (tt)*config['bacth_size_targets']
                                        end = min([len(mask[mask]),(tt+1)*config['bacth_size_targets']])

                                        c = targets[1].columns
                                        cols = [] 
                                        for k in c:
                                            try:
                                                cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name][mask,:][start:end,:]))
                                            except:
                                                cols.append(pf.Column(name=k.name,format=k.format,array=targets[1].data[k.name][mask][start:end]))


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
                                            targets_i.writeto(output_folder+'/{0}targets_sample_g_{1}_{2}_{3}.fits'.format(add,noise_tiers[run_count+comm.rank],n,tt))
                                        except:
                                            #try:
                                                targets_i.writeto(output_folder+'/{0}targets_sample_g_{1}_{2}_{3}.fits'.format(add,noise_tiers[run_count+comm.rank],n,tt),clobber = True)# 
                                            #except:
                                            #    pass
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
                    
                    if config['option'] == 'default':
                        '''
                        #chunks target, go through all the noisetiers.
                        
                        '''
                        run_count = 0
                        while run_count< config['chunks']:
                            comm = MPI.COMM_WORLD
                            if run_count+comm.rank<config['chunks']:
                                    for u in range(len(noise_tiers)):
                                        targetfile = output_folder+'/{0}targets_sample_g_{1}.fits'.format(add,run_count+comm.rank)
                                        noistierfile = output_folder+'noisetiers.fits'
                                        templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[u])
                                        str_ = '{3}tableIntegrate -targetfile={0} -noisetierFile={1} -templateFile={2}'.format(targetfile,noistierfile,templatesfile,path_cpp)
                                        os.system(str_)
                            run_count+=comm.size
                            comm.bcast(run_count,root = 0)
                            comm.Barrier() 
                            
                            
                    elif config['option'] == 'noisetiers':
                        run_count = 0
                        while run_count< len(noise_tiers):
                            comm = MPI.COMM_WORLD
                            if run_count+comm.rank<len(noise_tiers):
                            
                                targetfile = output_folder+'/nt_{0}targets_sample_g_{1}.fits'.format(add,noise_tiers[run_count+comm.rank])
                                noistierfile = output_folder+'noisetiers.fits'
                                templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[run_count+comm.rank])
                                str_ = '{3}tableIntegrate -targetfile={0} -noisetierFile={1} -templateFile={2}'.format(targetfile,noistierfile,templatesfile,path_cpp)
                                os.system(str_)
                            run_count+=comm.size
                            comm.bcast(run_count,root = 0)
                            comm.Barrier() 
                            
                            
                    elif config['option'] == 'noisetiers_chunks':
                        run_count = 0
            
                        runstodo =[]
                        while run_count< len(noise_tiers):
                            if run_count<len(noise_tiers):
                                templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[run_count])
                                templates_ = pf.open(templatesfile)
                                len_templ = len(templates_[1].data['ID'])
                                n_batch_templates = math.ceil(len_templ/config['bacth_size_templates'])
                                del templates_
                                import gc
                                gc.collect()
                                
                                for n in range(n_batch_templates):
                                    
                                    
                                    mask = targets[1].data['NOISETIER']  == noise_tiers[run_count]
                                    n_batch_targets = math.ceil(len(mask[mask])/config['bacth_size_targets'])
                                
                                    for tt in range(n_batch_targets):
                                       
                                        
                                        
                                        targetfile = output_folder+'/{0}targets_sample_g_{1}_{2}_{3}.fits'.format(add,noise_tiers[run_count],n,tt)

                                        # try to see if it has been run already ----
                                        m_ = pf.open(targetfile)
                                        #try:
                                        #    m_[1].data['NUNIQUE']
                                        #except:
                                        if 1==1:
                                            runstodo.append([run_count,n,tt])
                            run_count+=1
                            
                        
                        run_count = 0
                        
                        while run_count< len(runstodo):
                            comm = MPI.COMM_WORLD
                            if run_count+comm.rank<len(runstodo):
                            
                                xx,n,tt = runstodo[run_count+comm.rank]
                                
                                templatesfile = output_folder+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[xx])
                               
                                targetfile = output_folder+'/{0}targets_sample_g_{1}_{2}_{3}.fits'.format(add,noise_tiers[xx],n,tt)
                                 # try to see if it has been run already ----
                                m_ = pf.open(targetfile)
                                #try:
                                #    m_[1].data['NUNIQUE']
                                #except:
                                if 1==1:
                                    noistierfile = output_folder+'noisetiers.fits'
                                    templatesfile = output_folder+'/templates_NOISETIER_{0}_{1}.fits'.format(noise_tiers[xx],n)
                                    str_ = '{3}/tableIntegrate -targetfile={0} -noisetierFile={1} -templateFile={2}'.format(targetfile,noistierfile,templatesfile,path_cpp)
                                    os.system(str_)
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
                            AREA = np.zeros(len_targets)
                            count =0
                            
                            
                            
                            
                            
                            if config['option'] == 'default':
                                chunks = config['chunks']
                                chunk_size = math.ceil(len_targets/chunks)


                                for run_count in range(chunks):
                                    start = (run_count)*chunk_size
                                    end = min([len_targets,(run_count+1)*chunk_size])


                                    targets_i = pf.open(path_templates+'/{0}targets_sample_g_{1}.fits'.format(add,run_count))           
                                    NUNIQUE[start:end] = targets_i[1].data['NUNIQUE']
                                    #AREA[start:end] = targets_i[1].data['AREA']
                                    pqr[start:end] = targets_i[1].data['PQR']

        
                            
                            elif config['option'] == 'noisetiers':
                                chunks = len(noise_tiers)
                                for run_count in range(chunks):
                                    mask_targets = targets[1].data['NOISETIER']  == noise_tiers[run_count]
                                    targets_i = pf.open(path_templates+'/nt_{0}targets_sample_g_{1}.fits'.format(add,noise_tiers[run_count]))           
                                    NUNIQUE[mask_targets] = targets_i[1].data['NUNIQUE']
                                    pqr[mask_targets] = targets_i[1].data['PQR']
                                    

                            #if 1==1:        
                            elif config['option'] == 'noisetiers_chunks':
                                chunks = len(noise_tiers)
                                for run_count in range(chunks):
                                    mask_targets = targets[1].data['NOISETIER']  == noise_tiers[run_count]


                                    templatesfile = path_templates+'/templates_NOISETIER_{0}.fits'.format(noise_tiers[run_count])
                                    templates_ = pf.open(templatesfile)
                                    len_templ = len(templates_[1].data['ID'])
                                    n_batch_templates = math.ceil(len_templ/config['bacth_size_templates'])
                                    del templates_
                                    import gc
                                    gc.collect()

                                    for n in range(n_batch_templates):                 
                                        # chunks temlates

                                        
                                        mask = targets[1].data['NOISETIER']  == noise_tiers[run_count]
                                        n_batch_targets = math.ceil(len(mask[mask])/config['bacth_size_targets'])

                                        for tt in range(n_batch_targets):
                                            start = (tt)*config['bacth_size_targets']
                                            end = min([len(mask[mask]),(tt+1)*config['bacth_size_targets']])

                                            try:
                                                targets_i = pf.open(path_templates+'/{0}targets_sample_g_{1}_{2}_{3}.fits'.format(add,noise_tiers[run_count],n,tt))  
                                                ma = np.in1d(np.arange(len(mask_targets)),np.arange(len(mask_targets))[mask_targets][start:end][targets_i[1].data['SELECT']])
                                                NUNIQUE[ma] += targets_i[1].data['NUNIQUE'][targets_i[1].data['SELECT']]
                                                pqr[ma] += targets_i[1].data['PQR'][targets_i[1].data['SELECT']]

                                                ma = np.in1d(np.arange(len(mask_targets)),np.arange(len(mask_targets))[mask_targets][start:end][~targets_i[1].data['SELECT']])
                                                NUNIQUE[ma] = targets_i[1].data['NUNIQUE'][~targets_i[1].data['SELECT']]
                                                pqr[ma] = targets_i[1].data['PQR'][~targets_i[1].data['SELECT']]
                                            except:
                                                print  ('failed ---: ',path_templates+'/{0}targets_sample_g_{1}_{2}.fits'.format(add,noise_tiers[run_count],n))



                                     
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



