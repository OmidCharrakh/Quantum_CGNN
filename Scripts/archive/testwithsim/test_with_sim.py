#script to test distance measures with simulated datasets
import pandas as pd
import distance_functions as distFun
import torch as th
from torch.utils.data import DataLoader
import numpy as np
import time
import utilities_functions as uFun
import matplotlib.pyplot as plt
from tqdm import tqdm #library for showing progress bars
from copy import deepcopy

class Dist_Tester():
    def __init__(self, sample_size, reps, nb_top):
        self.sample_size = sample_size
        self.reps = reps
        self.nb_top = nb_top
        self._load_and_sample_data(sample_size)
        
    def _load_and_sample_data(self, sample_size):
        #load data
        dataHigh = pd.read_csv('./Data/dat_train.csv')
        dataHigh2 = pd.read_csv('./Data/dat_calib_high.csv')
        dataLow = pd.read_csv('./Data/dat_calib_noCorr.csv')
        #select samples
        self.dataHigh = th.Tensor(dataHigh.sample(n=sample_size).values)
        self.dataHigh2 = th.Tensor(dataHigh2.sample(n=sample_size).values)
        self.dataLow = th.Tensor(dataLow.sample(n=sample_size).values)
    
    def _create_spec_test_data(self, gen_size):
        exp_std = 1.8849964391113314
        data = th.zeros(gen_size,4)
        dat_obs = th.zeros(gen_size).normal_(0,exp_std)
        dat_set = th.zeros(gen_size).uniform_(0, 1)
        data[:,0] = dat_obs
        data[:,1] = dat_obs
        data[:,2] = dat_set
        data[:,3] = dat_set
        return data

    def compare_dists(self, distFuns, batch_sizes):
        labels_vals = ['toHigh','toNone']
        #prepare parameters
        labels_distFuns = [c.get_name(False) for c in distFuns]
        nb_batch_sizes = len(batch_sizes)
        nb_dFuns = len(labels_distFuns)
        nbs_batches = np.zeros(len(batch_sizes))
        labels_bs = [str(batch_size) for batch_size in batch_sizes]
        vals = np.zeros([nb_dFuns,nb_batch_sizes,2])
        stds = np.zeros([nb_dFuns,nb_batch_sizes,2])
        rTim = np.zeros([nb_dFuns,nb_batch_sizes,2])
        #main loops
        for ind_bs,batch_size in enumerate(batch_sizes):
            print('batch-size:',str(batch_size))
            for dF in distFuns:
                dF.preallocate_memory(batch_size)
            dataloaders = self._batch_data(batch_size)
            nbs_batches[ind_bs] = len(dataloaders[0])
            ind_dFun = 0
            for dFun in tqdm(distFuns):
                vals[ind_dFun,ind_bs,:],stds[ind_dFun,ind_bs,:],rTim[ind_dFun,ind_bs,:] = self._loop_over_batches(dFun,dataloaders)
                ind_dFun += 1 #done manually so that tdqm works properly
        #[show plots]
        #mean values per batch
        for k in range(nb_batch_sizes):
            fig,ax = plt.subplots()
            plot_data = np.transpose(vals[:,k,:]/vals[:,k,[0]]) #normalized
            flucts = np.transpose(stds[:,k,:]/vals[:,k,[0]]) #normalized as well
            uFun.customBarPlot(ax,plot_data,labels_vals,labels_distFuns,errorBars=flucts,logPlot=False,enforceZero=True)
            plt.title('mean distance per batch (batchsize ' + str(batch_sizes[k]) + ')', fontsize=15)
            plt.show()
        #total times for the distances
        for k in range(nb_batch_sizes):
            fig,ax = plt.subplots()
            plot_data = rTim[:,k,:]
            uFun.customBarPlot(ax,plot_data,labels_distFuns,labels_vals,logPlot=False,enforceZero=True)
            plt.title('total time (batchsize ' + str(batch_sizes[k]) + ')', fontsize=15)
            plt.show()
        #scaling of time per with batch size
        fig,ax = plt.subplots()
        plot_data = np.transpose(np.mean(rTim,2))
        uFun.customBarPlot(ax,plot_data,labels_bs,labels_distFuns,logPlot=False,enforceZero=True)
        plt.title('total time scaling with batchsize', fontsize=15)
        plt.show()
        
    def comp_marg(self, distFuns, batch_sizes):
        labels_distFuns = [c.get_name(False) for c in distFuns]
        labels_prop = ['n','u']
        labels_comp = ['nP','nS','nN','nNS','uS','uN','uNS']
        labels_jnt = [lp + '-' + lc for lp in labels_prop for lc in labels_comp]
        #prepare parameters
        nb_dFuns = len(labels_distFuns)
        nb_batch_sizes = len(batch_sizes)
        labels_bs = [str(batch_size) for batch_size in batch_sizes]
        vals = np.zeros([nb_dFuns,nb_batch_sizes,2,7]) #there are 2x7=14 values to test
        stds = np.zeros([nb_dFuns,nb_batch_sizes,2,7])
        rTim = np.zeros([nb_dFuns,nb_batch_sizes])
        #main loops
        for ind_bs,batch_size in enumerate(batch_sizes):
            print('batch-size:',str(batch_size))
            for dF in distFuns:
                dF.preallocate_memory(batch_size)
            nb_batches = np.round(self.sample_size/batch_size).astype(int)
            vals[:,ind_bs,:,:], stds[:,ind_bs,:,:], rTim[:,ind_bs] = self._loop_margs_over_pseudo_batches(distFuns,nb_batches,batch_size)
        #process values
        diffs = vals - 1
        snr = diffs / stds
        #[show plots]
        #distance for each batch_size and dFun
        fig,ax = plt.subplots()
        plot_data = np.transpose(np.reshape(np.mean(vals,axis=1),[nb_dFuns,14]))
        flucts = np.transpose(np.reshape(np.mean(stds,axis=1),[nb_dFuns,14]))
        uFun.customBarPlot(ax,plot_data,labels_jnt,labels_distFuns,errorBars=flucts,logPlot=True)
        ax.axhline(1,c='black',ls=':')
        plt.title('distance mean over batch-sizes', fontsize=15)
        plt.show()
        #snr for each batch_size and dFun
        fig,ax = plt.subplots()
        plot_data = np.transpose(np.reshape(np.mean(snr,axis=1),[nb_dFuns,14]))
        uFun.customBarPlot(ax,plot_data,labels_jnt,labels_distFuns,logPlot=True)
        ax.axhline(100,c='green',ls=':')
        ax.axhline(1,c='black',ls=':')
        plt.title('snr mean over batch-sizes', fontsize=15)
        plt.show()
        #mean snr over batch_size
        fig,ax = plt.subplots()
        plot_data = np.mean(np.mean(snr,3),2)
        uFun.customBarPlot(ax,plot_data,labels_distFuns,labels_bs,logPlot=True)
        plt.title('mean snr', fontsize=15)
        plt.show()
        #runtimes over batch_size
        fig,ax = plt.subplots()
        plot_data = rTim
        uFun.customBarPlot(ax,plot_data,labels_distFuns,labels_bs,logPlot=False,enforceZero=True)
        plt.title('total time', fontsize=15)
        plt.show()
        
    def opt_distance(self, dist, batch_sizes, test_parameters, par_name, show_plots):
        nb_bs = len(batch_sizes)
        nb_par = len(test_parameters)
        vals = np.zeros([nb_bs,nb_par,2,self.reps]) #the 2 in 3rd dimension stands for the two tests: difference between two highly correlated distributions (eff. zero) and difference between correlated and uncorrelated (benchmark)
        stds = np.zeros([nb_bs,nb_par,2,self.reps])
        rTim = np.zeros([nb_bs,nb_par,2,self.reps])
        print('Optimizing Distance [' + dist.get_name(True) + ']')
        for ind_bs,bs in enumerate(batch_sizes):
            print('batch-size:',str(bs))
            dist.preallocate_memory(bs)
            for rep in range(self.reps):
                print('rep [',str(rep+1),'/',str(self.reps),']')
                dataloaders = self._batch_data(bs)
                ind_p = 0
                for p in tqdm(test_parameters):
                    dist.set_main_par(p)
                    vals[ind_bs,ind_p,:,rep],stds[ind_bs,ind_p,:,rep],rTim[ind_bs,ind_p,:,rep] = self._loop_over_batches(dist,dataloaders)
                    ind_p += 1 #done manually so that tdqm works properly
        #calculate mean values over reps
        vals = np.mean(vals,axis=3)
        stds = np.mean(stds,axis=3)
        rTim = np.mean(rTim,axis=3)
        #process data
        diffs = (vals[:,:,1] - vals[:,:,0]) / vals[:,:,0]
        snr = (vals[:,:,1] - vals[:,:,0]) / stds[:,:,1]
        #get rid of negative values
        diffs[diffs <= 0] = 1e-9
        snr[snr <= 0] = 1e-9
        #[FIND best SNRs and diffs]
        self._display_n_max_values(self.nb_top,snr,'SNR',batch_sizes,test_parameters,['bs',par_name],rTim[:,:,1])
        self._display_n_max_values(self.nb_top,diffs,'DIFF',batch_sizes,test_parameters,['bs',par_name],rTim[:,:,1])
        if not show_plots:
            return
        #[PLOTS]
        #prepare general plot parameters
        labels_bs = [str(batch_size) for batch_size in batch_sizes]
        ticks_bs = list(range(len(batch_sizes)))
        labels_par = [str(par) for par in test_parameters]
        ticks_par = list(range(len(test_parameters)))
        sX,sY = np.meshgrid(ticks_par,ticks_bs)
        #plot optimization surface - diffs
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(sX,sY,np.log10(diffs),cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)
        fig.suptitle('surf - diffs', fontsize=16)
        #plot optimization surface - snr
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(sX,sY,np.log10(snr),cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)
        fig.suptitle('surf - snr', fontsize=16)
        #plot parameter dependence for each batch_size
        for ind,bs in enumerate(batch_sizes):
            fig,ax = plt.subplots()
            ax.plot(ticks_par,diffs[ind,:])
            ax.plot(ticks_par,stds[ind,:,1])
            ax.plot(ticks_par,snr[ind,:])
            ax.plot(ticks_par,np.mean(rTim,axis=2)[ind,:])
            ax.legend(['diffs','std','snr','time'])
            ax.axhline(1,c='black',ls=':')
            ax.axhline(100,c='green',ls='--')
            ax.set_yscale('log')
            ax.set_xticks(ticks_par)
            ax.set_xticklabels(labels_par)
            fig.suptitle('dependence - bs=' + str(bs), fontsize=16)
        #plot duration scaling over batches
        fig,ax = plt.subplots()
        rTim_bs = np.mean(np.mean(rTim,axis=2),axis=1)
        ax.plot(ticks_bs,rTim_bs)
        ax.set_xticks(ticks_bs)
        ax.set_xticklabels(labels_bs)
        fig.suptitle('Duration over bs', fontsize=16)
        plt.show()
        
    def opt_dist_2par(self, dist, batch_size, test_parameters_1, test_parameters_2, par_names, show_plots):
        nb_par_1 = len(test_parameters_1)
        nb_par_2 = len(test_parameters_2)
        vals = np.zeros([nb_par_1,nb_par_2,2,self.reps]) #the 2 in 3rd dimension stands for the two tests: difference between two highly correlated distributions (eff. zero) and difference between correlated and uncorrelated (benchmark)
        stds = np.zeros([nb_par_1,nb_par_2,2,self.reps])
        rTim = np.zeros([nb_par_1,nb_par_2,2,self.reps])
        print('Optimizing Dist [' + dist.get_name(short=True) + '] - 2 Par')
        dist.preallocate_memory(batch_size)
        for rep in range(self.reps):
            print('rep [',str(rep+1),'/',str(self.reps),']')
            dataloaders = self._batch_data(batch_size)
            for ind_p1,p1 in enumerate(test_parameters_1):
                print(par_names[0]+':',str(p1))
                ind_p2 = 0
                for p2 in tqdm(test_parameters_2):
                    dist.set_main_pars(p1,p2)
                    vals[ind_p1,ind_p2,:,rep],stds[ind_p1,ind_p2,:,rep],rTim[ind_p1,ind_p2,:,rep] = self._loop_over_batches(dist,dataloaders)
                    ind_p2 += 1 #done manually so that tdqm works properly
        #calculate mean values over reps
        vals = np.mean(vals,axis=3)
        stds = np.mean(stds,axis=3)
        rTim = np.mean(rTim,axis=3)
        #process data
        diffs = (vals[:,:,1] - vals[:,:,0]) / vals[:,:,0]
        snr = (vals[:,:,1] - vals[:,:,0]) / stds[:,:,1]
        #get rid of negative values
        diffs[diffs <= 0] = 1e-9
        snr[snr <= 0] = 1e-9
        #[FIND best SNRs and diffs]
        self._display_n_max_values(self.nb_top,snr,'SNR',test_parameters_1,test_parameters_2,par_names,rTim[:,:,1])
        self._display_n_max_values(self.nb_top,diffs,'DIFF',test_parameters_1,test_parameters_2,par_names,rTim[:,:,1])
        if not show_plots:
            return
        #[PLOTS]
        #prepare general plot parameters
        labels_p1 = [str(p) for p in test_parameters_1]
        ticks_p1 = list(range(len(test_parameters_1)))
        labels_p2 = [str(p) for p in test_parameters_2]
        ticks_p2 = list(range(len(test_parameters_2)))
        sX,sY = np.meshgrid(ticks_p2,ticks_p1)
        #plot optimization surface - diffs
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(sX,sY,np.log10(diffs),cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)
        fig.suptitle('surf - diffs', fontsize=16)
        #plot optimization surface - snr
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(sX,sY,np.log10(snr),cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)
        fig.suptitle('surf - snr', fontsize=16)
        #plot parameter_2 dependence for each parameter_1
        for ind,p1 in enumerate(test_parameters_1):
            fig,ax = plt.subplots()
            ax.plot(ticks_p2,diffs[ind,:])
            ax.plot(ticks_p2,stds[ind,:,1])
            ax.plot(ticks_p2,snr[ind,:])
            ax.plot(ticks_p2,np.mean(rTim,axis=2)[ind,:])
            ax.legend(['diffs','std','snr','time'])
            ax.axhline(1,c='black',ls=':')
            ax.axhline(100,c='green',ls='--')
            ax.set_yscale('log')
            ax.set_xticks(ticks_p2)
            ax.set_xticklabels(labels_p2)
            fig.suptitle(par_names[0] + '=' + str(p1) + ' changing ' + par_names[1], fontsize=16)
        #plot duration scaling over parameters_1
        fig,ax = plt.subplots()
        rTim_bs = np.mean(np.mean(rTim,axis=2),axis=1)
        ax.plot(ticks_p1,rTim_bs)
        ax.set_xticks(ticks_p1)
        ax.set_xticklabels(labels_p1)
        fig.suptitle('mean runtimes over ' + par_names[0], fontsize=16)
        plt.show()        
        
    def opt_marg(self, dist, batch_sizes, test_parameters, par_name, show_plots):
        #prepare parameters
        nb_bs = len(batch_sizes)
        nb_par = len(test_parameters)
        zer = np.zeros([nb_bs,nb_par,2,self.reps])
        vals = np.zeros([nb_bs,nb_par,2,2,4,self.reps])
        stds = np.zeros([nb_bs,nb_par,2,2,4,self.reps])
        rTim = np.zeros([nb_bs,nb_par,2,2,4,self.reps])
        #3rd-5th[2,3,4] dim stands for [reg. u/n, comp. set u/n, comp ind]
        #6th[5] dim stands for reps 
        #create labels
        #labels_prop = ['N','U']
        #labels_comp = [['N_p','N_s','N_n','N_ns'],['U_t','U_s','U_n','U_ns']]
        #labels_jnt = [[[lp + '-' + labels_comp[indS][indD] for indD in range(4)] for indS in range(2)] for lp in labels_prop]
        #main loops
        print('Optimizing Distance [' + dist.get_name(True) + ']')
        for ind_bs,bs in enumerate(batch_sizes):
            print('batch-size:',str(bs))
            nb_batches = np.round(self.sample_size/bs).astype(int)
            dist.preallocate_memory(bs)
            for rep in range(self.reps):
                print('rep [',str(rep+1),'/',str(self.reps),']')
                ind_p = 0
                for p in tqdm(test_parameters):
                    dist.set_main_par(p)
                    zer[ind_bs,ind_p,:,rep],vals[ind_bs,ind_p,:,:,:,rep],stds[ind_bs,ind_p,:,:,:,rep],rTim[ind_bs,ind_p,:,:,:,rep] = self._loop_margs_over_pseudo_batches(dist,nb_batches,bs)
                    ind_p += 1 #done manually so that tdqm works properly
        #calculate mean values over reps
        zer = np.mean(zer,axis=3)
        vals = np.mean(vals,axis=5)
        stds = np.mean(stds,axis=5)
        rTim = np.mean(rTim,axis=5)
        #process data
        zer = zer[:,:,:,None,None]
        diffs = (vals - zer) / zer
        snr = (vals - zer) / stds
        #find relevant snr/diffs, i.e. comparing only N with N and U with U
        snrRel = np.mean([np.mean(snr[:,:,0,0,:],axis=2),np.mean(snr[:,:,1,1,:],axis=2)],axis=0)
        snrWrst = np.min([np.min(snr[:,:,0,0,:],axis=2),np.min(snr[:,:,1,1,:],axis=2)],axis=0)
        diffsRel = np.mean([np.mean(diffs[:,:,0,0,:],axis=2),np.mean(diffs[:,:,1,1,:],axis=2)],axis=0)
        stdsRel = np.sqrt(np.mean([np.mean(stds[:,:,0,0,:]**2,axis=2),np.mean(stds[:,:,1,1,:]**2,axis=2)],axis=0))
        rTimRel = np.mean([np.mean(rTim[:,:,0,0,:],axis=2),np.mean(rTim[:,:,1,1,:],axis=2)],axis=0)
        #get rid of negative values
        diffs[diffs <= 0] = 1e-9
        snr[snr <= 0] = 1e-9
        diffsRel[diffsRel <= 0] = 1e-9
        snrRel[snrRel <= 0] = 1e-9
        #[FIND best SNRs and diffs]
        self._display_n_max_values(self.nb_top,snrRel,'SNR',batch_sizes,test_parameters,['bs',par_name],rTimRel,extra_qnt=snrWrst,extra_name='SNRwc')
        self._display_n_max_values(self.nb_top,diffsRel,'DIFF',batch_sizes,test_parameters,['bs',par_name],rTimRel,extra_qnt=snrWrst,extra_name='SNRwc')
        if not show_plots:
            return
        #[PLOTS]
        #prepare general plot parameters
        labels_bs = [str(batch_size) for batch_size in batch_sizes]
        ticks_bs = list(range(len(batch_sizes)))
        labels_par = [str(par) for par in test_parameters]
        ticks_par = list(range(len(test_parameters)))
        sX,sY = np.meshgrid(ticks_par,ticks_bs)
        #plot optimization surface - diffs
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(sX,sY,np.log10(diffsRel),cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)
        fig.suptitle('surf - diffs', fontsize=16)
        #plot optimization surface - snr
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(sX,sY,np.log10(snrRel),cmap=plt.cm.coolwarm,linewidth=0,antialiased=False)
        fig.suptitle('surf - snr', fontsize=16)
        #plot parameter dependence for each batch_size
        for ind,bs in enumerate(batch_sizes):
            fig,ax = plt.subplots()
            ax.plot(ticks_par,diffsRel[ind,:])
            ax.plot(ticks_par,stdsRel[ind,:])
            ax.plot(ticks_par,snrRel[ind,:])
            ax.plot(ticks_par,rTimRel[ind,:])
            ax.plot(ticks_par,snrWrst[ind,:])
            ax.legend(['diffs','std','snr','time','snr(w.c.)'])
            ax.axhline(1,c='black',ls=':')
            ax.axhline(100,c='green',ls='--')
            ax.set_yscale('log')
            ax.set_xticks(ticks_par)
            ax.set_xticklabels(labels_par)
            fig.suptitle('dependence - bs=' + str(bs), fontsize=16)
        #plot duration scaling over batches
        fig,ax = plt.subplots()
        rTim_bs = np.mean(np.mean(np.mean(np.mean(rTim,axis=4),axis=3),axis=2),axis=1)
        ax.plot(ticks_bs,rTim_bs)
        ax.set_xticks(ticks_bs)
        ax.set_xticklabels(labels_bs)
        fig.suptitle('Duration over bs', fontsize=16)
        plt.show()
        
    # def check_spec_dist(self, dFun, batch_sizes):
    #     #refDFun = distFun.Binned_Marg('JS',sampling_rate = 10)
    #     refDFun = distFun.EDF_Marg('CvM')
    #     for ind,bs in enumerate(batch_sizes):
    #         dataloaders = self._batch_data(bs)
    #         test = th.zeros([len(dataloaders[0]),2,4])
    #         for ind_batch,(dHigh,dHigh2,dLow) in enumerate(zip(dataloaders[0],dataloaders[1],dataloaders[2])):
    #             dSpec = self._create_spec_test_data(bs)
    #             test[ind_batch,0,0] = dFun.forward(dHigh,dHigh2)
    #             test[ind_batch,0,1] = dFun.forward(dHigh,dLow)
    #             test[ind_batch,0,2] = dFun.forward(dSpec,dHigh)
    #             test[ind_batch,0,3] = dFun.forward(dSpec,dLow)
    #             test[ind_batch,1,0] = refDFun.forward(dHigh,dHigh2)
    #             test[ind_batch,1,1] = refDFun.forward(dHigh,dLow)
    #             test[ind_batch,1,2] = refDFun.forward(dSpec,dHigh)
    #             test[ind_batch,1,3] = refDFun.forward(dSpec,dLow)
    #         stds = th.std(test,dim=0)
    #         test = th.mean(test,dim=0)
        
    def _batch_data(self, batch_size):
        dataloaders = [None for k in range(3)]
        dataloaders[0] = DataLoader(self.dataHigh,batch_size=batch_size,shuffle=True,drop_last=True)
        dataloaders[1] = DataLoader(self.dataHigh2,batch_size=batch_size,shuffle=True,drop_last=True)
        dataloaders[2] = DataLoader(self.dataLow,batch_size=batch_size,shuffle=True,drop_last=True)
        return dataloaders
        
    def _loop_over_batches(self, dist, dataloaders):
        distances_batch = np.zeros([len(dataloaders[0]),2])
        runTimes_batch = np.zeros([len(dataloaders[0]),2])
        #loop over batches
        for ind_batch,(dHigh,dHigh2,dLow) in enumerate(zip(dataloaders[0],dataloaders[1],dataloaders[2])):
            cTime0 = time.time()
            distances_batch[ind_batch,0] = dist.forward(dHigh,dHigh2)
            cTime1 = time.time()
            distances_batch[ind_batch,1] = dist.forward(dHigh,dLow)
            cTime2 = time.time()
            #store runtimes
            runTimes_batch[ind_batch,0] = cTime1 - cTime0
            runTimes_batch[ind_batch,1] = cTime2 - cTime1
        #calculate mean and sum values + std-deviation
        distances = np.mean(distances_batch,axis=0)
        dist_std = np.std(distances_batch,axis=0)
        runTimes = np.sum(runTimes_batch,axis=0)
        return distances, dist_std, runTimes
    
    def _loop_margs_over_pseudo_batches(self, dFun, nb_batches, batch_size):
        vals = np.zeros([nb_batches,2,2,4])
        rTim = np.zeros([nb_batches,2,2,4])
        zer = np.zeros([nb_batches,2])
        for ind_b in range(nb_batches):
            #create new data-set
            mdat_prop, mdat_prop2, mdat_comp_n, mdat_comp_u = uFun.create_marginal_test_data(batch_size)
            #loop over proper distributions
            for ind_prop,(dProp,dProp2) in enumerate(zip(mdat_prop,mdat_prop2)):
                zer[ind_b,ind_prop] = dFun.forward(dProp,dProp2)
                #loop over bad distributions
                for ind_comp,(dCompN,dCompU) in enumerate(zip(mdat_comp_n,mdat_comp_u)):
                    sTime = time.time()
                    vals[ind_b,ind_prop,0,ind_comp] = dFun.forward(dProp,dCompN)
                    rTim[ind_b,ind_prop,0,ind_comp] = time.time() - sTime
                    sTime = time.time()
                    vals[ind_b,ind_prop,1,ind_comp] = dFun.forward(dProp,dCompU)
                    rTim[ind_b,ind_prop,1,ind_comp] = time.time() - sTime
        stds = np.std(vals,axis = 0)
        vals = np.mean(vals,axis = 0)
        rTim = np.sum(rTim,axis = 0)
        zer = np.mean(zer,axis=0)
        return zer, vals, stds, rTim
    
    def _display_n_max_values(self, nb_top, quantity, quantity_name, parameters_1, parameters_2, par_names, run_times, extra_qnt = [], extra_name = False):        
        test_q = deepcopy(quantity)
        print('\n\nTop',str(nb_top),quantity_name,':')
        for k in range(nb_top):
            #find maximal value and its location
            max_val = np.amax(test_q)
            ind = np.where(test_q == max_val)
            ind = [ind[0][0],ind[1][0]]
            ind_coord = (np.array([ind[0]]),np.array([ind[1]]))
            #display result
            rTim = run_times*1000;
            if extra_name:
                print('[',str(k+1),'] ',str(max_val),' (' + par_names[0] + ':',str(parameters_1[ind[0]]),',',par_names[1],':',str(parameters_2[ind[1]]),',',extra_name,':',str(extra_qnt[ind_coord]),',',str(rTim[ind[0],ind[1]].round(decimals = 1)),'ms)')
            else:
                print('[',str(k+1),'] ',str(max_val),' (' + par_names[0] + ':',str(parameters_1[ind[0]]),',',par_names[1],':',str(parameters_2[ind[1]]),',',str(rTim[ind[0],ind[1]].round(decimals = 1)),'ms)')
            #exclude current max value from next iteration
            test_q[ind_coord] = 0
            
#[INITIALIZE]
dT = Dist_Tester(sample_size = 32000, reps = 10, nb_top = 20)
       
#[SINGLE TESTS OF FULL DISTANCES]
#TESTING BinnedD
# test_dist = distFun.BinnedD('JS')
# batch_sizes = [8000]
# test_parameters = list(range(4,6))
# dT.opt_distance(test_dist,batch_sizes,test_parameters,'samp_rate',show_plots=True)
#dT.opt_marg(test_dist,batch_sizes,test_parameters,'samp_rate',show_plots=True)
#RMSD 8000/4 - 15.8/1.3/4.9ms
#JS 8000/4 - 22.1/4.5/5.2ms
#He 8000/4 - 26/1.5/4.6ms
#Tot 8000/5 - 5.4/1.1/4.9ms

#TESTING NpMom
# test_dist = distFun.NpMom(num_moments = 4, weighting_exp = 2)
# batch_sizes = [8000]
# test_parameters = list(range(3,6))
# dT.opt_distance(test_dist,batch_sizes,test_parameters,'num_moments',show_plots=True)
#dT.opt_marg(test_dist,batch_sizes,test_parameters,'num_moments',show_plots=True)

#TESTING CndD
# test_dist = distFun.CndD(sample_weighting = True, sampling_rate = 4, num_std_moments = 4, weighting_exp = 2)
# batch_size = 8000
# batch_sizes = [8000]
# test_parameters_1 = list(range(3,6))
# test_parameters_2 = list(range(3,7))
#dT.opt_distance(test_dist,batch_sizes,test_parameters_1,'samp_rate',show_plots=True)
# dT.opt_dist_2par(test_dist,batch_size,test_parameters_1,test_parameters_2,['samp_rate','num_std_moments'],show_plots=True)
#dT.opt_marg(test_dist,batch_sizes,test_parameters_1,'samp_rate',show_plots=True)

#TESTING CorrD
# test_dist = distFun.CorrD(corr_types=[0,1],sampling_rate=5)
# batch_sizes = [8000]
# #batch_sizes = [2000,4000,6400,8000]
# test_parameters = list(range(3,6))
# # dT.opt_distance(test_dist,batch_sizes,test_parameters,'sampling_rate',show_plots=True)
# dT.opt_marg(test_dist,batch_sizes,test_parameters,'sampling_rate',show_plots=True)
#a[1,1] 8000/2 - 51.0/10.2/65.4ms
#a[1,1] 8000/3 - 40.0/7.8/101.9ms
#a[1,1] 8000/4 - 36.1/6.0/147.4ms
#a[1,1] 8000/5 - 40.3/5.2/205.7ms
#3[0,1] 8000/2 - 57.1/10.3/23ms
#3[0,1] 8000/3 - 46.8/10.3/28.7ms
#3[0,1] 8000/4 - 39.4/8.1/31.9ms
#3[0,1] 8000/5 - 55.1/7.0/36.7ms
#2[1,0] 8000/3 - 52.2/6.4/71.1ms

stds
tensor([0.2756, 0.2097, 0.0546, 0.0736], grad_fn=<StdBackward0>)

stds
tensor([1.8946, 1.8786, 0.2896, 0.2889])

moms
tensor([-0.0260, -0.0590, -3.3363, -1.3659], grad_fn=<DivBackward0>)

moms
tensor([  1.0376,   3.0952, 671.2680, 204.0572], grad_fn=<DivBackward0>)

#TESTING CorrD_pen
# test_dist = distFun.CorrD_pen(sampling_rate=5)
# batch_sizes = [2000,4000,6400,8000]
# test_parameters = list(range(2,10))
# # dT.opt_distance(test_dist,batch_sizes,test_parameters,'sampling_rate',show_plots=True)
# dT.opt_marg(test_dist,batch_sizes,test_parameters,'sampling_rate',show_plots=True)

#TESTING CorrD_N
# test_dist = distFun.CorrD_N()
# batch_sizes = [2000,4000,6400,8000]
# test_parameters = list(range(2,4))
# # dT.opt_distance(test_dist,batch_sizes,test_parameters,'dummy_par',show_plots=True)
# dT.opt_marg(test_dist,batch_sizes,test_parameters,'dummy_par',show_plots=True)

#TESTING MargD
test_dist = distFun.MargD2(num_std_moments = 6)
batch_sizes = [8000]
test_parameters = list(range(3,7))
# dT.opt_distance(test_dist,batch_sizes,test_parameters,'num_std_moments',show_plots=True)
dT.opt_marg(test_dist,batch_sizes,test_parameters,'num_std_moments',show_plots=True)

#TESTING BinnedMarg
# test_dist = distFun.Binned_Marg('He',sampling_rate=5)
# batch_sizes = [8000]
# test_parameters = list(range(3,6))
# #dT.opt_distance(test_dist,batch_sizes,test_parameters,'sampling_rate',show_plots=True)
# dT.opt_marg(test_dist,batch_sizes,test_parameters,'sampling_rate',show_plots=True)

#TESTING EDF_Marg
# test_dist = distFun.EDF_Marg('KS')
# batch_sizes = [8000]
# test_parameters = list(range(2,4))
# #dT.opt_distance(test_dist,batch_sizes,test_parameters,'dummy_par',show_plots=True)
# dT.opt_marg(test_dist,batch_sizes,test_parameters,'dummy_par',show_plots=True)

#TESTING MMD_cdt
# test_dist = distFun.MMD_cdt()
# batch_sizes = [400,500]
# test_parameters = [[0.1,1],[1],[1,10],[0.1,1,10]]
# # dT.opt_distance(test_dist,batch_sizes,test_parameters,'bandwidths',show_plots=True)
# dT.opt_marg(test_dist,batch_sizes,test_parameters,'bandwidths',show_plots=True)
# #[0.01] 400 0/0/91.8
# #[0.1] 400 0.5/0.3/91.3
# #[1] 400 1.3/0.3/83.4
# #[10] 400 1.1/0/413.0
# #[100] 400 0.3/0/422.5
# #[0.1] 500 0.5/0.3/147.4
# #[1] 500 1.7/0.4/152.9
# #[10] 500 1.5/0/539.5

#TESTING MMD_Fourier
# test_dist = distFun.MMD_Fourier(n_RandComps = 100)
# #batch_sizes = [400,500]
# batch_sizes = [8000]
# #test_parameters = [[0.01],[0.1],[1],[10],[100]]
# test_parameters = [[0.01,0.1,1,10,100],[0.1,1,10],[0.1,1,10,100],[0.01,0.1,1,10],[1],[10],[100],[1,10],[0.1,1],[10,100]]
# dT.opt_distance(test_dist,batch_sizes,test_parameters,'bandwidths',show_plots=True)

#TESTING MMD_s_pro
# test_dist = distFun.MMD_s_pro(kernels = ['RBF'])
# #test_dist = distFun.MMD_s_pro(kernels = ['Cos'])
# #test_dist = distFun.MMD_s_pro(kernels = ['Exp'])
# batch_size = 400
# # test_parameters_1 = [[0.01],[0.1],[1],[10],[100]]
# # test_parameters_2 = [[0.01],[0.1],[1],[10],[100]]
# test_parameters_1 = [[0.1,1],[1],[1,10],[0.1,1,10]]
# test_parameters_2 = [[0.1],[1],[10],[0.1,1],[1,10],[0.1,1,10]]
# dT.opt_dist_2par(test_dist,batch_size,test_parameters_1,test_parameters_2,['bandwidths','variances'],show_plots=True)
# #RBF: 500/1/any(0.01,0.1,1,10,100) 1.4/0.5/180
# #Cos: 500/1/any(0.01,0.1,1,10,100) 1.3/1.4/215
# #Exp: 500/1/any(0.01,0.1,1,10,100) 1.4/0/280
# #Exp: 500/0.1/any(0.01,0.1,1,10,100) 1.1/0/280
# #RBF: 500/[1,10]/[0.1,1,10] 1.4/0.6/220
# #Cos: 500/[1,10]/[0.1,1,10] 1.4/1.5/250
# #Exp: 500/[1,10]/[0.1,1,10] 1.5/0.3/290

# [OLD] #######################################################################

#[JOINT TEST OF FULL DISTANCES]
# #batch_sizes = [500,1000,2000]
# batch_sizes = [500,1000,2000,5000,8000]
# #distFuns = [distFun.MMD_cdt(),distFun.CorrD_pen(wg_penalty=0),distFun.BinnedD('JS',6),distFun.CorrD([1,1,1],6),distFun.CorrD([1,1,1],7),distFun.CorrD([1,1,1],8)]
# distFuns = [distFun.CorrD_pen(wg_penalty=0),distFun.BinnedD('JS',6),distFun.CorrD([1,1],6),distFun.CorrD([1,1],7),distFun.CorrD([1,1],8)]
# #distFuns = [distFun.BinnedD('JS',5),distFun.BinnedD('He',5),distFun.BinnedD('JS',6),distFun.BinnedD('He',6),distFun.BinnedD('JS',7),distFun.BinnedD('He',7)]
# #distFuns = [distFun.CorrD([1,0,0],6),distFun.CorrD([1,1,1],6),distFun.CorrD([1,0,0],7),distFun.CorrD([1,1,1],7),distFun.CorrD([1,0,0],8),distFun.CorrD([1,1,1],8)]
# dT.compare_dists(distFuns,batch_sizes)

#[JOINT TEST OF MARGINAL DISTANCES]
# batch_sizes = [500,1000,2000,5000,8000]
# distFuns = [distFun.MargD(num_std_moments=4),distFun.EDF_Marg('CvM'),distFun.Binned_Marg('JS',sampling_rate=50)]
# #distFuns = [distFun.MargD(num_std_moments=2),distFun.MargD(num_std_moments=3),distFun.MargD(num_std_moments=4)]
# #distFuns = [distFun.EDF_Marg('CvM',interp_samples=1),distFun.EDF_Marg('CvM',interp_samples=10),distFun.EDF_Marg('CvM',interp_samples=100),distFun.EDF_Marg('CvM',interp_samples=1000)]
# #distFuns = [distFun.EDF_Marg('AD',interp_samples=1),distFun.EDF_Marg('AD',interp_samples=10),distFun.EDF_Marg('AD',interp_samples=100),distFun.EDF_Marg('CvM',interp_samples=1000)]
# #distFuns = [distFun.EDF_Marg('KS'),distFun.EDF_Marg('CvM'),distFun.EDF_Marg('AD')]
# #distFuns = [distFun.Binned_Marg('JS',sampling_rate=10),distFun.Binned_Marg('JS',sampling_rate=100),distFun.Binned_Marg('JS',sampling_rate=1000)]
# dT.comp_marg(distFuns,batch_sizes)

#[JOINT TEST OF SPECIAL MARGINAL CORRELATION DISTANCE]
# batch_sizes = [500,1000,2000,5000,8000]
# batch_sizes = [5000]
# dT.check_spec_dist(distFun.Marg_Corr(),batch_sizes)