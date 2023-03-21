import distance_functions as dFun
import utilities_functions as uFun

import os
import numpy as np
import pandas as pd
import torch as th; from torch.utils.data import DataLoader

class Evaluator():
    def __init__(self, data_paths, data_paths_calib, data_path_saving, show_cand_plot):
        #[CONFIG]
        cr_list = [dFun.MMD_cdt(bandwidth=[1]),
                   #dFun.MMD_s_pro(kernels=['Cos'],bandwidths=[1,10],variances=[0.1,1,10]),
                   dFun.MMD_Fourier(bandwidths = [0.1, 1, 10, 100], n_RandComps=100),
                   dFun.CorrD([1,1],3),
                   dFun.BinnedD('JS',4),
                   dFun.CorrD_N(),
                   dFun.NpMom(num_moments = 3, weighting_exp = 2),
                   dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1),
                   #dFun.Binned_Marg('He',sampling_rate=4),
                   #dFun.MargD(num_std_moments=4),
                   dFun.EDF_Marg('AD')]
        cr_inds = [0,1,2,3,4,5,6,7] #which criteria should be used for validation & testing
        self.cr = [cr_list[i] for i in cr_inds]
        self.cr_names = [cr.get_name(True) for cr in self.cr]
        self.nb_cr = len(self.cr)
        self.display_calib_digits = 3
        self.eval_name = ''
        #store settings
        self.show_cand_plot = show_cand_plot
        #load data
        print('Loading data...')
        self.full_data_train = pd.read_csv(data_paths[0])
        self.full_data_valid = pd.read_csv(data_paths[1])
        self.full_data_test = pd.read_csv(data_paths[2])
        self.full_data_calib_high = pd.read_csv(data_paths_calib[0])
        self.full_data_calib_no = pd.read_csv(data_paths_calib[1])
        self.data_path_saving=data_path_saving

    def set_current_name(self, eval_name):
        self.eval_name = eval_name
        
    def preallocate_memory(self, batch_size_REG, batch_size_MMD):
        self.batch_size_REG = batch_size_REG
        self.batch_size_MMD = batch_size_MMD
        for cr in self.cr:
            if cr.is_MMD():
                cr.preallocate_memory(batch_size_MMD)
            else:
                cr.preallocate_memory(batch_size_REG)
        
    def set_containers(self, prof, nb_cand, nb_runs, train_epochs, test_epochs):
        self.prof = prof
        self.nb_runs = nb_runs
        self.test_epochs = test_epochs
        self.dist_valid  = th.zeros([nb_cand, nb_runs, train_epochs, self.nb_cr]) #distance values from validation
        self.dist_test = th.zeros([nb_cand, nb_runs, test_epochs, self.nb_cr]) #distance values from testing

    def sample_data(self, train_sample_size, valid_sample_size, test_sample_size, calib_sample_size, gen_sample_size):
        self.gen_sample_size = gen_sample_size
        self.dataset_train = th.Tensor(self.full_data_train.sample(n=train_sample_size).values)
        self.dataset_valid = th.Tensor(self.full_data_valid.sample(n=valid_sample_size).values)
        self.dataset_test = th.Tensor(self.full_data_test.sample(n=test_sample_size).values)
        self.dataset_calib_high = th.Tensor(self.full_data_calib_high.sample(n=calib_sample_size).values)
        self.dataset_calib_no = th.Tensor(self.full_data_calib_no.sample(n=calib_sample_size).values)
        return self.dataset_train
    
    def set_candidate(self, cand_ind, adjacency_matrix):
        self.cand_ind = cand_ind
        self.cand_model_list = [None for r in range(self.nb_runs)]
        self.current_adj_matrix = adjacency_matrix
        
    def set_run(self, run_ind):
        self.run_ind = run_ind
    
    def calibrate_distances(self, criteria_train_REG, criteria_train_MMD, batch_size_train_REG, batch_size_train_MMD):
        weights_REG = th.zeros([len(criteria_train_REG)])
        weights_MMD = th.zeros([len(criteria_train_MMD)])
        print('[Calculating reference-values]');
        print('training (REG):')
        for ind, c in enumerate(criteria_train_REG):
            print('- ' + c.get_name(True))
            snr = self._calibrate_distance(c, batch_size_train_REG)
            weights_REG[ind] = snr
        print('training (MMD):')
        for ind, c in enumerate(criteria_train_MMD):
            print('- ' + c.get_name(True))
            snr = self._calibrate_distance(c, batch_size_train_MMD)
            weights_MMD[ind] = snr
        if len(weights_REG) > 0:
            weights_REG = weights_REG/th.min(weights_REG)
        if len(weights_MMD) > 0:
            weights_MMD = weights_MMD/th.min(weights_MMD)
        #ref values for validation/test criteria
        print('validation/test:')
        cr_snr_REG = np.array([self._calibrate_distance(cr) for cr in self.cr if not cr.is_MMD()]); 
        cr_snr_MMD = np.array([self._calibrate_distance(cr) for cr in self.cr if cr.is_MMD()]); 
        self.eval_criteria_weights = list(cr_snr_REG/np.min(cr_snr_REG))+list(cr_snr_MMD/np.min(cr_snr_MMD))
        return weights_REG, weights_MMD
    
    def validate_model(self, model, epoch):
        #initialize DataLoaders
        loader_REG = DataLoader(self.dataset_valid, batch_size=self.batch_size_REG, shuffle=True, drop_last=True)
        loader_MMD = DataLoader(self.dataset_valid, batch_size=self.batch_size_MMD, shuffle=True, drop_last=True)
        #loop over validation distances
        for cr_ind, cr in enumerate(self.cr):
            if cr.is_MMD():
                val_mean,_ = self._average_model_dist_over_batches(cr, model, self.batch_size_MMD, loader_MMD, True, 'va', epoch)
            else:
                val_mean,_ = self._average_model_dist_over_batches(cr, model, self.batch_size_REG, loader_REG, True, 'va', epoch)
            self.dist_valid[self.cand_ind, self.run_ind, epoch, cr_ind] = val_mean.abs().detach()
        return self.dist_valid[self.cand_ind, self.run_ind, epoch, :].mean()
    
    def test_model(self, model):
        # save model
        self.cand_model_list[self.run_ind] = model
        #loop over test epochs
        print('[TESTING]:')
        for epoch in range(self.test_epochs):
            #initialize DataLoaders
            loader_REG = DataLoader(self.dataset_test, batch_size=self.batch_size_REG, shuffle=True, drop_last=True)
            loader_MMD = DataLoader(self.dataset_test, batch_size=self.batch_size_MMD, shuffle=True, drop_last=True)
            #loop over test distances
            for cr_ind, cr in enumerate(self.cr):
                if cr.is_MMD():
                    val_mean, _ = self._average_model_dist_over_batches(cr, model, self.batch_size_MMD, loader_MMD, True, 'te')
                else:
                    val_mean, _ = self._average_model_dist_over_batches(cr, model, self.batch_size_REG, loader_REG, True, 'te')
                self.dist_test[self.cand_ind, self.run_ind, epoch, cr_ind] = val_mean.abs().detach() #store test values

    def set_best_run(self): #omid
        #check if current run is best so far
        mean_test_loss = th.mean(self.dist_test[self.cand_ind, self.run_ind, :, :].abs())
        if self.run_ind == 0:
            self.best_loss = mean_test_loss
            self.best_run_ind = self.run_ind
            print('   mean-test-loss: {:.02f} [CURRENT BEST]\n'.format(mean_test_loss.item()))
        elif mean_test_loss < self.best_loss:
            self.best_loss = mean_test_loss
            self.best_run_ind = self.run_ind
            print('   mean-test-loss: {:.02f} [CURRENT BEST]\n'.format(mean_test_loss.item()))
        else:
            print('   mean-test-loss: {:.02f}\n'.format(mean_test_loss.item()))

    def save_synthetic(self, cand_name):
        file_id = 'c{}{}'.format(cand_name, self.eval_name)
        template_path = os.path.join(self.data_path_saving, 'xxx', file_id + '.csv')
        model = self.cand_model_list[self.best_run_ind]
        df = pd.DataFrame(model(self.gen_sample_size, False).detach().numpy())
        df.to_csv(template_path.replace('xxx', 'synthetic'), index=False) 

    def save_losses(self, cand_name):
        self.prof.start() #[PROFILING]
        file_id = 'c{}_r{}{}'.format(cand_name, self.run_ind, self.eval_name)
        template_path = os.path.join(self.data_path_saving,'xxx', file_id + '.csv')

        df = self._store_run_to_dataFrame(self.cand_ind, self.run_ind, 'train')
        df.to_csv(template_path.replace('xxx', 'losses/train'), index=False)

        df = self._store_run_to_dataFrame(self.cand_ind, self.run_ind, 'valid')
        df.to_csv(template_path.replace('xxx', 'losses/valid'), index=False) 

        df = self._store_run_to_dataFrame(self.cand_ind, self.run_ind, 'test')
        df.to_csv(template_path.replace('xxx', 'losses/test'), index=False)

        self.prof.stop('sav') #[PROFILING]

    def _calibrate_distance(self, criterion, custom_batch_size = None):
        #config
        nb_marg_batches = 10;
        #decide which batch_size to use
        if not custom_batch_size == None:
            bs = custom_batch_size
        elif criterion.is_MMD():
            bs = self.batch_size_MMD
        else:
            bs = self.batch_size_REG
        if criterion.is_Marg():
            ref_val, ref_val_std, benchmark, benchmark_std = self._average_marg_dist_over_simData(criterion,bs,nb_marg_batches)
        else:
            #initialize DataLoaders
            loader_train = DataLoader(self.dataset_train,batch_size=bs,shuffle=True,drop_last=True)
            loader_calib_high = DataLoader(self.dataset_calib_high,batch_size=bs,shuffle=True,drop_last=True)
            loader_calib_no = DataLoader(self.dataset_calib_no,batch_size=bs,shuffle=True,drop_last=True)
            #calculate reference value
            ref_val, ref_val_std = self._average_samples_dist_over_batches(criterion,loader_train,loader_calib_high,False)
            #calculate benchmark
            benchmark, benchmark_std = self._average_samples_dist_over_batches(criterion,loader_train,loader_calib_no,False)
        #print results
        snr = (benchmark.item() - ref_val.item())/benchmark_std.item()
        print('\trefVal: ' + str(np.round(ref_val.item(),decimals=self.display_calib_digits)) + ' (+-' + str(np.round(ref_val_std.item(),decimals=self.display_calib_digits)) + '), bchmrk: ' + str(np.round(benchmark.item(),decimals=self.display_calib_digits)) + ' (+-' + str(np.round(benchmark_std.item(),decimals=self.display_calib_digits)) + ')')
        print('\tSNR: ' + str(np.round(snr,decimals=self.display_calib_digits)) + ', DIFF: ' + str(np.round((benchmark.item() - ref_val.item())/ref_val.item(),decimals=self.display_calib_digits)))
        criterion.save_calib(ref_val, benchmark)
        return snr
        
    def _average_samples_dist_over_batches(self, criterion, loader1, loader2, normalize):
        dist_list = th.zeros(len(loader1))
        for ind,(d1,d2) in enumerate(zip(loader1,loader2)):
            dist_list[ind] = criterion.forward(d1,d2,normalize)
        return th.mean(dist_list), th.std(dist_list)

    def _average_model_dist_over_batches(self, criterion, model, bs, loader, normalize, eval_str, epoch = None):
        dist_list = th.zeros(len(loader))
        for ind,data in enumerate(loader):
            self.prof.start() #[PROFILING]
            gen_data = model(bs)
            self.prof.stop('gen_' + eval_str,epoch) #[PROFILING]
            self.prof.start() #[PROFILING]
            dist_list[ind] = criterion(gen_data,data,normalize)
            self.prof.stop('ev_' + eval_str,epoch) #[PROFILING]
        return th.mean(dist_list), dist_list.detach().tolist()
    
    def _average_marg_dist_over_simData(self, criterion, batch_size, nb_marg_batches):
        list_ref_vals_n = th.zeros(nb_marg_batches,1)
        list_ref_vals_u = th.zeros(nb_marg_batches,1)
        list_benchmarks_n = th.zeros(nb_marg_batches,1)
        list_benchmarks_u = th.zeros(nb_marg_batches,1)
        for k in range(nb_marg_batches):
            mdat_prop,mdat_prop2,mdat_comp_n,mdat_comp_u = create_marginal_test_data(batch_size)
            #ref-vals
            list_ref_vals_n[k] = criterion.forward(mdat_prop[0],mdat_prop2[0])
            list_ref_vals_u[k] = criterion.forward(mdat_prop[1],mdat_prop2[1])
            subList_BM_n = th.zeros(2*len(mdat_comp_n),1)
            for j in range(len(mdat_comp_n)):
                subList_BM_n[2*j] = criterion.forward(mdat_prop[0],mdat_comp_n[j])
                subList_BM_n[2*j+1] = criterion.forward(mdat_prop[1],mdat_comp_n[j])
            subList_BM_u = th.zeros(2*len(mdat_comp_u),1)
            for j in range(len(mdat_comp_u)):
                subList_BM_u[2*j] = criterion.forward(mdat_prop[0], mdat_comp_u[j])
                subList_BM_u[2*j+1] = criterion.forward(mdat_prop[1], mdat_comp_u[j])
            list_benchmarks_n[k] = th.mean(subList_BM_n)
            list_benchmarks_u[k] = th.mean(subList_BM_u)
        list_ref_vals = th.concat((list_ref_vals_n, list_ref_vals_u))
        list_benchmarks = th.concat((list_benchmarks_n, list_benchmarks_u))
        return th.mean(list_ref_vals), th.std(list_ref_vals), th.mean(list_benchmarks), th.std(list_benchmarks)

    def _store_run_to_dataFrame(self, cand_ind, run_ind, ref_set):
        df = pd.DataFrame()
        if ref_set == 'train':
            df['loss'] = self.dist_train
        elif ref_set == 'valid':
            max_valid_epoch = uFun.find_MaxEpochs(self.dist_valid[cand_ind, run_ind, :, :].numpy())
            for cr_ind, cr in enumerate(self.cr):
                df[cr.get_name(True)] = self.dist_valid[cand_ind, run_ind, :max_valid_epoch, cr_ind]
        else:
            for cr_ind, cr in enumerate(self.cr):
                df[cr.get_name(True)] = self.dist_test[cand_ind, run_ind, :, cr_ind]
        return df


def create_marginal_test_data(gen_size):
    # proper distributions
    mdat_n = th.zeros(gen_size,1).normal_(0, 1)
    mdat_u = th.zeros(gen_size,1).uniform_(0, 1)
    mdat_prop = [mdat_n,mdat_u]
    for ind,d in enumerate(mdat_prop):
        mdat_prop[ind] = d.repeat(1,4)  # create the same distribution for all variables
    #second set of proper distributions
    mdat_n2 = th.zeros(gen_size,1).normal_(0, 1)
    mdat_u2 = th.zeros(gen_size,1).uniform_(0, 1)
    mdat_prop2 = [mdat_n2,mdat_u2]
    for ind,d in enumerate(mdat_prop2):
        mdat_prop2[ind] = d.repeat(1,4)  # create the same distribution for all variables
    # distributions for comparison
    mdat_nP = th.abs(th.zeros(gen_size,1).normal_(0,1))  # only positive half
    mdat_nS = th.zeros(gen_size,1).normal_(2.5,1)  # shifted
    mdat_nN = th.zeros(gen_size,1).normal_(0,0.25)  # narrow
    mdat_nNS = th.zeros(gen_size,1).normal_(1.5,0.5)  # narrow & shifted
    mdat_comp_n = [mdat_nP,mdat_nS,mdat_nN,mdat_nNS]
    for ind,d in enumerate(mdat_comp_n):
        mdat_comp_n[ind] = d.repeat(1,4) # create the same distribution for all variables
    mdat_uT = ( th.zeros(gen_size,1).uniform_(0, 1) )**(1/6) # tilted
    mdat_uS = th.zeros(gen_size,1).uniform_(1-1/np.sqrt(2),2-1/np.sqrt(2)) # shifted
    mdat_uN = th.zeros(gen_size,1).uniform_(0.5-0.25,0.5+0.25) #narrow
    mdat_uNS = th.zeros(gen_size,1).uniform_(0.5-0.78/2+0.27,0.5+0.78/2+0.27) # narrow & shifted
    mdat_comp_u = [mdat_uT,mdat_uS,mdat_uN,mdat_uNS]
    for ind,d in enumerate(mdat_comp_u):
        mdat_comp_u[ind] = d.repeat(1,4) #create the same distribution for all variables
    return mdat_prop, mdat_prop2, mdat_comp_n, mdat_comp_u




def get_cr_weights(criteria, cw_dir):
    cfg = uFun.load_json(os.path.join(cw_dir, 'Data/cfg.json'))
    data_paths = [
        os.path.join(cw_dir, 'Data/datasets/dat_train.csv'),
        os.path.join(cw_dir, 'Data/datasets/dat_val.csv'),
        os.path.join(cw_dir, 'Data/datasets/dat_test.csv'),
        ]

    data_paths_calib = [
        os.path.join(cw_dir, 'Data/datasets/dat_calib_high.csv'),
        os.path.join(cw_dir, 'Data/datasets/dat_calib_noCorr.csv')
    ]
    cr_REG, cr_MMD = [], []
    for c in criteria:
        if c.is_MMD():
            c.preallocate_memory(cfg['batch_size_MMD'])
            cr_MMD.append(c)
        else:
            c.preallocate_memory(cfg['batch_size_REG'])
            cr_REG.append(c)
    ev = Evaluator(data_paths, data_paths_calib, None, False)
    ev.preallocate_memory(cfg['batch_size_REG'], cfg['batch_size_MMD'])
    ev.sample_data(cfg['train_sample_size'], cfg['valid_sample_size'], cfg['test_sample_size'], cfg['calib_sample_size'], cfg['gen_sample_size'])
    weights_REG, weights_MMD = ev.calibrate_distances(cr_REG, cr_MMD, cfg['batch_size_REG'], cfg['batch_size_MMD'])
    weights = [w.item() for w in weights_MMD]+[w.item() for w in weights_REG]
    names = [cr.get_name(True) for cr in cr_MMD]+[cr.get_name(True) for cr in cr_REG]
    return dict(zip(names, weights))