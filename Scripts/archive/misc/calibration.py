
import os
import pandas as pd
import torch as th
from torch.utils.data import DataLoader
import sys
sys.path.insert(1, '..')
from evaluator import Evaluator
import utilities_functions as uFun

class Calibrator:
    def __init__(self, data_path_train, data_paths_calib, train_sample_size, calib_sample_size, batch_size_MMD,batch_size_REG):
        self.batch_size_MMD = batch_size_MMD
        self.batch_size_REG = batch_size_REG
        self._load_data(data_path_train, data_paths_calib)
        self._sample_data(train_sample_size, calib_sample_size)
    
    def forward(self, criteria_list):
        calib_list = []
        nb_marg_batches = 10
        for criterion in criteria_list:
            if criterion.is_MMD():
                bs = self.batch_size_MMD
            else:
                bs = self.batch_size_REG
            if criterion.is_Marg():
                ref_val, ref_val_std, benchmark, benchmark_std = self._average_marg_dist_over_simData(criterion, bs, nb_marg_batches)
            else:
                loader_train      = DataLoader(self.dataset_train,      batch_size=bs, shuffle=True, drop_last=True)
                loader_calib_high = DataLoader(self.dataset_calib_high, batch_size=bs, shuffle=True, drop_last=True)
                loader_calib_no   = DataLoader(self.dataset_calib_no,   batch_size=bs, shuffle=True, drop_last=True)
                ref_val, ref_val_std     = self._average_samples_dist_over_batches(criterion, loader_train, loader_calib_high, False)
                benchmark, benchmark_std = self._average_samples_dist_over_batches(criterion, loader_train, loader_calib_no,   False)
            snr = (benchmark.item() - ref_val.item())/benchmark_std.item()
            criterion.save_calib(ref_val, benchmark)
            calib_list.append(criterion)
        return calib_list
 
    def _average_marg_dist_over_simData(self, criterion, batch_size, nb_marg_batches):
        list_ref_vals_n = th.zeros(nb_marg_batches,1)
        list_ref_vals_u = th.zeros(nb_marg_batches,1)
        list_benchmarks_n = th.zeros(nb_marg_batches,1)
        list_benchmarks_u = th.zeros(nb_marg_batches,1)
        for k in range(nb_marg_batches):
            mdat_prop,mdat_prop2,mdat_comp_n,mdat_comp_u = uFun.create_marginal_test_data(batch_size)
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
    
    def _average_samples_dist_over_batches(self, criterion, loader1, loader2, normalize):
        dist_list = th.zeros(len(loader1))
        for ind,(d1,d2) in enumerate(zip(loader1,loader2)):
            dist_list[ind] = criterion.forward(d1,d2,normalize)
        return th.mean(dist_list), th.std(dist_list)

    def _load_data(self, data_path_train, data_paths_calib):
        self.full_data_train = pd.read_csv(data_path_train)
        self.full_data_calib_high = pd.read_csv(data_paths_calib[0])
        self.full_data_calib_no = pd.read_csv(data_paths_calib[1])
        
    def _sample_data(self, train_sample_size, calib_sample_size):
        self.dataset_train = th.Tensor(self.full_data_train.sample(n=train_sample_size).values)
        self.dataset_calib_high = th.Tensor(self.full_data_calib_high.sample(n=calib_sample_size).values)
        self.dataset_calib_no = th.Tensor(self.full_data_calib_no.sample(n=calib_sample_size).values)
