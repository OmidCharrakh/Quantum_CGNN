#script to load various training results and plot them for comparison
import pandas as pd
import math
# import distance_functions as distFun
# import torch as th
# from torch.utils.data import DataLoader
import numpy as np
# import time
import utilities_functions as uFun
import matplotlib.pyplot as plt
# from tqdm import tqdm #library for showing progress bars
# from copy import deepcopy

#CONFIG
n_runs = 4
n_tEpochs = 20
cand_id = 0
id_string = 'comb_allN_ep200_r4'
dist_list = ['MMD_cdt','MMD_Fr','CorrD','BinnedD','CorrD_N','NpMom','CndD','EDF_Mrg']
#data_list = ['comb1_eq','comb1_snr_basic_mixedN_100','comb2_eq','comb2_snr_basic_mixedN_100','comb3_eq','comb3_snr_basic_mixedN_100','comb4_eq','comb4_snr_basic_mixedN_100']
#data_list = ['comb1p_eq_basic_mixedN_100','comb1p_snr_basic_mixedN_100','comb2p_eq_basic_mixedN_100','comb2p_snr_basic_mixedN_100','comb3p_eq_basic_mixedN_100','comb3p_snr_basic_mixedN_100','comb4p_eq_basic_mixedN_100','comb4p_snr_basic_mixedN_100']
#data_list = ['comb2_eq','comb2_snr_basic_mixedN_100','comb3_eq','comb3_snr_basic_mixedN_100','comb4_eq','comb4_snr_basic_mixedN_100','comb1p_eq_basic_mixedN_100','comb1p_snr_basic_mixedN_100','comb2p_eq_basic_mixedN_100','comb2p_snr_basic_mixedN_100','comb3p_eq_basic_mixedN_100','comb3p_snr_basic_mixedN_100','comb4p_eq_basic_mixedN_100','comb4p_snr_basic_mixedN_100']
#data_list = ['comb2_eq','comb2_snr_basic_mixedN_100','comb3_eq','comb3_snr_basic_mixedN_100','comb4_snr_basic_mixedN_100','comb2p_eq_basic_mixedN_100','comb2p_snr_basic_mixedN_100','comb3p_eq_basic_mixedN_100','comb3p_snr_basic_mixedN_100']
#data_list = ['comb3_eq','comb3_snr_basic_mixedN_100','comb3p_eq_basic_mixedN_100','comb3p_snr_basic_mixedN_100']
#data_list = ['comb3_eq','comb3_snr_basic_mixedN_100','comb3_eq_basic_gaussN_100','comb3_snr_basic_gaussN_100']
#data_list = ['comb3_eq_basic_mixedN_20','comb3_eq_basic_mixedN_50','comb3_eq_basic_mixedN_100','comb3_eq_basic_mixedN_150','comb3_eq_basic_mixedN_200']
#data_list = ['comb3_snr_basic_mixedN_20','comb3_snr_basic_mixedN_50','comb3_snr_basic_mixedN_100','comb3_snr_basic_mixedN_150','comb3_snr_basic_mixedN_200']
#data_list = ['comb3_eq_basic_mixedN_20','comb3_eq_basic_mixedN_50','comb3_eq_basic_mixedN_100','comb3_snr_basic_mixedN_20','comb3_snr_basic_mixedN_50','comb3_snr_basic_mixedN_100']
#data_list = ['comb3_snr_basic_mixedN_50','comb3_snr_0wg_1l_mixedN_50_newArc']
#data_list = ['comb3_snr_basic_mixedN_100','comb3_snr_0wg_1l_mixedN_100_newArc']
#data_list = ['comb3_snr_0wg_1l_mixedN_50_newArc','comb3_snr_0wg_2l_mixedN_50_newArc','comb3_snr_0.5wg_1l_mixedN_50_newArc','comb3_snr_0.5wg_2l_mixedN_50_newArc','comb3_snr_1wg_1l_mixedN_50_newArc','comb3_snr_1wg_2l_mixedN_50_newArc']
#data_list = ['comb3_snr_basic_mixedN_100','comb3_snr_0wg_2l_mixedN_100_newArc','comb3_snr_0.5wg_1l_mixedN_100_newArc','comb3_snr_0.5wg_2l_mixedN_100_newArc','comb3_snr_1wg_1l_mixedN_100_newArc','comb3_snr_1wg_2l_mixedN_100_newArc']
data_list = ['comb3_snr_1wg_2l_mixedN_50_newArc','comb3_snr_0.5wg_2l_mixedN_100_newArc','comb3_snr_1wg_2l_mixedN_100_newArc'];


def load_data(data_str):
    cur_data = [None for k in range(n_runs)]
    for r in range(n_runs):
        file_id = 'c{}_r{}'.format(str(cand_id).zfill(4),str(r).zfill(2))
        cur_data[r] = pd.read_csv('Results/progress/'+ data_str + '_' + file_id + '.csv')
    return cur_data

def plot_data(dist_data,dist_str,data_list,ax):
    n_sets = len(data_list) #training mode
    n_val = len(dist_str) #distance
    data_min = np.zeros([n_sets,n_val])
    data_mean = np.zeros([n_sets,n_val])
    data_std = np.zeros([n_sets,n_val])
    #fill data
    for val_ind,d_str in enumerate(dist_str):
        for set_ind in range(n_sets):
            coll_run_data = np.zeros([n_tEpochs,n_runs])
            for run_ind in range(n_runs):
                curFrame = dist_data[set_ind][run_ind]
                coll_run_data[:,run_ind] = curFrame[d_str].to_numpy()
            data_min[set_ind,val_ind] = np.min(coll_run_data)
            data_mean[set_ind,val_ind] = np.mean(coll_run_data)
            data_std[set_ind,val_ind] = np.std(coll_run_data)
    uFun.customBarPlot(ax, data_min, data_list, dist_str, errorBars = data_std, errorAnchorPoints = data_mean, logPlot = True, enforceZero = False, norm_mode = 'minAnch')
    return data_mean

#SCRIPT
#load data
num_dist = len(dist_list)
dist_data = [None for k in range(len(data_list))]
for ind,data_str in enumerate(data_list):
    dist_data[ind] = load_data(data_str)

#plot comparison of performance
num_sub_fig = 2
dist_sum_ind = [2,3,4,6,7]
data_mean_coll = np.zeros([len(data_list),0])
for rep in range(math.ceil(num_dist/num_sub_fig)):
    #open new figure
    fig,ax = plt.subplots()
    #choose dist_str
    if len(dist_list) > num_sub_fig:
        dist_str = dist_list[0:num_sub_fig]
        dist_list = dist_list[num_sub_fig:]
    else:
        dist_str = dist_list
    #plot data
    data_mean_cur = plot_data(dist_data,dist_str,data_list,ax)
    plt.savefig('./Analysis/' + id_string + '_' + str(rep) + '.png',dpi=150, bbox_inches='tight')
    #add data to total data
    data_mean_coll = np.concatenate((data_mean_coll,data_mean_cur),axis=1)
#pick quantities relevant for summing and sum them
data_sums = np.transpose(np.atleast_2d(np.sum(data_mean_coll[:,dist_sum_ind],axis=1)))
#plot sums
fig,ax = plt.subplots()
uFun.customBarPlot(ax, data_sums, data_list, ['sum'], logPlot = False)