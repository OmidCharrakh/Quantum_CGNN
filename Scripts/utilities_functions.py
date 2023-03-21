import numpy as np
import torch as th
from torch.utils.data import DataLoader
import pandas as pd
import networkx as nx
import itertools
import json
import pickle
import os
from copy import deepcopy
from scipy.stats import zscore
from fcit import fcit


def flatten_list(x):
    depth = find_depth_list(x)
    for i in range(depth - 1):
        x = list(itertools.chain(*x))
    return x


def find_depth_list(x):
    if isinstance(x, list):
        return 1 + max(find_depth_list(i) for i in x)
    else:
        return 0


def save_as_pickle(file, filepath):

    with open(filepath, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def object_saver(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def object_loader(path):
    with open(path, 'rb') as file:
        loaded_obj = pickle.load(file)
        return loaded_obj


def save_as_json(dictionary, target):
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def stable_argsort(arr, dim=-1, descending=False):
    arr_np = arr.detach().cpu().numpy()
    if descending:
        indices = np.argsort(-arr_np, axis=dim, kind='stable')
    else:
        indices = np.argsort(arr_np, axis=dim, kind='stable')
    return th.from_numpy(indices).long().to(arr.device)


def stable_sort(arr, dim=-1, descending=False):
    arr_np = arr.detach().cpu().numpy()
    if descending:
        indices = np.sort(-arr_np, axis=dim, kind='stable')
    else:
        indices = np.sort(arr_np, axis=dim, kind='stable')
    return th.from_numpy(indices).as_type(arr)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def count_connections(A):
    G = nx.DiGraph(A)
    connections=[]
    for i in range(4):
        for j in range(1+i,4):
            connections.append(not nx.d_separated(G, {i}, {j}, {}))
    return np.sum(connections)


def get_mechanism_importance(A, node_index):
    A0, A1 = A.copy(), A.copy()
    A0[:,node_index]=0
    c1, c0 = count_connections(A1), count_connections(A0)
    return (c1-c0)/6


def count_hu(A, nb_base_hu, wg_conn, nb_layers):
    nb_vars = A.shape[0]
    hu_matrix = np.zeros((nb_vars, nb_layers))
    for n in range(nb_vars):
        mechanism_importance = get_mechanism_importance(A,n)
        n_incommings = A[:,n].sum()
        nb_hu = np.round(nb_base_hu*(1+wg_conn*n_incommings*mechanism_importance))
        nb_layers_hu = np.ones([nb_layers],int) * np.floor(nb_hu/nb_layers).astype(int)

        index_layer = 0
        while np.sum(nb_layers_hu) < nb_hu:
            nb_layers_hu[index_layer] += 1
            index_layer += 1
        hu_matrix[n] = nb_layers_hu
    hu_matrix=hu_matrix.astype(int)
    return hu_matrix


def prune_adjacency(A):
    connected_nodes = []
    for i in range(A.shape[0]):
        n_incomming = A[:,i].sum()
        n_outgoing = A[i,:].sum()
        if n_incomming or n_outgoing:
            connected_nodes.append(i)
    if np.sum(A) == 0:
        return np.zeros((4, 4)).astype(int)
    else:
        return A[connected_nodes, :][:, connected_nodes]


def pad_adjacency(A, target_nodes):
    return np.pad(A, [(0, target_nodes-A.shape[0])], constant_values=0)


def cal_unique_metric(df_prog, cr_weights):
    cr_means = df_prog.mean(0).abs()
    cr_weights = pd.Series(cr_weights).abs()
    res = cr_means * cr_weights
    return res.sum()/cr_weights[res.notna()].sum()


def reduce_df(df, reduction='min'):
    uniqune_ns = df['nb_base'].unique()
    df_new = pd.DataFrame()
    df_new['nb_base'] = uniqune_ns
    df_new['perf'] = None
    df_new['run_index'] = None
    if reduction == 'max':
        for index, row in df_new.iterrows():
            df_new.loc[index, 'perf'] = df[df['nb_base']==row['nb_base']]['perf'].max()
            df_new.loc[index, 'run_index'] = df[df['nb_base']==row['nb_base']]['perf'].argmax()
    elif reduction == 'min':
        for index, row in df_new.iterrows():
            df_new.loc[index, 'perf'] = df[df['nb_base']==row['nb_base']]['perf'].min()
            df_new.loc[index, 'run_index'] = df[df['nb_base']==row['nb_base']]['perf'].argmin()
    elif reduction == 'mean':
        for index, row in df_new.iterrows():
            df_new.loc[index, 'perf'] = df[df['nb_base']==row['nb_base']]['perf'].mean()
    elif reduction == 'std':
        for index, row in df_new.iterrows():
            df_new.loc[index, 'perf'] = df[df['nb_base']==row['nb_base']]['perf'].std()
    df_new['perf'] = df_new['perf'].astype(float)
    return df_new


class Measure_Distance():
    def __init__(self, criterion, n_epochs=20):
        if criterion.is_MMD():
            self.batch_size = 500
        else:
            self.batch_size = 8000
        self.criterion = criterion
        self.criterion.preallocate_memory(self.batch_size)
        self.n_epochs = n_epochs

    def forward(self, dataset_1, dataset_2):
        loss_list = []
        dataset_1 = th.Tensor(dataset_1)
        dataset_2 = th.Tensor(dataset_2)
        for _ in range(self.n_epochs):
            loader_1 = DataLoader(dataset_1, batch_size=self.batch_size, shuffle=True, drop_last=True)
            loader_2 = DataLoader(dataset_2, batch_size=self.batch_size, shuffle=True, drop_last=True)
            for (data_1, data_2) in zip(loader_1, loader_2):
                loss = self.criterion(data_1, data_2, False)
                loss_list.append(loss)
        return th.tensor(loss_list).detach().mean().abs().item()


class CorrPatterns:
    def __init__(self, sampling_rate=3):
        super().__init__()
        self.sampling_rate = sampling_rate
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        #create lists of variable combinations
        self.var_list_2 = [np.array(vS) for vS in list(itertools.combinations([0,1,2,3],2))]
        self.var_cnd_list_2 = [np.setdiff1d([0,1,2,3],l) for l in self.var_list_2]
        self.var_list_3 = [np.array(vS) for vS in list(itertools.combinations([0,1,2,3],3))]
        self.var_cnd_list_3 = [np.setdiff1d([0,1,2,3],l) for l in self.var_list_3]
        #calculate bin edges
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = self._calc_edges(
            self.sampling_rate, self.range_obs, self.range_set)
        
        self.edgeDict = {
            0: np.sort(list(set(self.edg_obs_low).union(set(self.edg_obs_upp)))),
            1: np.sort(list(set(self.edg_obs_low).union(set(self.edg_obs_upp)))),
            2: np.sort(list(set(self.edg_set_low).union(set(self.edg_set_upp)))), 
            3: np.sort(list(set(self.edg_set_low).union(set(self.edg_set_upp)))), 
        }

    def get_patterns(self, dstr, corr_type='(2,2)'):
        dstr = th.tensor(dstr)
        edgeDict = self.edgeDict
        #segregate all data into bins
        binned_data_inds = self._bin_all_marginals(
            dstr, self.sampling_rate, self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp)
        if corr_type == '(2,2)':
            corrMats, sampMats = self._detCorrs2(dstr, binned_data_inds)
            corr_dict = {
                '01_23' : {'corr': corrMats[0], 'samp': sampMats[0], 'cv0': edgeDict[2], 'cv1': edgeDict[3]}, 
                '02_13' : {'corr': corrMats[1], 'samp': sampMats[1], 'cv0': edgeDict[1], 'cv1': edgeDict[3]}, 
                '03_12' : {'corr': corrMats[2], 'samp': sampMats[2], 'cv0': edgeDict[1], 'cv1': edgeDict[2]}, 
                '12_03' : {'corr': corrMats[3], 'samp': sampMats[3], 'cv0': edgeDict[0], 'cv1': edgeDict[3]}, 
                '13_02' : {'corr': corrMats[4], 'samp': sampMats[4], 'cv0': edgeDict[0], 'cv1': edgeDict[2]}, 
                '23_01' : {'corr': corrMats[5], 'samp': sampMats[5], 'cv0': edgeDict[0], 'cv1': edgeDict[1]},
            }
        elif corr_type == '(1,3)':
            corrVecs, sampVecs = self._detCorrs3(dstr, binned_data_inds)
            corr_dict = {
                '0_123' : {'corr': corrVecs[3], 'samp': sampVecs[3], 'cv0': edgeDict[1], 'cv1': edgeDict[2], 'cv2': edgeDict[3]},
                '1_023' : {'corr': corrVecs[2], 'samp': sampVecs[2], 'cv0': edgeDict[0], 'cv1': edgeDict[2], 'cv2': edgeDict[3]},
                '2_013' : {'corr': corrVecs[1], 'samp': sampVecs[1], 'cv0': edgeDict[0], 'cv1': edgeDict[1], 'cv2': edgeDict[3]},
                '3_012' : {'corr': corrVecs[0], 'samp': sampVecs[0], 'cv0': edgeDict[0], 'cv1': edgeDict[1], 'cv2': edgeDict[2]},
            }
        else:
            corr_dict = None
        return corr_dict

    
    def _detCorrs2(self, d, binned_data_inds):
        corrMats = th.zeros([6,self.sampling_rate,self.sampling_rate]) #we have 6 possible pairs of observables
        corrSamp = th.zeros([6,self.sampling_rate,self.sampling_rate]) #we have 6 possible pairs of observables
        for ind,(vs,vCs) in enumerate(zip(self.var_list_2,self.var_cnd_list_2)):
            corrMats[ind,:,:], corrSamp[ind,:,:] = self._calc_corr_mat(d, vs, vCs, binned_data_inds)
        return corrMats.numpy(), corrSamp.numpy()
    
    def _detCorrs3(self, d, binned_data_inds):
        corrVecs = th.zeros([4,self.sampling_rate]) #we have 4 possible triples of observables
        corrSamp = th.zeros([4,self.sampling_rate]) #we have 4 possible triples of observables
        for ind,(vs,vCs) in enumerate(zip(self.var_list_3,self.var_cnd_list_3)):
            corrVecs[ind,:],corrSamp[ind,:] = self._calc_corr_vec(d,vs,vCs,binned_data_inds)
        return corrVecs.numpy(), corrSamp.numpy()

    def _calc_corr_mat(self, d, vs, vCs, binned_data_inds):
        corrMat = th.zeros([self.sampling_rate,self.sampling_rate])
        sampMat = th.zeros([self.sampling_rate,self.sampling_rate])
        for i1 in range(self.sampling_rate):
            for i2 in range(self.sampling_rate):
                #merge indices
                cond_inds = th.logical_and(binned_data_inds[vCs[0],i1,:],binned_data_inds[vCs[1],i2,:])
                num_samp = th.count_nonzero(cond_inds)
                sampMat[i1,i2] = num_samp
                if num_samp == 0:
                    #no correlation value can be calculated here, default to 0
                    corrMat[i1,i2] = 0
                    continue
                #crop to conditioned two party distribution
                dC = d[:,vs]
                dC = dC[cond_inds,:]
                #calculate correlation value
                corrMat[i1,i2] = self._calc_full_corr(dC)
        return corrMat, sampMat

    def _calc_full_corr(self, d):
        #calculate marginal standard deviations
        stdProd = th.nan_to_num(th.prod(th.std(d,dim=0)))
        if stdProd == 0:
            return 0 #at least one of the variables does not change, they cannot be correlated
        #calculate correlation
        return th.mean(th.prod(d-th.mean(d,dim=0),dim=1)) / stdProd
    
    def _calc_edges(self, sampling_rate, range_obs, range_set):
        fac_low = np.linspace(0,sampling_rate-1,sampling_rate)
        fac_upp = np.linspace(1,sampling_rate, sampling_rate)
        step_obs = np.diff(range_obs) / sampling_rate
        step_set = np.diff(range_set) / sampling_rate
        edg_obs_low = fac_low*step_obs + range_obs[0]
        edg_obs_upp = fac_upp*step_obs + range_obs[0]
        edg_set_low = fac_low*step_set + range_set[0]
        edg_set_upp = fac_upp*step_set + range_set[0]
        return edg_obs_low, edg_obs_upp, edg_set_low, edg_set_upp
    
    def _bin_all_marginals(self, d, sampling_rate, edg_obs_low, edg_obs_upp, edg_set_low, edg_set_upp):
        binned_data_inds = th.zeros([4,sampling_rate,d.shape[0]],dtype=th.bool)
        binned_data_inds[0,:,:] = self._bin_marginal(d[:,0], sampling_rate, edg_obs_low, edg_obs_upp)
        binned_data_inds[1,:,:] = self._bin_marginal(d[:,1], sampling_rate, edg_obs_low, edg_obs_upp)
        binned_data_inds[2,:,:] = self._bin_marginal(d[:,2], sampling_rate, edg_set_low, edg_set_upp)
        binned_data_inds[3,:,:] = self._bin_marginal(d[:,3], sampling_rate, edg_set_low, edg_set_upp)
        return binned_data_inds
    
    def _bin_marginal(self, marg,sampling_rate,edges_low,edges_upp):
        inds_bool = th.zeros([sampling_rate,marg.shape[0]],dtype=th.bool)
        for ind,(eL,eU) in enumerate(zip(edges_low,edges_upp)):
            inds_bool[ind] = th.logical_and(marg > eL,marg <= eU)
        return inds_bool
    
    def _calc_corr_vec(self, d, vs, vC, binned_data_inds):
        corrVec = th.zeros([self.sampling_rate])
        sampVec = th.zeros([self.sampling_rate])
        for ind in range(self.sampling_rate):
            #merge indices
            cond_inds = binned_data_inds[vC[0],ind,:]
            num_samp = th.count_nonzero(cond_inds)
            sampVec[ind] = num_samp
            if num_samp == 0:
                #no correlation value can be calculated here, default to 0
                corrVec[ind] = 0
                continue
            #crop to conditioned three party distribution
            dC = d[:,vs]
            dC = dC[cond_inds,:]
            #calculate correlation value
            corrVec[ind] = self._calc_full_corr(dC)
        return corrVec, sampVec

def categorize_loss(current_loss, all_losses, n_bins):
    bins = np.histogram(all_losses, bins=n_bins)[1]
    bins_range = [(bins[i], bins[i+1]) for i in range(n_bins)]
    bools = [(current_loss>=b[0]) and (current_loss<=b[1]) for b in bins_range]
    return bools.index(True)


def container_to_df(container):
    df = pd.DataFrame()
    df['cand_name'] = [key for key in container.keys()]
    for column in list(container.values())[0].keys():
        df[column] = [container[key][column] for key in container.keys()]
    return df


def df_to_container(df):
    container = {cand_name: {} for cand_name in list(df.cand_name)}
    for index, row in df.iterrows():
        container[row['cand_name']] = row.to_dict()
    return container


def prepare_results(results_dir, data_dir, cr_names, weight_cr=True, reduction='mean', handle_outliers=True):
    # COFIG
    n_bins = 10
    n_runs = 4
    last_epochs = 30
    outliers_thresholds = (0, 20)
    outliers_replacement = 'max'
    wg_dict_all = {'MMD_cdt': 1.0, 'MMD_Fr': 4.90, 'CorrD': 23.75, 'BinnedD': 10.89, 'CorrD_N': 2.15, 'NpMom': 1.03, 'CndD': 6.85, 'EDF_Mrg': 4.28}

    if weight_cr:
        cr_weights = {k: v for k, v in wg_dict_all.items() if k in cr_names}
    else:
        cr_weights = {k: 1 for k, v in wg_dict_all.items() if k in cr_names}

    descriptions = [
        'all disconnected',
        'Näger unsuccessful', 
        'Näger unsuccessful', 'Näger unsuccessful', 'None', 'None', 'None', 'None',
        'WS superluminal', 'None', 'None', 'None', 'None', 'None',
        'None', 'None', 
        'None',
        'Evans retrocausal; Fribe nonseparable indirect', 'leifer zigzag', 'Fribe superluminal indirect; Näger internal cancelling', 
        'Näger indirect interactive; Näger Internal cancelling', 'Näger conspiratorial', 'one affect all', 'Omid entanglement intuition 1', 'Bell local model', 'one affect settings', 
        'WS retrocausal', 'WS superdeterministic', 'WS superluminal', 'WS superluminal', 'LP no-retrocausality lambda-mediation', 'Fribe retrocausal', 'Fribe superluminal direct', 'Fribe superluminal nonlocal; Näger masking', 'Omid entanglement intuition-2', 
        'WS & Evans & Fribe: retrocausal; Symmetric version of Näger internal canceling; Symmetric version of Omid Entanglement intuition-1; separate CC system',
        'WS superdeterminism', 'WS superluminal', 'Näger cancelling', 'Näger cancelling', 'Symmetric version of Nager masking; Symmetric version of WS superluminal', 
        'LP Satisfies No-Retro', 'Symmetric version of WS superluminal', 
        'Symmetric version of Näger canceling',
        'WS Retrocausal', 'WS superdeterminism', 'Fribe non-separable & direct', 'Fribe non-separable & non-local', 'Näger masking', 'None', 'None', 
        'Gross MI relaxation; Fribe Violation of intervention', 'Gross general communication from Alice to Bob where M is a classical bit', 
        'Symmetric version of Näger masking',
        'Symmetric version of WS retrocausal', 'Symmetric version of WS superdeterministic',
        'cc3'
    ]
    descriptions.append('qcm')
    descriptions.append('icm')
    descriptions.append('scc')

    cand_names = [f'c{i}' for i in range(57)]
    cand_names.append('c500')  # qcm
    cand_names.append('c600')  # icm
    cand_names.append('c700')  # scc

    adjacencies = [prune_adjacency(A.reshape(8,8)) for A in np.loadtxt(os.path.join(data_dir, 'candidates/ccm.txt')).astype(int)]
    adjacencies.append(np.loadtxt(os.path.join(data_dir, 'candidates/qcm.txt')).astype(int).reshape(5,5))
    adjacencies.append(np.loadtxt(os.path.join(data_dir, 'candidates/icm.txt')).astype(int).reshape(4,4))
    adjacencies.append(np.loadtxt(os.path.join(data_dir, 'candidates/scc.txt')).astype(int).reshape(8,8))

    experiments = ['ccm'] * 57 + ['qcm', 'icm', 'scc']
    container_dict = {cand_name: {} for cand_name in cand_names}

    for i, cand_prop in enumerate(container_dict.values()):
        exp = experiments[i]
        if exp == 'ccm':
            cand_id = i
        else:
            cand_id = 0

        cand_prop['experiment'] = exp
        cand_prop['description'] = descriptions[i]
        cand_prop['adjacency'] = adjacencies[i]
        cand_prop['n_nodes'] = adjacencies[i].shape[0]
        cand_prop['n_edges'] = adjacencies[i].sum()
        loss_tr_all, loss_va_all, loss_te_all, n_epochs_all = [], [], [], []
    
        for run_id in range(n_runs):
            df_tr = pd.read_csv(os.path.join(results_dir, f'{exp}/losses/train/c{cand_id}_r{run_id}.csv'))
            loss_tr = df_tr.iloc[-last_epochs:,:].mean(0).sum()
            loss_tr_all.append(round(loss_tr, 3))
            df_va = pd.read_csv(os.path.join(results_dir, f'{exp}/losses/valid/c{cand_id}_r{run_id}.csv'))
            loss_va = weight_loss(df_va.iloc[-last_epochs:,:], cr_weights, cr_names)
            loss_va_all.append(round(loss_va, 3))
            df_te = pd.read_csv(os.path.join(results_dir, f'{exp}/losses/test/c{cand_id}_r{run_id}.csv'))
            loss_te = weight_loss(df_te, cr_weights, cr_names)
            loss_te_all.append(round(loss_te, 3))
            n_epochs_all.append(len(df_tr))
        if handle_outliers:
            loss_tr_all = handle_outlier(loss_tr_all, outliers_thresholds, outliers_replacement)
            loss_va_all = handle_outlier(loss_va_all, outliers_thresholds, outliers_replacement)
            loss_te_all = handle_outlier(loss_te_all, outliers_thresholds, outliers_replacement)
    
        cand_prop['loss_tr_all'] = loss_tr_all
        cand_prop['loss_va_all'] = loss_va_all
        cand_prop['loss_te_all'] = loss_te_all
        cand_prop['n_epochs_all'] = n_epochs_all
        cand_prop['best_index'] = best_index = np.argmin(cand_prop['loss_te_all'])
        
        if reduction == 'mean':
            cand_prop['loss_tr'] = np.mean(cand_prop['loss_tr_all'])
            cand_prop['loss_va'] = np.mean(cand_prop['loss_va_all'])
            cand_prop['loss_te'] = np.mean(cand_prop['loss_te_all'])
            
        if reduction == 'min':
            cand_prop['loss_tr'] = np.min(cand_prop['loss_tr_all'])
            cand_prop['loss_va'] = np.min(cand_prop['loss_va_all'])
            cand_prop['loss_te'] = np.min(cand_prop['loss_te_all'])

        cand_prop['loss_tr_std'] = np.std(cand_prop['loss_tr_all'])
        cand_prop['loss_va_std'] = np.std(cand_prop['loss_va_all'])
        cand_prop['loss_te_std'] = np.std(cand_prop['loss_te_all'])

        cand_prop['loss_tr_link'] = os.path.join(results_dir, f'{exp}/losses/train/c{cand_id}_r{best_index}.csv')
        cand_prop['loss_va_link'] = os.path.join(results_dir, f'{exp}/losses/valid/c{cand_id}_r{best_index}.csv')
        cand_prop['loss_te_link'] = os.path.join(results_dir, f'{exp}/losses/test/c{cand_id}_r{best_index}.csv')
        cand_prop['syn_link'] = os.path.join(results_dir, f'{exp}/synthetic/c{cand_id}.csv')

    losses_tr = [cand_prop['loss_tr'] for cand_prop in container_dict.values()]
    losses_va = [cand_prop['loss_va'] for cand_prop in container_dict.values()]
    losses_te = [cand_prop['loss_te'] for cand_prop in container_dict.values()]

    for i, cand_prop in enumerate(container_dict.values()):
        cand_prop['catg_tr'] = categorize_loss(cand_prop['loss_tr'], losses_tr, n_bins)
        cand_prop['catg_va'] = categorize_loss(cand_prop['loss_va'], losses_va, n_bins)
        cand_prop['catg_te'] = categorize_loss(cand_prop['loss_te'], losses_te, n_bins)
    return container_dict


def weight_loss(df_loss, wg_dict, cr_names):
    df_loss = deepcopy(df_loss)
    if wg_dict is None:
        wg_dict = {key:1 for key in df_loss.columns}
    if cr_names is None:
        cr_names = list(set(wg_dict.keys()).intersection(set(df_loss.columns)))
    df_loss = df_loss[cr_names]
    wg_dict = {key: val for key, val in wg_dict.items() if key in cr_names}
    weights = pd.Series(wg_dict)
    weights = weights/weights.sum()
    losses = df_loss.mean(0)
    return (weights*losses).sum()


def find_MaxEpochs(losses_array):
    if losses_array.ndim == 1:
        losses = list(losses_array)
    elif losses_array.ndim == 2:
        losses = list(losses_array.mean(1))
    try:
        epoch_max = losses.index(0)
    except:
        epoch_max = len(losses)
    return epoch_max


def get_separate_performances(results_dir, data_dir, weight_cr, reduction, handle_outliers):
    
    cr_seen = ['MMD_cdt', 'CorrD', 'CndD']
    cr_unseen = ['MMD_Fr', 'BinnedD', 'CorrD_N', 'NpMom', 'EDF_Mrg']
    cr_mixed = ['MMD_cdt', 'MMD_Fr', 'CorrD', 'BinnedD', 'CorrD_N', 'NpMom', 'CndD', 'EDF_Mrg']

    container_0 = prepare_results(
        results_dir=results_dir, data_dir=data_dir, cr_names=cr_seen, weight_cr=weight_cr, reduction=reduction, handle_outliers=handle_outliers,)
    container_1 = prepare_results(
        results_dir=results_dir, data_dir=data_dir, cr_names=cr_unseen, weight_cr=weight_cr, reduction=reduction, handle_outliers=handle_outliers,)
    container_2 = prepare_results(
        results_dir=results_dir, data_dir=data_dir, cr_names=cr_mixed, weight_cr=weight_cr, reduction=reduction, handle_outliers=handle_outliers,)

    df_0 = container_to_df(container_0)
    df_1 = container_to_df(container_1)
    df_2 = container_to_df(container_2)
    
    cols_old = ['loss_tr', 'loss_va', 'loss_te', 'loss_tr_std', 'loss_va_std', 'loss_te_std', 'catg_tr', 'catg_va', 'catg_te']
    cols_merge = ['cand_name', 'experiment', 'description', 'n_nodes', 'n_edges']
    
    columns_0 = dict(zip(cols_old, [c+'_seen' for c in cols_old]))
    columns_1 = dict(zip(cols_old, [c+'_unseen' for c in cols_old]))
    columns_2 = dict(zip(cols_old, [c+'_mixed' for c in cols_old]))

    df_0 = df_0.rename(columns=columns_0)
    df_1 = df_1.rename(columns=columns_1)
    df_2 = df_2.rename(columns=columns_2)
    
    df = pd.merge(df_0, df_1, on=cols_merge)
    df = pd.merge(df, df_2, on=cols_merge)
    return df


def perform_independence_test(data, test_type=2, num_perm=50, cv_grid=None):
    # test_type 0=> x,y ; 1=> x,y||z ; 2=> x,y||z,w
    #cv_grid = [1e-2, .2, .4, 2, 8, 64, 512]
    indep = '\perp\!\!\!\!\!\perp'
    
    OA = data[:, 0:1]
    OB = data[:, 1:2]
    SA = data[:, 2:3]
    SB = data[:, 3:4]
    
    OA_OB = np.concatenate((OA, OB), axis=1)
    OA_SA = np.concatenate((OA, SA), axis=1)
    OA_SB = np.concatenate((OA, SB), axis=1)
    OB_SA = np.concatenate((OB, SA), axis=1)
    OB_SB = np.concatenate((OB, SB), axis=1)
    SA_SB = np.concatenate((SA, SB), axis=1)

    if test_type == 0:
        pval_dict = {
            f'$O_A {indep} O_B$': fcit.test(x=OA, y=OB, z=None, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_A {indep} S_A$': fcit.test(x=OA, y=SA, z=None, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_A {indep} S_B$': fcit.test(x=OA, y=SB, z=None, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_B {indep} S_A$': fcit.test(x=OB, y=SA, z=None, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_B {indep} S_B$': fcit.test(x=OB, y=SB, z=None, cv_grid=cv_grid, num_perm=num_perm),
            f'$S_A {indep} S_B$': fcit.test(x=SA, y=SB, z=None, cv_grid=cv_grid, num_perm=num_perm),
        }
        
    if test_type == 1:
        pval_dict = {
            f'$O_A {indep} O_B | S_A$': fcit.test(x=OA, y=OB, z=SA, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_A {indep} S_A | O_B$': fcit.test(x=OA, y=SA, z=OB, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_A {indep} S_B | O_B$': fcit.test(x=OA, y=SB, z=OB, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_B {indep} S_A | O_A$': fcit.test(x=OB, y=SA, z=OA, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_B {indep} S_B | O_A$': fcit.test(x=OB, y=SB, z=OA, cv_grid=cv_grid, num_perm=num_perm),
            f'$S_A {indep} S_B | O_A$': fcit.test(x=SA, y=SB, z=OA, cv_grid=cv_grid, num_perm=num_perm),
            
            f'$O_A {indep} O_B | S_B$': fcit.test(x=OA, y=OB, z=SB, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_A {indep} S_A | S_B$': fcit.test(x=OA, y=SA, z=SB, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_A {indep} S_B | S_A$': fcit.test(x=OA, y=SB, z=SA, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_B {indep} S_A | S_B$': fcit.test(x=OB, y=SA, z=SB, cv_grid=cv_grid, num_perm=num_perm),
            f'$O_B {indep} S_B | S_A$': fcit.test(x=OB, y=SB, z=SA, cv_grid=cv_grid, num_perm=num_perm),
            f'$S_A {indep} S_B | O_B$': fcit.test(x=SA, y=SB, z=OB, cv_grid=cv_grid, num_perm=num_perm),
        }

    if test_type == 2:
        pval_dict = {
            f'$(O_A {indep} O_B)|(S_A, S_B)$': fcit.test(x=OA, y=OB, z=SA_SB, cv_grid=cv_grid, num_perm=num_perm),
            f'$(O_A {indep} S_A)|(O_B, S_B)$': fcit.test(x=OA, y=SA, z=OB_SB, cv_grid=cv_grid, num_perm=num_perm),
            f'$(O_A {indep} S_B)|(O_B, S_A)$': fcit.test(x=OA, y=SB, z=OB_SA, cv_grid=cv_grid, num_perm=num_perm),
            f'$(O_B {indep} S_A)|(O_A, S_B)$': fcit.test(x=OB, y=SA, z=OA_SB, cv_grid=cv_grid, num_perm=num_perm),
            f'$(O_B {indep} S_B)|(O_A, S_A)$': fcit.test(x=OB, y=SB, z=OA_SA, cv_grid=cv_grid, num_perm=num_perm),
            f'$(S_A {indep} S_B)|(O_A, O_B)$': fcit.test(x=SA, y=SB, z=OA_OB, cv_grid=cv_grid, num_perm=num_perm),
        }
    return pval_dict


def handle_outlier(values, thresholds=(0, 30), replace='max'):

    vals_nrm = [v for v in values if (thresholds[0] < v < thresholds[1])]

    if replace == 'min':
        val_rep = np.min(vals_nrm)
    elif replace == 'max':
        val_rep = np.max(vals_nrm)
    elif replace == 'mean':
        val_rep = np.mean(vals_nrm)
    elif replace == 'remove':
        val_rep = None
    else: 
        val_rep = replace
    
    result = []
    for v in values:
        if v in vals_nrm:
            result.append(v)
        elif val_rep is not None:
            result.append(val_rep)
        else:
            pass
            
    return result


def decompose_conditionals(data1, data2, sampling_rate, corr_type):
    #['01_23', '02_13', '03_12', '12_03', '13_02', '23_01'],
    #['3_012', '2_013', '1_023', '0_123'],
    self = CorrPatterns(sampling_rate=sampling_rate)
    patterns1 = self.get_patterns(data1, corr_type)
    patterns2 = self.get_patterns(data2, corr_type)
    distance_dict = {}
    for k in patterns1.keys():
        corrMats1 = patterns1[k]['corr']
        sampMats1 = patterns1[k]['samp']
        corrMats2 = patterns2[k]['corr']
        sampMats2 = patterns2[k]['samp']
        diffCorrs = (corrMats1 - corrMats2) * np.sqrt(sampMats1 * sampMats2) / ((np.mean(sampMats1) + np.mean(sampMats2))/2)
        distance_dict[k] = np.sqrt(np.mean(diffCorrs**2))
    return distance_dict


def measure_marginal_distance(data, metrics):
    O_A = data[:, 0] 
    O_B = data[:, 1]
    S_A = data[:, 2]
    S_B = data[:, 3]

    meas_dict = {
        '$O_A-O_B$': [m.predict(O_A, O_B) for m in metrics],
        '$O_A-S_A$': [m.predict(O_A, S_A) for m in metrics],
        '$O_A-S_B$': [m.predict(O_A, S_B) for m in metrics],
        '$O_B-S_A$': [m.predict(O_B, S_A) for m in metrics],
        '$O_B-S_B$': [m.predict(O_B, S_B) for m in metrics],
        '$S_A-S_B$': [m.predict(S_A, S_B) for m in metrics],
    }
    return meas_dict


def get_calibX(data_true, measure, n_epochs, n_outputs=10):
    batch_size = 8000
    measure.preallocate_memory(batch_size)
    loader_true1 = DataLoader(data_true, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_true2 = DataLoader(data_true, batch_size=batch_size, shuffle=True, drop_last=True)
    n_batches = min(len(loader_true1), len(loader_true2))
    loss_tensor = th.zeros((n_outputs, n_epochs, n_batches))
    for index_epoch in range(n_epochs):
        for index_batch, (batch_true1, batch_true2) in enumerate(zip(loader_true1, loader_true2)):
            loss_tensor[:, index_epoch, index_batch] = measure(batch_true1, batch_true2, False)
    return loss_tensor.detach().mean(2).mean(1)

def get_calib(data_true, measure, n_epochs):
    if measure.is_MMD():
        batch_size = 500
    else:
        batch_size = 8000
    measure.preallocate_memory(batch_size)
    loader_true1 = DataLoader(data_true, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_true2 = DataLoader(data_true, batch_size=batch_size, shuffle=True, drop_last=True)
    n_batches = min(len(loader_true1), len(loader_true2))
    loss_tensor = th.zeros((n_epochs, n_batches))  
    for index_epoch in range(n_epochs):
        for index_batch, (batch_true1, batch_true2) in enumerate(zip(loader_true1, loader_true2)):
            loss_tensor[index_epoch, index_batch] = measure(batch_true1, batch_true2, False)
    return th.mean(loss_tensor).detach()

def get_calibs(data_true, measures_list, n_epochs):
    calibrations = th.zeros(len(measures_list))
    for index_measure, measure in enumerate(measures_list):
        calibrations[index_measure] = get_calib(data_true, measure, n_epochs)
    return calibrations

def calibrate_loss(loss_tensor, calib_tensor):
    loss_calib = deepcopy(loss_tensor)
    n_candidates, n_measures, n_epochs = loss_tensor.shape
    for i in range(n_candidates):
        for k in range(n_epochs):
            loss_calib[i, :, k] = (loss_tensor[i, :, k]/calib_tensor)-1
    loss_calib[loss_calib<0]=0
    return loss_calib

def average_over_batches(data_true, data_pred, measure):
    if measure.is_MMD():
        batch_size = 500
    else:
        batch_size = 8000
    measure.preallocate_memory(batch_size)
    loader_true = DataLoader(data_true, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_pred = DataLoader(data_pred, batch_size=batch_size, shuffle=True, drop_last=True)
    n_batches = min(len(loader_true), len(loader_pred))
    loss_tensor = th.zeros(n_batches)
    for index_batch, (batch_true, batch_pred) in enumerate(zip(loader_true, loader_pred)):
        loss_tensor[index_batch] = measure(batch_true, batch_pred, False)
    return loss_tensor.mean().detach()

def evaluate_candidates(candidates_list, measures_list, data_true, n_epochs, container):
    n_candidates = len(candidates_list)
    n_measures = len(measures_list)
    loss_tensor = th.zeros((n_candidates, n_measures, n_epochs))
    for index_candidate, candidate in enumerate(candidates_list):
        data_pred = container[candidate]['syn_data']
        for index_measure, measure in enumerate(measures_list):
            for index_epoch in range(n_epochs):
                loss = average_over_batches(data_true, data_pred, measure)
                loss_tensor[index_candidate, index_measure, index_epoch] = loss
    return loss_tensor