import torch as th
import numpy as np
import itertools #used for combinations in CorrD
from pyro.contrib.gp.kernels import Exponential
from pyro.contrib.gp.kernels import Cosine
from pyro.contrib.gp.kernels import RBF

#base class for distance functions
##############################################################################
class dist_base_class(th.nn.Module):
    def __init__(self, name, is_MMD = False, is_Marg = False, is_Fourier = False):
        super().__init__()
        #basic properties
        self.name = name
        self.is_MMD_fun = is_MMD
        self.is_Marg_fun = is_Marg
        self.is_Fourier = is_Fourier
        #calibration a*x + b
        self.valRefVal = 1
        self.valBnchmark = 10
        self.a = 1
        self.b = 0
    
    def preallocate_memory(self, batch_size):
        pass
    
    def set_main_par(self, par):
        pass
    
    def set_main_pars(self, par1, par2):
        pass
    
    def get_name(self, short):
        if short:
            return self.name
        return self.name + '(' + self._create_description() + ')'
    
    def _create_description(self):
        return ''
        
    def is_MMD(self):
        return self.is_MMD_fun
    
    def is_Marg(self):
        return self.is_Marg_fun
    
    def is_Fourier(self):
        return self.is_Fourier
    
    def save_calib(self, ref_val, benchmark):
        self.a = (self.valBnchmark-self.valRefVal) / (benchmark-ref_val)
        self.b = (self.valRefVal*benchmark - self.valBnchmark*ref_val) / (benchmark-ref_val)
    
    def forward(self, d1, d2, normalize = False):
        if normalize:
            return self.a * self._forward_implementation(d1,d2) + self.b
        return self._forward_implementation(d1,d2)
        

#definitions of particular distance functions
##############################################################################
class MMD_cdt(dist_base_class):
    '''
    MMD_cdt is the standard MMD implemented in the CDT package.
    '''
    def __init__(self, bandwidth = [0.01, 0.1, 1, 10, 100]):
        super().__init__('MMD_cdt', is_MMD=True)
        self.bandwidths = th.Tensor(bandwidth)
    
    def preallocate_memory(self, batch_size):
        self.batch_size = batch_size
        s = th.cat([th.ones([batch_size, 1]) / batch_size, th.ones([batch_size, 1]) / -batch_size], 0)
        self.register_buffer('bw', self.bandwidths.unsqueeze(0).unsqueeze(0))
        self.register_buffer('S', (s @ s.t()))
        
    def set_main_par(self, par):
        self.bandwidths = th.Tensor(par)
        self.preallocate_memory(self.batch_size)
    
    def _create_description(self):
        if len(self.bandwidths) == 1:
            return '(bw:' + str(self.bandwidths)
        return '(mult-bw)'
    
    def test_bw_presence(self):
        return self.bw
        
    def _forward_implementation(self, x, y):
        X = th.cat([x, y], 0)
        XX = X @ X.t()
        X2 = (X * X).sum(dim=1).unsqueeze(0)
        exponent = -2*XX + X2.expand_as(XX) + X2.t().expand_as(XX)
        b = exponent.unsqueeze(2).expand(-1,-1, self.bw.shape[2]) * -self.bw
        return th.sum(self.S.unsqueeze(2) * b.exp())

class MMD_s_pro(dist_base_class):
    '''
    MMD_s_pro implements the standard MMD only.
    It can use 1) different kernels (RBF, Cosine, Exponential), 2) variances, and 3) lengthscales.
    '''
    def __init__(self, kernels = ['RBF'], bandwidths = [[1]], variances = [[1]]):
        super().__init__('MMD_s_pro',is_MMD=True)
        self.bandwidths = th.tensor(bandwidths)
        self.variances = th.tensor(variances)
        self.kernels = kernels
        
    def set_main_pars(self, par1, par2):
        self.bandwidths = th.tensor(par1)
        self.variances = th.tensor(par2)
        
    def _create_description(self):
        if len(self.bandwidths) == 1:
            str_bw =  'bw:' + str(self.bandwidths.detach().tolist())
        else:
            str_bw =  '(mult-bw)'
        if len(self.variances) == 1:
            str_vr = 'vr:' + str(self.variances.detach().tolist())
        else:
            str_vr =  '(mult-vr)'
        return str_bw + ',' + str_vr
            
    def _forward_implementation(self, d1, d2):
        distances = th.zeros(len(self.kernels))
        for ind,(kernel,bw,vr) in enumerate(zip(self.kernels, self.bandwidths, self.variances)):
            K_11 = _K_giver_pro(d1, d1, kernel_name=kernel, lengthscale=bw, variance=vr)
            K_22 = _K_giver_pro(d2, d2, kernel_name=kernel, lengthscale=bw, variance=vr)
            K_12 = _K_giver_pro(d1, d2, kernel_name=kernel, lengthscale=bw, variance=vr)
            distances[ind] = th.mean(K_11+K_22-2*K_12)
        return th.mean(distances)
    
class MMD_Fourier(dist_base_class):
    '''
    MMD_Fourier is a low-dim approximation of the standard MMD proposed in the RCC paper
    '''
    def __init__(self, bandwidths = [0.01, 0.1, 1, 10, 100], n_RandComps=100):
        super().__init__('MMD_Fr', is_Fourier=False) #set to False since it can use the larger batch size
        self.scales = th.Tensor(bandwidths)
        self.n_RandComps = n_RandComps
        #self.W = {n_vars: self._w_giver(n_vars) for n_vars in range(1, 10)}
        self.W = self._w_giver(4)
        
    def set_main_par(self, par):
        self.scales = th.Tensor(par)

    def _forward_implementation(self, x, y):
        mu_1 = self._mu_giver(x)
        mu_2 = self._mu_giver(y)
        distance = (mu_1-mu_2).pow(2).mean()
        return distance
    
    def _w_giver(self, n_vars):
        p1 = th.cat([scale*th.randn(self.n_RandComps, n_vars) for scale in self.scales])
        p2 = 2*np.pi*th.rand(self.n_RandComps*len(self.scales),1)
        return th.cat((p1, p2),1).T
    
    def _mu_giver(self, x):
        n_points, n_features = x.shape
        #w_x = self.W[n_features]
        w_x = self.W
        b_x = th.ones((n_points,1), requires_grad=True)
        coef = np.sqrt(2/w_x.shape[1])
        return coef * th.mm(th.cat((x, b_x),1), w_x).cos().mean(0)    

class NpMom(dist_base_class):
    '''
    Distance which consider n-party moments
    '''
    def __init__(self, num_moments=4, weighting_exp=2):
        super().__init__('NpMom')
        self.num_moments = num_moments
        self.weighting_exp = weighting_exp

    def _create_description(self):
        return str(self.num_moments) + 'm'
        
    def set_main_par(self, par):
        self.num_moments = par

    def _forward_implementation(self, d1, d2):
        moms1 = th.zeros([self.num_moments+1,self.num_moments+1,self.num_moments+1,self.num_moments+1])
        moms2 = th.zeros([self.num_moments+1,self.num_moments+1,self.num_moments+1,self.num_moments+1])
        centers1, stds1 = _calc_moments(d1,2)
        centers2, stds2 = _calc_moments(d2,2)
        for k in range(self.num_moments+1):
            for l in range(self.num_moments+1):
                for m in range(self.num_moments+1):
                    for n in range(self.num_moments+1):
                        if k*l*m*n == 0:
                            continue
                        moms1[k,l,m,n] = _calc_cross_moment(d1,th.tensor([[k,l,m,n]]),centers1,stds1)
                        moms2[k,l,m,n] = _calc_cross_moment(d2,th.tensor([[k,l,m,n]]),centers2,stds2)
        return th.mean(th.abs(moms1 - moms2)**self.weighting_exp)

class CndD(dist_base_class):
    '''
    Distance which considers all 1D conditional distributions (considering
    all possible permutation of variables) and evaluates single distribution
    moments. Note, that the joint distribution can be decomposed into such distributions
    '''
    def __init__(self, sample_weighting=True, sampling_rate=3, num_std_moments=4, weighting_exp=1):
        super().__init__('CndD')
        self.sample_weighting = sample_weighting
        self.sampling_rate = sampling_rate
        self.num_std_moments = num_std_moments
        self.weighting_exp = weighting_exp
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        #calculate bin edges
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
        #create lists of variable combinations
        self._create_var_cnd_lists()
            
    def _create_description(self):
        return str(self.sampling_rate) + 's' + str(self.num_std_moments) + 'm'
        
    def set_main_par(self, par):
        self.sampling_rate = par
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
    
    def set_main_pars(self, par1, par2):
        self.sampling_rate = par1
        self.num_std_moments = par2
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
    
    def _create_var_cnd_lists(self):
        self.list_var = [[] for k in range(4)]
        self.list_cnd = [[] for k in range(4)]
        for k in range(4):
            for comb in list(itertools.combinations([0,1,2,3],k)):
                self.list_var[k].append(np.setdiff1d([0,1,2,3],comb))
                self.list_cnd[k].append(np.array(comb))
    
    def _forward_implementation(self, d1, d2):
        #segregate all data into bins
        binned_data_inds1 = _bin_all_marginals(d1,self.sampling_rate,self.edg_obs_low,self.edg_obs_upp,self.edg_set_low,self.edg_set_upp)
        binned_data_inds2 = _bin_all_marginals(d2,self.sampling_rate,self.edg_obs_low,self.edg_obs_upp,self.edg_set_low,self.edg_set_upp)
        dist_list = th.zeros([4])
        #loop over amount of conditioned variables
        #no conditioning
        dist_list[0] = (_calc_marg_mom_diff(d1,d2,self.num_std_moments))**2 #same as MargD
        #condition on one variable
        dist_list[1] = self._calc_distr_diff_1(d1,d2,binned_data_inds1,binned_data_inds2,self.list_var[1],self.list_cnd[1])
        #condition on two variables
        dist_list[2] = self._calc_distr_diff_2(d1,d2,binned_data_inds1,binned_data_inds2,self.list_var[2],self.list_cnd[2])
        #condition on three variables
        dist_list[3] = self._calc_distr_diff_3(d1,d2,binned_data_inds1,binned_data_inds2,self.list_var[3],self.list_cnd[3])
        return th.sqrt(th.sum(dist_list))

    def _calc_distr_diff_1(self, d1 , d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_cnd),self.sampling_rate])
        weights = th.ones([len(list_cnd),self.sampling_rate])
        for ind_cmb,(vrs,cmb) in enumerate(zip(list_var,list_cnd)):
            for k in range(self.sampling_rate):
                dC1 = _condition_distribution(d1, binned_data_inds1, vrs, cmb, [k])
                dC2 = _condition_distribution(d2, binned_data_inds2, vrs, cmb, [k])
                if self.sample_weighting:
                    weights[ind_cmb,k] = (dC1.shape[0] * dC2.shape[0])**self.weighting_exp
                m1o,m1e = _calc_moments(dC1,self.num_std_moments)
                m2o,m2e = _calc_moments(dC2,self.num_std_moments)
                diff = th.cat((m1o-m2o,m1e-m2e),dim=0)
                diffs[ind_cmb,k] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum
    
    def _calc_distr_diff_2(self, d1 , d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_cnd),self.sampling_rate,self.sampling_rate])
        weights = th.ones([len(list_cnd),self.sampling_rate,self.sampling_rate])
        for ind_cmb,(vrs,cmb) in enumerate(zip(list_var,list_cnd)):
            for k in range(self.sampling_rate):
                for l in range(self.sampling_rate):
                    dC1 = _condition_distribution(d1, binned_data_inds1, vrs, cmb, [k,l])
                    dC2 = _condition_distribution(d2, binned_data_inds2, vrs, cmb, [k,l])
                    if self.sample_weighting:
                        weights[ind_cmb,k,l] = (dC1.shape[0] * dC2.shape[0])**self.weighting_exp
                    m1o,m1e = _calc_moments(dC1,self.num_std_moments)
                    m2o,m2e = _calc_moments(dC2,self.num_std_moments)
                    diff = th.cat((m1o-m2o,m1e-m2e),dim=0)
                    diffs[ind_cmb,k,l] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum
    
    def _calc_distr_diff_3(self, d1 , d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_cnd),self.sampling_rate,self.sampling_rate,self.sampling_rate])
        weights = th.ones([len(list_cnd),self.sampling_rate,self.sampling_rate,self.sampling_rate])
        for ind_cmb,(vrs,cmb) in enumerate(zip(list_var,list_cnd)):
            for k in range(self.sampling_rate):
                for l in range(self.sampling_rate):
                    for m in range(self.sampling_rate):
                        dC1 = _condition_distribution(d1, binned_data_inds1, vrs, cmb, [k,l,m])
                        dC2 = _condition_distribution(d2, binned_data_inds2, vrs, cmb, [k,l,m])
                        if self.sample_weighting:
                            weights[ind_cmb,k,l,m] = (dC1.shape[0] * dC2.shape[0])**self.weighting_exp
                        m1o,m1e = _calc_moments(dC1,self.num_std_moments)
                        m2o,m2e = _calc_moments(dC2,self.num_std_moments)
                        diff = th.cat((m1o-m2o,m1e-m2e),dim=0)
                        diffs[ind_cmb,k,l,m] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum


    
class CorrD(dist_base_class):
    '''
    Custom distance which considers two types of correlation values from
    1) it calculates conditional joint distributions for each pair of two variables
       condiditoned on the remaining two observables, considering all possible
       combinations
    2) calculates distribution for each variable conditioned on the remaining three
       observables.
    '''
    def __init__(self, corr_types = [1,1], sampling_rate = 4):
        super().__init__('CorrD')
        self.corr_types = corr_types
        self.sampling_rate = sampling_rate
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        #calculate bin edges
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
        #create lists of variable combinations
        self.var_list_2 = [np.array(vS) for vS in list(itertools.combinations([0,1,2,3],2))]
        self.var_cnd_list_2 = [np.setdiff1d([0,1,2,3],l) for l in self.var_list_2]
        self.var_list_3 = [np.array(vS) for vS in list(itertools.combinations([0,1,2,3],3))]
        self.var_cnd_list_3 = [np.setdiff1d([0,1,2,3],l) for l in self.var_list_3]
        
    def _create_description(self):
        if self.corr_types == [1,0]:
            typestring = '2'
        elif self.corr_types == [0,1]:
            typestring = '3'
        elif self.corr_types == [1,1]:
            typestring = 'a'
        return typestring + ',' + str(self.sampling_rate) + 's'
        
    def set_main_par(self, par):
        self.sampling_rate = par
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
        
    def _forward_implementation(self, d1, d2):
        #segregate all data into bins
        binned_data_inds1 = _bin_all_marginals(d1,self.sampling_rate,self.edg_obs_low,self.edg_obs_upp,self.edg_set_low,self.edg_set_upp)
        binned_data_inds2 = _bin_all_marginals(d2,self.sampling_rate,self.edg_obs_low,self.edg_obs_upp,self.edg_set_low,self.edg_set_upp)
        corr_dists = th.zeros([2])
        if self.corr_types[0]:
            #calculate sets of correlation matrices
            corrMats1,sampMats1 = self._detCorrs2(d1,binned_data_inds1)
            corrMats2,sampMats2 = self._detCorrs2(d2,binned_data_inds2)
            #calculate differences in correlation values weighted by numbers of samples in each region
            diffCorrs2 = (corrMats1 - corrMats2) * th.sqrt(sampMats1 * sampMats2) / ((th.mean(sampMats1) + th.mean(sampMats2))/2) #weigh correlation values with amount of samples
            corr_dists[0] = th.sqrt(th.mean(diffCorrs2**2))
        if self.corr_types[1]:
            #calculate sets of correlation matrices
            corrVecs1,sampVecs1 = self._detCorrs3(d1,binned_data_inds1)
            corrVecs2,sampVecs2 = self._detCorrs3(d2,binned_data_inds2)
            #calculate differences in correlation values weighted by numbers of samples in each region
            diffCorrs3 = (corrVecs1 - corrVecs2) * th.sqrt(sampVecs1 * sampVecs2) / ((th.mean(sampVecs1) + th.mean(sampVecs2))/2) #weigh correlation values with amount of samples
            corr_dists[1] = th.sqrt(th.mean(diffCorrs3**2))
        return th.sqrt(th.sum(corr_dists**2))
    
    def _detCorrs2(self, d, binned_data_inds):
        corrMats = th.zeros([6,self.sampling_rate,self.sampling_rate]) #we have 6 possible pairs of observables
        corrSamp = th.zeros([6,self.sampling_rate,self.sampling_rate]) #we have 6 possible pairs of observables
        for ind,(vs,vCs) in enumerate(zip(self.var_list_2,self.var_cnd_list_2)):
            corrMats[ind,:,:],corrSamp[ind,:,:] = self._calc_corr_mat(d,vs,vCs,binned_data_inds)
        return corrMats, corrSamp
    
    def _detCorrs3(self, d, binned_data_inds):
        corrVecs = th.zeros([4,self.sampling_rate]) #we have 4 possible triples of observables
        corrSamp = th.zeros([4,self.sampling_rate]) #we have 4 possible triples of observables
        for ind,(vs,vCs) in enumerate(zip(self.var_list_3,self.var_cnd_list_3)):
            corrVecs[ind,:],corrSamp[ind,:] = self._calc_corr_vec(d,vs,vCs,binned_data_inds)
        return corrVecs, corrSamp
    
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
    
    def _calc_full_corr(self, d):
        #calculate marginal standard deviations
        stdProd = th.nan_to_num(th.prod(th.std(d,dim=0)))
        if stdProd == 0:
            return 0 #at least one of the variables does not change, they cannot be correlated
        #calculate correlation
        return th.mean(th.prod(d-th.mean(d,dim=0),dim=1)) / stdProd
    
class CorrD_pen(dist_base_class):
    '''
    Old version of biased CorrD with an additional penalty for regions with zero sampling
    '''
    def __init__(self, sampling_rate = 4, wg_penalty = 1):
        super().__init__('CorrD_pen')
        self.sampling_rate = sampling_rate
        self.wg_penalty = wg_penalty
    
    def _create_description(self):
        return str(self.sampling_rate) + 's'
    
    def set_main_par(self, par):
        self.sampling_rate = par
    
    def _forward_implementation(self, d1, d2):
        #diffCorrs = th.tensor(0.0, requires_grad=True)
        #calculate correlation matrix for observables conditioned on settings
        corrsObs1,corrSet1,samples1,_ = _detCorrs(d1,self.sampling_rate)
        corrsObs2,corrSet2,samples2,_ = _detCorrs(d2,self.sampling_rate)
        #select only inner area of matrix corresponding to physical setting values
        corrsObs1 = corrsObs1[1:self.sampling_rate+1,1:self.sampling_rate+1]
        corrsObs2 = corrsObs2[1:self.sampling_rate+1,1:self.sampling_rate+1]
        samples1 = samples1[1:self.sampling_rate+1,1:self.sampling_rate+1]
        samples2 = samples2[1:self.sampling_rate+1,1:self.sampling_rate+1]
        #calculate correlation values weighted by numbers of samples in each region
        diffCorrs = (corrsObs1 - corrsObs2) * th.sqrt(samples1 * samples2) / ((th.mean(samples1) + th.mean(samples2))/2) #weigh correlation values with amount of samples
        #diffCorrs = th.sqrt(th.mean(diffCorrs**2) + (corrSet1 - corrSet2)**2)
        diffCorrs = th.sqrt(th.mean(diffCorrs**2))
        #add penalty for undersampled regions
        penalty_zero = th.tensor(2.0,requires_grad=True) * np.abs(np.count_nonzero(samples1) - np.count_nonzero(samples2)) / (samples1.shape[0]*samples1.shape[1])
        return (self.wg_penalty*penalty_zero+diffCorrs) / (1+self.wg_penalty)
    
class BinnedD(dist_base_class):
    '''
    A group of distances which bins the distributions to discrete ones and uses metrics for discrete distributions
    '''
    def __init__(self, dist_type='JS', sampling_rate=4):
        super().__init__('BinnedD')
        self.dist_type = dist_type
        self.sampling_rate = sampling_rate
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        #calculate bin edges
        self.range_full = [(self.range_obs[0],self.range_obs[1]),(self.range_obs[0],self.range_obs[1]),(self.range_set[0],self.range_set[1]),(self.range_set[0],self.range_set[1])]

    def _create_description(self):
        return self.dist_type + ',' + str(self.sampling_rate) + 's'
        
    def set_main_par(self, par):
        self.sampling_rate = par
        
    def _forward_implementation(self, d1, d2):
        #convert to numpy
        d1n = d1.detach().numpy()
        d2n = d2.detach().numpy()
        #bin data
        b1,_ = np.histogramdd(d1n,bins=self.sampling_rate,range=self.range_full)
        b2,_ = np.histogramdd(d2n,bins=self.sampling_rate,range=self.range_full)
        #convert counts to probs (no normalization, since some counts might be lost if outside of considered ranges)
        b1 /= d1n.shape[0]
        b2 /= d2n.shape[0]
        #calculate distance
        return th.tensor(self._calc_dist(b1,b2))
    
    def _calc_dist(self, b1, b2):
        if self.dist_type == 'RMSD': #rmsd
            return np.sqrt(np.mean((b1 - b2)**2))
        elif self.dist_type == 'JS': #Jensen–Shannon
            b1q = b1.copy()
            b2q = b2.copy()
            c = (b1 + b2) / 2
            #avoid nan values without changing overall result
            c[c==0] = 1
            b1q[b1q==0] = 1
            b2q[b2q==0] = 1
            #calculate values
            div1 = np.sum(b1 * np.log(b1q / c))
            div2 = np.sum(b2 * np.log(b2q / c))
            return (div1 + div2)/2
        elif self.dist_type == 'He': #Hellinger
            return np.sqrt(np.sum((np.sqrt(b1) - np.sqrt(b2))**2)) / np.sqrt(2)
        elif self.dist_type == 'Tot': #total variational distance
            return np.max(np.abs(b1 - b2))
        #else:
        self._print_Binning_warning_text('BinnedD')
        self.dist_type = 'JS'
        return self._calc_dist(b1,b2)

class CorrD_N(dist_base_class):
    '''
    A distance which considers unconditioned n-party correlations
    '''
    def __init__(self):
        super().__init__('CorrD_N')
        #create lists of variable combinations
        self.var_list_2 = [np.array(vS) for vS in list(itertools.combinations([0,1,2,3],2))]
        self.var_list_3 = [np.array(vS) for vS in list(itertools.combinations([0,1,2,3],3))]
        
    def _forward_implementation(self, d1, d2):
        corr_dists = th.zeros([3])
        #calculate marginal two party correlations
        corrVec2_1 = self._detMargCorrs2(d1)
        corrVec2_2 = self._detMargCorrs2(d2)
        #calculate differences
        corr_dists[0] = th.sqrt(th.mean((corrVec2_1 - corrVec2_2)**2))
        #calculate marginal three party correlations
        corrVec3_1 = self._detMargCorrs3(d1)
        corrVec3_2 = self._detMargCorrs3(d2)
        #calculate differences
        corr_dists[1] = th.sqrt(th.mean((corrVec3_1 - corrVec3_2)**2))
        #calculate full correlations
        corr1 = self._calc_full_corr(d1)
        corr2 = self._calc_full_corr(d2)
        #calculate difference in correlation value
        corr_dists[2] = corr1 - corr2
        return th.sqrt(th.sum(corr_dists**2))
    
    def _detMargCorrs2(self, d):
        corrVec = th.zeros([6])
        for ind,vs in enumerate(self.var_list_2):
            corrVec[ind] = self._calc_full_corr(d[:,vs])
        return corrVec
    
    def _detMargCorrs3(self, d):
        corrVec = th.zeros([4])
        for ind,vs in enumerate(self.var_list_3):
            corrVec[ind] = self._calc_full_corr(d[:,vs])
        return corrVec
    
    def _calc_full_corr(self, d):
        #calculate marginal standard deviations
        stdProd = th.nan_to_num(th.prod(th.std(d,dim=0)))
        if stdProd == 0:
            return 0 #at least one of the variables does not change, they cannot be correlated
        #calculate correlation
        return th.mean(th.prod(d-th.mean(d,dim=0),dim=1)) / stdProd

#Marginal Distances
##############################################################################
class MargD(dist_base_class):
    '''
    A distance which considers only the distance between the marginals
    based on moments of distributions
    '''
    def __init__(self, num_std_moments = 4):
        super().__init__('MargD',is_Marg=True)
        self.num_std_moments = num_std_moments
        
    def _create_description(self):
        return str(self.num_std_moments) + 'm'
    
    def set_main_par(self, par):
        self.num_std_moments = par
        
    def _forward_implementation(self, d1, d2):
        return _calc_marg_mom_diff(d1,d2,self.num_std_moments)

class EDF_Marg(dist_base_class):
    '''
    A distance which considers only the distance between the marginals
    according to 1D EDF based distances with the same options as EDF_Multi
    '''
    def __init__(self, dist_type='AD', interp_samples=1):
        super().__init__('EDF_Mrg',is_Marg=True)
        self.dist_type = dist_type
        self.interp_samples = interp_samples
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        
    def _create_description(self):
        return self.dist_type + ',' + str(self.interp_samples) + 's'
        
    def set_main_par(self, par):
        self.interp_samples = par
    
    def _forward_implementation(self, d1, d2):
        dist_list = th.zeros(4)
        for v in range(4):
            cums, coords = self._calc_cums(d1[:,[v]],d2[:,[v]])
            dist_list[v] = self._calc_dist(cums, coords)
        return th.sqrt(th.sum(dist_list**2))
    
    def _calc_cums(self, d1, d2):
        #join arrays (coords) and create two counting columns (steps)
        nSamp1 = d1.shape[0]
        nSamp2 = d2.shape[0]
        dM = th.cat((th.cat((th.ones(nSamp1,1)/nSamp1,th.zeros(nSamp1,1),d1),axis=1),th.cat((th.zeros(nSamp2,1),th.ones(nSamp2,1)/nSamp2,d2),axis=1)),axis=0)
        #sort
        inds = th.argsort(dM[:,2])
        coords = dM[inds,2] #required for potential interpolation later
        dSort = dM[inds,0:2]
        #calculate cumulative distributions
        return th.cumsum(dSort,dim=0), coords
            
    def _calc_dist(self, cums, coords):
        if self.dist_type == 'KS': #Kolmogorov-Smirnov
            return th.max(th.abs(cums[:,0] - cums[:,1]))
        elif self.dist_type == 'CvM': #Cramer-vanMises
            return th.sqrt(th.mean((cums[:,0] - cums[:,1])**2))
        elif self.dist_type == 'AD': #Anderson-Darling
            cums = cums[:-1,:] #discard the last value, which is by construction 1 for both EDFs (first value is never 0)
            cums_mean = th.mean(cums,dim=1)
            wg = cums_mean * (1 - cums_mean)
            return th.sqrt(th.mean((cums[:,0] - cums[:,1])**2 / wg))
        else:
            _print_EDF_warning_text('EDF_Marg')
            self.dist_type = 'KS'
            return self._calc_dist(cums, coords)

class Binned_Marg(dist_base_class):
    '''
    A group of distances which bins the distributions to discrete ones and uses metrics for discrete distributions
    '''
    def __init__(self, dist_type, sampling_rate = 4):
        super().__init__('B_Marg',is_Marg=True)
        self.dist_type = dist_type
        self.sampling_rate = sampling_rate
        #[CONFIG]
        self.range_obs = np.array([-5,5])
        self.range_set = np.array([0,1])
        #calculate bin edges
        self._adjust_ranges()
        
    def _create_description(self):
        return self.dist_type + ',' + str(self.sampling_rate) + 's'
    
    def set_main_par(self, par):
        self.sampling_rate = par
        self._adjust_ranges()
        
    def _forward_implementation(self, d1, d2):
        #convert to numpy
        d1n = d1.detach().numpy()
        d2n = d2.detach().numpy()
        #prepare container
        dist_list = th.zeros([4])
        #loop ove variables
        for v in range(4):
            if v < 2:
                ran = self.range_obs
            else:
                ran = self.range_set
            b1,ed1 = np.histogram(d1n[:,v],bins=self.sampling_rate+2,range=ran) #+2 for the outer bins
            b2,ed2 = np.histogram(d2n[:,v],bins=self.sampling_rate+2,range=ran)
            dist_list[v] = self._calc_dist(b1,b2)
        return th.sqrt(th.mean(dist_list**2))

    def _adjust_ranges(self):
        bSize_obs = np.diff(self.range_obs) / self.sampling_rate
        bSize_set = np.diff(self.range_set) / self.sampling_rate
        self.range_obs[0] = self.range_obs[0] - bSize_obs
        self.range_obs[1] = self.range_obs[1] + bSize_obs
        self.range_set[0] = self.range_set[0] - bSize_set
        self.range_set[1] = self.range_set[1] + bSize_set
    
    def _calc_dist(self, b1, b2):
        if self.dist_type == 'RMSD': #rmsd
            return np.sqrt(np.mean((b1 - b2)**2))
        elif self.dist_type == 'JS': #Jensen–Shannon
            b1q = b1.copy()
            b2q = b2.copy()
            c = (b1 + b2) / 2
            #avoid nan values without changing overall result
            c[c==0] = 1
            b1q[b1q==0] = 1
            b2q[b2q==0] = 1
            #calculate values
            div1 = np.sum(b1 * np.log(b1q / c))
            div2 = np.sum(b2 * np.log(b2q / c))
            return (div1 + div2)/2
        elif self.dist_type == 'He': #Hellinger
            return np.sqrt(np.sum((np.sqrt(b1) - np.sqrt(b2))**2)) / np.sqrt(2)
        elif self.dist_type == 'Tot': #total variational distance
            return np.max(np.abs(b1 - b2))
        #else:
        self._print_Binning_warning_text('Binned_Marg')
        self.dist_type = 'JS'
        return self._calc_dist(b1,b2)

#binning
###############################################################################
def _calc_edges(sampling_rate, range_obs, range_set):
    fac_low = np.linspace(0,sampling_rate-1,sampling_rate)
    fac_upp = np.linspace(1,sampling_rate,sampling_rate)
    step_obs = np.diff(range_obs) / sampling_rate
    step_set = np.diff(range_set) / sampling_rate
    edg_obs_low = fac_low*step_obs + range_obs[0]
    edg_obs_upp = fac_upp*step_obs + range_obs[0]
    edg_set_low = fac_low*step_set + range_set[0]
    edg_set_upp = fac_upp*step_set + range_set[0]
    return edg_obs_low, edg_obs_upp, edg_set_low, edg_set_upp

def _bin_all_marginals(d, sampling_rate, edg_obs_low, edg_obs_upp, edg_set_low, edg_set_upp):
    binned_data_inds = th.zeros([4,sampling_rate,d.shape[0]],dtype=th.bool)
    binned_data_inds[0,:,:] = _bin_marginal(d[:,0],sampling_rate,edg_obs_low,edg_obs_upp)
    binned_data_inds[1,:,:] = _bin_marginal(d[:,1],sampling_rate,edg_obs_low,edg_obs_upp)
    binned_data_inds[2,:,:] = _bin_marginal(d[:,2],sampling_rate,edg_set_low,edg_set_upp)
    binned_data_inds[3,:,:] = _bin_marginal(d[:,3],sampling_rate,edg_set_low,edg_set_upp)
    return binned_data_inds

def _bin_marginal(marg,sampling_rate,edges_low,edges_upp):
    inds_bool = th.zeros([sampling_rate,marg.shape[0]],dtype=th.bool)
    for ind,(eL,eU) in enumerate(zip(edges_low,edges_upp)):
        inds_bool[ind] = th.logical_and(marg > eL,marg <= eU)
    return inds_bool

def _condition_distribution(d, binned_data_inds, var_inds, cnd_inds, bin_inds):
    #merge indices
    res_inds = th.ones(d.shape[0],dtype=bool)
    for k in range(len(cnd_inds)):
        res_inds = th.logical_and(res_inds,binned_data_inds[cnd_inds[k],bin_inds[k],:])
    dC = d[res_inds,:]
    return dC[:,var_inds]

#auxiliary
###############################################################################
def _calc_num_moment_types(num_moments):
    num_odd = int(np.ceil(num_moments/2))
    num_even = int(np.floor(num_moments/2))
    return num_odd, num_even

def _calc_marg_mom_diff(d1, d2, num_std_moments):
    m1o,m1e = _calc_moments(d1,num_std_moments)
    m2o,m2e = _calc_moments(d2,num_std_moments)
    diff_odd = m1o - m2o
    mean_even = (m1e + m2e) / 2
    diff_even = (m1e - m2e) / mean_even
    diff_even = m1e - m2e
    diff = th.cat((diff_odd,diff_even),dim=0)
    return th.sqrt(th.mean(diff**2))

def _calc_moments(d, num_std_moments):
    #0: mean
    #1: std
    #2: skewness
    #3: kurtosis
    if num_std_moments < 2:
        print('ERROR: at least 2 std moments needed')
        num_std_moments = 2
    num_odd,num_even = _calc_num_moment_types(num_std_moments)
    nb_var = d.shape[1]
    moments_odd = th.zeros([num_odd,nb_var])
    moments_even = th.zeros([num_even,nb_var])
    #check for zero amount of data
    if d.shape[0] == 0:
        return moments_odd, moments_even
    #calculate mean value
    means = th.mean(d,dim=0)
    moments_odd[0,:] = means
    #check for only a single datapoint
    if d.shape[0] == 1:
        return moments_odd, moments_even
    #calculate standard deviation
    stds = th.std(d,dim=0)
    moments_even[0,:] = stds
    #loop over remaining moments
    fill_ind = 1
    go_even = False
    for j in range(num_std_moments-2):
        moms = th.mean((d - means)**(j+3),dim=0) / stds**(j+3)
        if go_even:
            moments_even[fill_ind,:] = moms
            fill_ind += 1
            go_even = False
        else:
            moments_odd[fill_ind,:] = moms
            go_even = True
    return moments_odd, moments_even

def _calc_cross_moment(d, orders, centers, stds):
    stds_eff = th.ones(stds.shape)
    stds_eff[orders >= 3] = stds[orders >= 3] #only normalize to std if order is >= 3
    return th.mean(th.prod((d-centers)**orders / stds_eff**orders,dim=1))

def _detCorrs(dstr, sampling_rate):
    #columns 0 and 1 are the observables 2 and 3 of d are the settings
    minVal = 0;
    maxVal = 1;
    step = (maxVal-minVal) / sampling_rate;
    corrsObs = th.zeros([sampling_rate+2,sampling_rate+2]);
    corrSet = 0;
    samples = th.zeros([sampling_rate+2,sampling_rate+2]);
    #[Correlation of settings]
    corrSet = th.mean((dstr[:,2] - th.mean(dstr[:,2])) * (dstr[:,3] - th.mean(dstr[:,3]))) / (th.std(dstr[:,2]) * th.std(dstr[:,3]));
    #[Conditional correlations of observables]
    encounteredZeroSamplesInCoreRegion = False;
    for k in range(sampling_rate+2):
        #find lines where setting A is in certain range
        if k == 0:
            linesA = dstr[:,2] < minVal;
        elif k == sampling_rate+1:
            linesA = dstr[:,2] >= maxVal;
        else:
            linesA = th.logical_and(dstr[:,2] >= (k-1)*step+minVal,dstr[:,2] < k*step+minVal);
        for j in range (sampling_rate+2):
            #find lines where setting B is in certain range
            if j == 0:
                linesB = dstr[:,3] < minVal;
            elif j == sampling_rate+1:
                linesB = dstr[:,3] >= maxVal;
            else:
                linesB = th.logical_and(dstr[:,3] >= (j-1)*step+minVal,dstr[:,3] < j*step+minVal);
            #pick only the lines satisfying both conditions
            pick = dstr[th.logical_and(linesA,linesB),0:2];
            samples[k,j] = pick.shape[0]; #count how many samples are found in this particular setting ranges
            if samples[k,j] == 0:
                corrsObs[k,j] = 0;
                if (k > 0 and k < sampling_rate+1) and (j > 0 and j < sampling_rate+1):
                    encounteredZeroSamplesInCoreRegion = True;
            else:
                stdDevA = th.std(pick[:,0]);
                stdDevB = th.std(pick[:,1]);
                if not (stdDevA > 0 and stdDevB > 0):
                    corrsObs[k,j] = 0; #something that does not change, cannot be correlated
                else:    
                    corrsObs[k,j] = th.mean((pick[:,0] - th.mean(pick[:,0])) * (pick[:,1] - th.mean(pick[:,1]))) / (stdDevA * stdDevB);
    return corrsObs, corrSet, samples, encounteredZeroSamplesInCoreRegion;

def _K_giver_pro(a, b, kernel_name, **kwargs):
    input_dim = a.shape[1]
    if kernel_name == 'Cos':
        kernel = Cosine(input_dim=input_dim, **kwargs)
    elif kernel_name == 'Exp':
        kernel = Exponential(input_dim=input_dim, **kwargs)
    elif kernel_name == 'RBF':
        kernel = RBF(input_dim=input_dim, **kwargs) 
    K_ab = kernel.forward(a,b)
    return K_ab


def _print_Binning_warning_text(fun_str):
    print('WARNING: invalid option chosen for ' + fun_str + '!')
    print('Options are: "JS":   Jensen-Shannon\n')
    print('             "RMSD": Root-Mean-Squared-Distance\n')
    print('             "He":   Hellinger\n')
    print('             "Tot":  Total variational distance\n')
    print('choosing JS by default...')


def _print_EDF_warning_text(fun_str):
    print('WARNING: invalid option chosen for ' + fun_str + '!')
    print('Options are: "KS" : Kolmogorov-Smirnov\n')
    print('             "CvM": Cramer-vonMises\n')
    print('             "AD" : Anderson-Darling\n')
    print('choosing Kolmogorov-Smirnov by default...')


class CndDX(dist_base_class):
    '''
    Distance which considers all types of correlations
    '''
    def __init__(self, sample_weighting=True, sampling_rate=3, num_std_moments=4, weighting_exp=1):
        super().__init__('CndDX')
        self.sample_weighting = sample_weighting
        self.sampling_rate = sampling_rate
        self.num_std_moments = num_std_moments
        self.weighting_exp = weighting_exp
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        #calculate bin edges
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
        #create lists of variable combinations
        self._create_var_cnd_lists()
            
    def _create_description(self):
        return str(self.sampling_rate) + 's' + str(self.num_std_moments) + 'm'
        
    def set_main_par(self, par):
        self.sampling_rate = par
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
    
    def set_main_pars(self, par1, par2):
        self.sampling_rate = par1
        self.num_std_moments = par2
        self.edg_obs_low, self.edg_obs_upp, self.edg_set_low, self.edg_set_upp = _calc_edges(self.sampling_rate, self.range_obs, self.range_set)
    
    def _create_var_cnd_lists(self):
        self.list_var = [[] for _ in range(5)]
        self.list_cnd = [[] for _ in range(4)]
        for k in range(4):
            for comb in list(itertools.combinations([0,1,2,3], k)):
                diff = np.setdiff1d([0,1,2,3], comb) 
                self.list_var[len(diff)].append(list(diff))
                self.list_cnd[len(comb)].append(list(comb))
    
    def _forward_implementation(self, d1, d2):
        #segregate all data into bins
        binned_data_inds1 = _bin_all_marginals(d1,self.sampling_rate,self.edg_obs_low,self.edg_obs_upp,self.edg_set_low,self.edg_set_upp)
        binned_data_inds2 = _bin_all_marginals(d2,self.sampling_rate,self.edg_obs_low,self.edg_obs_upp,self.edg_set_low,self.edg_set_upp)
        dist_list = th.zeros([10])
        dist_list[0] = self._calc_distr_diff_0(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[1], self.list_cnd[0])
        dist_list[1] = self._calc_distr_diff_1(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[1], self.list_cnd[1])
        dist_list[2] = self._calc_distr_diff_2(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[1], self.list_cnd[2])
        dist_list[3] = self._calc_distr_diff_3(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[1], self.list_cnd[3])
        dist_list[4] = self._calc_distr_diff_0(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[2], self.list_cnd[0])
        dist_list[5] = self._calc_distr_diff_1(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[2], self.list_cnd[1])
        dist_list[6] = self._calc_distr_diff_2(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[2], self.list_cnd[2])
        dist_list[7] = self._calc_distr_diff_0(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[3], self.list_cnd[0])
        dist_list[8] = self._calc_distr_diff_1(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[3], self.list_cnd[1])
        dist_list[9] = self._calc_distr_diff_0(d1, d2, binned_data_inds1, binned_data_inds2, self.list_var[4], self.list_cnd[0])
        return dist_list
    
    def _calc_distr_diff_0(self, d1, d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_var), 1])
        weights = th.ones([len(list_var),1])
        for ind_cmb, vrs in enumerate(list_var):
            m1o,m1e = _calc_moments(d1[:, vrs], self.num_std_moments)
            m2o,m2e = _calc_moments(d2[:, vrs], self.num_std_moments)
            diff_odd = m1o - m2o
            mean_even = (m1e + m2e) / 2
            diff_even = (m1e - m2e) / mean_even
            diff_even = m1e - m2e
            diff = th.cat((diff_odd, diff_even), dim=0)
            diffs[ind_cmb] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum


    def _calc_distr_diff_1(self, d1 , d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_cnd),self.sampling_rate])
        weights = th.ones([len(list_cnd),self.sampling_rate])
        for ind_cmb,(vrs,cmb) in enumerate(zip(list_var,list_cnd)):
            for k in range(self.sampling_rate):
                dC1 = _condition_distribution(d1, binned_data_inds1, vrs, cmb, [k])
                dC2 = _condition_distribution(d2, binned_data_inds2, vrs, cmb, [k])
                if self.sample_weighting:
                    weights[ind_cmb,k] = (dC1.shape[0] * dC2.shape[0])**self.weighting_exp
                m1o,m1e = _calc_moments(dC1,self.num_std_moments)
                m2o,m2e = _calc_moments(dC2,self.num_std_moments)
                diff = th.cat((m1o-m2o,m1e-m2e),dim=0)
                diffs[ind_cmb,k] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum
    
    def _calc_distr_diff_2(self, d1 , d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_cnd),self.sampling_rate,self.sampling_rate])
        weights = th.ones([len(list_cnd),self.sampling_rate,self.sampling_rate])
        for ind_cmb,(vrs,cmb) in enumerate(zip(list_var,list_cnd)):
            for k in range(self.sampling_rate):
                for l in range(self.sampling_rate):
                    dC1 = _condition_distribution(d1, binned_data_inds1, vrs, cmb, [k,l])
                    dC2 = _condition_distribution(d2, binned_data_inds2, vrs, cmb, [k,l])
                    if self.sample_weighting:
                        weights[ind_cmb,k,l] = (dC1.shape[0] * dC2.shape[0])**self.weighting_exp
                    m1o,m1e = _calc_moments(dC1,self.num_std_moments)
                    m2o,m2e = _calc_moments(dC2,self.num_std_moments)
                    diff = th.cat((m1o-m2o,m1e-m2e),dim=0)
                    diffs[ind_cmb,k,l] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum
    
    def _calc_distr_diff_3(self, d1 , d2, binned_data_inds1, binned_data_inds2, list_var, list_cnd):
        diffs = th.zeros([len(list_cnd),self.sampling_rate,self.sampling_rate,self.sampling_rate])
        weights = th.ones([len(list_cnd),self.sampling_rate,self.sampling_rate,self.sampling_rate])
        for ind_cmb,(vrs,cmb) in enumerate(zip(list_var,list_cnd)):
            for k in range(self.sampling_rate):
                for l in range(self.sampling_rate):
                    for m in range(self.sampling_rate):
                        dC1 = _condition_distribution(d1, binned_data_inds1, vrs, cmb, [k,l,m])
                        dC2 = _condition_distribution(d2, binned_data_inds2, vrs, cmb, [k,l,m])
                        if self.sample_weighting:
                            weights[ind_cmb,k,l,m] = (dC1.shape[0] * dC2.shape[0])**self.weighting_exp
                        m1o,m1e = _calc_moments(dC1,self.num_std_moments)
                        m2o,m2e = _calc_moments(dC2,self.num_std_moments)
                        diff = th.cat((m1o-m2o,m1e-m2e),dim=0)
                        diffs[ind_cmb,k,l,m] = th.mean(diff**2)
        weight_sum = th.sum(weights)
        if weight_sum == 0:
            return th.sum(diffs)
        return th.sum(diffs * weights) / weight_sum