import torch as th
import numpy as np
import scipy as sp #used for multidimensional interpolation
from torchinterp1d import Interp1d #used for 1D interpolation install: https://github.com/aliutkus/torchinterp1d
import itertools #used for permutations in EDF_multi
import utilities_functions as uFun

#definitions of particular distance functions
##############################################################################
class MMD_s_pro(dist_base_class):
    '''
    MMD_s_pro implements the standard MMD only.
    It can use 1) different kernels (RBF, Cosine, Exponential), 2) variances, and 3) lengthscales.
    '''
    def __init__(self, kernel_counts = 5, kernel_name = 'RBF', variances = None, lengthscales = None):
        super().__init__('MMD_s_pro',is_MMD=True)
        self.kernel_counts = kernel_counts
        self.kernel_names = [kernel_name]*self.kernel_counts
        self.variances = variances if variances is not None else th.tensor(np.random.dirichlet([1 for i in range(self.kernel_counts)]).tolist()) 
        self.lengthscales =lengthscales if lengthscales is not None else th.tensor(np.logspace(start=-4, stop=+4, num=self.kernel_counts))
    
    def forward(self, d1, d2):
        distance_s = th.tensor(0.0, requires_grad=True)
        for kernel_name, lengthscale, variance in zip(self.kernel_names, self.lengthscales, self.variances):
            K_11 = _K_giver_pro(d1, d1, kernel_name=kernel_name, lengthscale=lengthscale, variance=variance)
            K_22 = _K_giver_pro(d2, d2, kernel_name=kernel_name, lengthscale=lengthscale, variance=variance)
            K_12 = _K_giver_pro(d1, d2, kernel_name=kernel_name, lengthscale=lengthscale, variance=variance)
            distance_s =+ th.mean(K_11+K_22-2*K_12)
        return distance_s

class MMD_c_pro(dist_base_class):
    '''
    MMD_c_pro implements the conditional MMD only.
    It can use 1) different kernels (RBF, Cosine, Exponential), 2) variances, and 3) lengthscales.
    '''
    def __init__(self, kernel_counts=1, kernel_name='RBF', lambda_c=10, bandwidth=[1]):
        super().__init__('MMD_c_pro', is_MMD=True)
        self.kernel_counts=kernel_counts
        self.kernel_names=[kernel_name]*self.kernel_counts
        self.bandwidth=th.tensor(bandwidth)
        self.lambda_c=lambda_c
    def forward(self, d1, d2):
        distance=th.tensor(0.0, requires_grad=True)
        o1 = d1[:,0:2]; s1 = d1[:,2:4]; o2 = d2[:,0:2]; s2 = d2[:,2:4]
        for kernel_name, lengthscale in zip(self.kernel_names, self.bandwidth):
            K1  = _K_giver_pro(s1, s1, kernel_name=kernel_name, lengthscale=lengthscale)
            K1_i= _inverse_giver(K1, self.lambda_c)
            L1  = _K_giver_pro(o1, o1, kernel_name=kernel_name, lengthscale=lengthscale)
            K2  = _K_giver_pro(s2, s2, kernel_name=kernel_name, lengthscale=lengthscale)
            K2_i= _inverse_giver(K2, self.lambda_c)
            L2  = _K_giver_pro(o2, o2, kernel_name=kernel_name, lengthscale=lengthscale)
            K21 = _K_giver_pro(s2, s1, kernel_name=kernel_name, lengthscale=lengthscale)
            L12 = _K_giver_pro(o1, o2, kernel_name=kernel_name, lengthscale=lengthscale)
            distance=+ th.trace(K1@K1_i@L1@K1_i+K2@K2_i@L2@K2_i-2*K21@K1_i@L12@K2_i)
        return distance

class EDF_multi(dist_base_class):
    '''
    Distance based on cumulative multivariate EDFs.
    It can use three different distance calculation methods: KS  - Kolmogorov-Smirnov: Maximal Distance
                                                             CvM - Cramer-vonMises   : RSMD
                                                             AD  - Anderson-Darling  : weighted RSMD
    has the option to add interpolation to regularize data
    '''
    def __init__(self, dist_type, interp_samples = 1, single_perm = False):
        super().__init__('EDF_mlt')
        self.dist_type = dist_type
        self.interp_samples = interp_samples
        self.perm_list = list(itertools.permutations((0,2,1,3)))
        self.perm_ind = 0
        self.single_perm = single_perm
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        self._prep_interp(interp_samples)
            
    def get_name(self):
        return self.name + '(' + self.dist_type + ',' + str(self.interp_samples) + 's)'
        
    def set_main_par(self, par):
        self._prep_interp(par)
        
    def forward(self, d1, d2):
        #join arrays with two counting columns
        nSamp1 = d1.shape[0]
        nSamp2 = d2.shape[0]
        dM = th.cat((th.cat((th.ones(nSamp1,1,requires_grad=True)/nSamp1,th.zeros(nSamp1,1,requires_grad=True),d1),axis=1),th.cat((th.zeros(nSamp2,1,requires_grad=True),th.ones(nSamp2,1,requires_grad=True)/nSamp2,d2),axis=1)),axis=0)
        if self.single_perm:
            cums, coords = self._calc_cums(dM,self.perm_list[self.perm_ind])
            self.perm_ind += 1
            self.perm_ind = self.perm_ind % 24 #there are 24 permutations in total
            return self._calc_dist(cums,coords)
        dist_list = th.zeros(len(self.perm_list))
        for ind,perm in enumerate(self.perm_list):
            cums, coords = self._calc_cums(dM,perm)
            dist_list[ind] = self._calc_dist(cums,coords)
        return th.sqrt(th.sum(dist_list**2))
    
    def _calc_cums(self, dM, perm):
        #sort columns cascadingly according to permutation
        dM = dM[th.argsort(dM[:,perm[0]+2]),:] # first unstable sort
        for d in range(1,4): #subsequent stable sorts
            dM = dM[uFun.stable_argsort(dM[:,perm[d]+2]),:]
        coords = dM[:,2:] #required for potential interpolation later
        dSort = dM[:,0:2]
        #calculate cumulants
        return th.cumsum(dSort,dim=0), coords
            
    def _calc_dist(self, cums, coords):
        if self.dist_type == 'KS': #Kolmogorov-Smirnov
            return th.max(th.abs(cums[:,0] - cums[:,1]))
        elif self.dist_type == 'CvM': #Cramer-vanMises
            if self.do_interp:
                self._interpolate(cums,coords) #interpolate
            return th.sqrt(th.mean((cums[:,0] - cums[:,1])**2))
        elif self.dist_type == 'AD': #Anderson-Darling
            if self.do_interp:
                self._interpolate(cums,coords) #interpolate
            cums = cums[:-1,:] #ditch the last value, which is by construction 1 for both EDFs (first value is never 0)
            cums_mean = th.mean(cums,dim=1)
            wg = cums_mean * (1 - cums_mean)
            return th.sqrt(th.mean((cums[:,0] - cums[:,1])**2 / wg))
        #else:
        _print_EDF_warning_text('EDF_multi')
        self.dist_type = 'KS'
        return self._calc_dist(cums)
       
    def _prep_interp(self, interp_samples):
        if interp_samples >= 2:
            self.do_interp = True
            self.interp_grid = self._create_interp_grid(interp_samples)                
        else:
            self.do_interp = False
    
    def _interpolate(self, cums, coords):
        #convert data
        cums_np = cums.detach().numpy()
        crds_np = coords.detach().numpy()
        resamp = np.zeros([self.interp_samples**4,2])
        resamp[:,0] = sp.interpolate.griddata(crds_np,cums_np[:,0],self.interp_grid,method='linear',rescale=False)
        resamp[:,1] = sp.interpolate.griddata(crds_np,cums_np[:,1],self.interp_grid,method='linear',rescale=False)
        return th.from_numpy(resamp)
    
    def _create_interp_grid(self, interp_samples):
        self.interp_samples = interp_samples
        grid = np.zeros([interp_samples**4,4])
        #calculate ranges such as to exclude edge points
        f_obs = (self.range_obs[1] - self.range_obs[0]) / interp_samples
        f_set = (self.range_set[1] - self.range_set[0]) / interp_samples
        #create linear chains
        ls_obs = np.linspace(self.range_obs[0]+f_obs/2,self.range_obs[1]-f_obs/2,num=interp_samples)
        ls_set = np.linspace(self.range_set[0]+f_set/2,self.range_set[1]-f_set/2,num=interp_samples)
        ind = 0
        for o1 in range(interp_samples):
            for o2 in range(interp_samples):
                for s1 in range(interp_samples):
                    for s2 in range(interp_samples):
                        grid[ind,:] = np.array([ls_obs[o1],ls_obs[o2],ls_set[s1],ls_set[s2]])
                        ind += 1
        return grid

class EDF_Marg(dist_base_class): #version with interpolation
    '''
    A distance which considers only the distance between the marginals
    according to 1D EDF based distances with the same options as EDF_Multi
    '''
    def __init__(self, dist_type, interp_samples = 1):
        super().__init__('EDF_Mrg',is_Marg=True)
        self.dist_type = dist_type
        self.interp_samples = interp_samples
        #[CONFIG]
        self.range_obs = [-5,5]
        self.range_set = [0,1]
        
    def get_name(self):
        return self.name + '(' + self.dist_type + ',' + str(self.interp_samples) + 's)'
        
    def forward(self, d1, d2):
        dist_list = th.zeros(4)
        for v in range(4):
            cums, coords = self._calc_cums(d1[:,[v]],d2[:,[v]])
            if v < 2:
                dist_list[v] = self._calc_dist(cums,coords,False)
            else:
                dist_list[v] = self._calc_dist(cums,coords,True)
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
            
    def _calc_dist(self, cums, coords, is_setting):
        if self.dist_type == 'KS': #Kolmogorov-Smirnov
            #no interpolation necessary
            return th.max(th.abs(cums[:,0] - cums[:,1]))
        elif self.dist_type == 'CvM': #Cramer-vanMises
            self._interpolate(cums,coords,is_setting) #interpolate
            return th.sqrt(th.mean((cums[:,0] - cums[:,1])**2))
        elif self.dist_type == 'AD': #Anderson-Darling
            self._interpolate(cums,coords,is_setting) #interpolate
            cums = cums[:-1,:] #ditch the last value, which is by construction 1 for both EDFs (first value is never 0)
            cums_mean = th.mean(cums,dim=1)
            wg = cums_mean * (1 - cums_mean)
            return th.sqrt(th.mean((cums[:,0] - cums[:,1])**2 / wg))
        #else:
        #_print_EDF_warning_text('EDF_Marg')
        self.dist_type = 'KS'
        return self._calc_dist(cums,is_setting)
    
    def _interpolate(self, cums, coords, is_setting):
        if self.interp_samples < 2:
            #no resampling
            return cums
        if is_setting:
            ran = self.range_set
        else:
            ran = self.range_obs
        coords_new = th.linspace(ran[0],ran[1],self.interp_samples)
        resamp = th.zeros([self.interp_samples,2])
        resamp[:,0] = Interp1d()(coords,cums[:,0],coords_new)
        resamp[:,1] = Interp1d()(coords,cums[:,1],coords_new)
        return resamp

#auxiliary
##############################################################################
def _inverse_giver(M, Lambda):
    inv_hat=th.linalg.inv(M+Lambda*th.eye(len(M)))
    return(inv_hat)

def _K_giver_pro(a, b, kernel_name, **kwargs):
    input_dim = a.shape[1]
    if kernel_name == 'Cosine':
        kernel = Cosine(input_dim=input_dim, **kwargs)
    elif kernel_name == 'Exponential':
        kernel = Exponential(input_dim=input_dim, **kwargs)
    elif kernel_name == 'RBF':
        kernel = RBF(input_dim=input_dim, **kwargs) 
    K_ab = kernel.forward(a,b)
    return K_ab

def _print_EDF_warning_text(fun_str):
    print('WARNING: invalid option chosen for ' + fun_str + '!')
    print('Options are: "KS" : Kolmogorov-Smirnov\n')
    print('             "CvM": Cramer-vonMises\n')
    print('             "AD" : Anderson-Darling\n')
    print('choosing Kolmogorov-Smirnov by default...')