import torch as th
import numpy as np

#AUX:
##############################################################################

#aux - MMD_total
def K_giver_t(a, b, Gamma):
    n_1, n_2 = a.size(0), b.size(0)
    #norm = 2. #unused
    norms_1 = th.sum(a**2, dim=1, keepdim=True)
    norms_2 = th.sum(b**2, dim=1, keepdim=True)
    norms = (norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2))
    distance_matrix = th.abs(norms - 2 * a.mm(b.t()))
    K_ab= th.zeros((n_1, n_2)) 
    for g in Gamma: 
        K_ab=K_ab+th.exp(-distance_matrix* g)
    return K_ab

def inverse_giver_t(M, Lambda):
    inv_hat=th.linalg.inv(M+Lambda*th.eye(len(M)))
    return(inv_hat)

#aux - CorrD
def detCorrs(dstr, sampling_rate):
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

def detMarginals(dstr, num_std_moments):
    #0: mean
    #1: std
    #2: skewness
    #3: kurtosis
    
    if num_std_moments < 2:
        print('ERROR: detMarginals needs at least 2 std moments');
        num_std_moments = 2;
    moments = th.zeros([num_std_moments,4]);
    
    for k in range(4):
        curData = dstr[:,k];
        curMean = th.mean(curData);
        curStd = th.std(curData);
        moments[0,k] = curMean;
        moments[1,k] = curStd;
        for j in range(num_std_moments-2):
            moments[j+2,k] = th.mean((curData - curMean)**(j+3)) / curStd**(j+3);
    
    return moments;

def batchSizeEstimator(sampling_rate,currentBatchSize,targetPZero):
    #assumes that there are no anomalous settings
    #assumes that the settings will be uniformly distributed
    opt = sampling_rate**2; 
    #calculate current pZero
    expVal = currentBatchSize / opt; #expectation value of settings per site
    currentP = np.exp(-expVal); #Poissonian probability to get zero settings at a site
    #calculate target batch size
    targetB = - opt * np.log(targetPZero); #b is batch_size
    return targetB,currentP;

#DISTANCES:
##############################################################################

class MMD_total(th.nn.Module):
    '''
    A Loss Function computing the standard and conditional MMD distances between two datasets; Example:
    Loss=MMD_total(bw_m=th.Tensor([.2, 1]), bw_c=th.Tensor([.2, 1]), l_c=100, l_t=1, only_standard=False)  
    Loss(th.from_numpy(d1), th.from_numpy(d2))
    '''
    def __init__(self, 
                 bw_m=th.Tensor([1, 5, 10]), 
                 bw_c=th.Tensor([2]), 
                 l_c=100, 
                 l_t=10, 
                 only_standard=False):
        super(MMD_total, self).__init__()
        self.bw_m= bw_m
        self.bw_c= bw_c
        self.l_c = l_c
        self.l_t = l_t
        self.only_standard = only_standard
    
    def forward(self, d1, d2):
        # standard MMD distance
        K_11 = K_giver_t(d1, d1, Gamma=self.bw_m)
        K_22 = K_giver_t(d2, d2, Gamma=self.bw_m)
        K_12 = K_giver_t(d1, d2, Gamma=self.bw_m)
        distance_m=th.mean(K_11) + th.mean(K_22)- 2*th.mean(K_12)
        if self.only_standard or self.l_t == 0:
            return distance_m
        else:
            # condtional distance
            o1 = d1[:,0:2]; s1 = d1[:,2:4]
            o2 = d2[:,0:2]; s2 = d2[:,2:4]
            K1  = K_giver_t(s1, s1, Gamma=self.bw_c)
            K1_i= inverse_giver_t(K1, self.l_c)
            L1  = K_giver_t(o1, o1, Gamma=self.bw_c)
            K2  = K_giver_t(s2, s2, Gamma=self.bw_c)
            K2_i= inverse_giver_t(K2, self.l_c)
            L2  = K_giver_t(o2, o2, Gamma=self.bw_c)
            K21 = K_giver_t(s2, s1, Gamma=self.bw_c)
            L12 = K_giver_t(o1, o2, Gamma=self.bw_c)
            distance_c= th.trace(K1@K1_i@L1@K1_i) + th.trace(K2@K2_i@L2@K2_i)-2*th.trace(K21@K1_i@L12@K2_i)
            # total distance
            distance_t = (distance_m + self.l_t * distance_c) / self.l_t;
            return distance_t 

class MMD_old(th.nn.Module):
    def __init__(self, kernel='multiscale', bandwidths= th.Tensor([5])):
        super(MMD_old, self).__init__()
        self.kernel = kernel
        self.bandwidths = bandwidths
        
    def forward(self, x, y):
        xx, yy, zz = th.mm(x, x.t()), th.mm(y, y.t()), th.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        dxx = rx.t() + rx - 2. * xx 
        dyy = ry.t() + ry - 2. * yy 
        dxy = rx.t() + ry - 2. * zz 
        XX, YY, XY = (th.zeros(xx.shape), th.zeros(xx.shape), th.zeros(xx.shape))
        if self.kernel == "multiscale":
            bandwidths = th.Tensor([0.2, 0.5, 0.9, 1.3])
            for a in bandwidths:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
        if self.kernel == "rbf": #[0.01, 0.1, 1, 10, 100]
            bandwidths = self.bandwidths #or something like [10, 15, 20, 50]
            for a in bandwidths:
                XX += th.exp(-0.5*dxx/a)
                YY += th.exp(-0.5*dyy/a)
                XY += th.exp(-0.5*dxy/a)
        return th.mean(XX + YY - 2. * XY)
    
class CorrD(th.nn.Module):
    def __init__(self, sampling_rate = 3, num_std_moments = 2, wgCorr = 1, wgMarg = 0):
        super(CorrD, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_std_moments = num_std_moments
        self.wgCorr = wgCorr;
        self.wgMarg = wgMarg;
    
    def forward(self, d1, d2):
        if self.wgCorr == 0 and self.wgMarg == 0:
            print('ERROR: Both weights for corrD are zero');
            return -1;
        else:
            wgCorr = self.wgCorr;
            wgMarg = self.wgMarg;
        
        #correlations    
        if wgCorr > 0:
            corrsObs1,corrSet1,samples1,zeroError1 = detCorrs(d1,self.sampling_rate); 
            corrsObs2,corrSet2,samples2,zeroError2 = detCorrs(d2,self.sampling_rate);
            if zeroError1 or zeroError2:
                print('ERROR: encountered setting region with zero samples, either increase batch size or reduce sampling size');
                #diffCorrs = th.tensor(2.0, requires_grad=True); #maximally possible value per site is 2
            #else:
                #diffCorrs = th.sqrt(th.mean((corrsObs1[1:self.sampling_rate+1,1:self.sampling_rate+1] - corrsObs2[1:self.sampling_rate+1,1:self.sampling_rate+1])**2 + (corrSet1 - corrSet2)**2));
            #cutout inner region
            corrsObs1 = corrsObs1[1:self.sampling_rate+1,1:self.sampling_rate+1];
            corrsObs2 = corrsObs2[1:self.sampling_rate+1,1:self.sampling_rate+1];
            samples1 = samples1[1:self.sampling_rate+1,1:self.sampling_rate+1];
            samples2 = samples2[1:self.sampling_rate+1,1:self.sampling_rate+1];
            diffCorrs = (corrsObs1 - corrsObs2) * th.sqrt(samples1 * samples2) / ((th.mean(samples1) + th.mean(samples2))/2); #weigh correlation values with amount of samples
            diffCorrs = th.sqrt(th.mean(diffCorrs**2) + (corrSet1 - corrSet2)**2); #add setting correlation
        else:
            diffCorrs = th.tensor(2.0, requires_grad=True);
        
        #marginals
        if wgMarg > 0:
            moments1 = detMarginals(d1,self.num_std_moments); 
            moments2 = detMarginals(d2,self.num_std_moments);
            diffMarg = th.sqrt(th.mean((moments1 - moments2)**2));
        else:
            diffMarg = th.tensor(0.0, requires_grad=True);
            
        #total distance
        totalDist = (wgCorr * diffCorrs + wgMarg * diffMarg) / (wgCorr + wgMarg);
        return totalDist;    
    
