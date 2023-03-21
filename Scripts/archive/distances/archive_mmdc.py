import torch as th
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from distance_functions import (CorrD, MMD_c_pro)


# Aim: finding an optimal bandwidth for MMD_c_pro such that mmd_c_pro's predictions be close to the the CorrD's predictions 
# Note: this analysis highly depends on the value of the batch_size. 
# Hence, we should first fix the batch_size and then find an optimal bandwidth for MMD_c_pro.

# 1) From each of the 3 calibrated datasets, take 1000 data points 
batch_size=1000
h1_c=th.Tensor(pd.read_csv('./Data/dat_calib_high.csv').sample(batch_size).values)
h2_c=th.Tensor(pd.read_csv('./Data/dat_calib_high.csv').sample(batch_size).values)
m_c=th.Tensor(pd.read_csv('./Data/archive/dat_calib_med.csv.csv').sample(batch_size).values)
l_c=th.Tensor(pd.read_csv('./Data/archive/dat_calib_low.csv.csv').sample(batch_size).values)    


# 2) We want to monitor the behaviour of CorrD with respect to different datasets and its hyperparameter(i.e., sampling rates) => 
# Find the CorrD between different datasets for different sampling rates => take an average  

numIter=100
sampling_rates=[3,4,5,6,7,8]    
dict_0={}    
dict_0['hl']=[np.mean([CorrD(sampling_rate=rate)(h1_c, l_c).item()/CorrD(sampling_rate=rate)(h1_c, h2_c).item() for rate in sampling_rates])]*numIter    
dict_0['hm']=[np.mean([CorrD(sampling_rate=rate)(h1_c, m_c).item()/CorrD(sampling_rate=rate)(h1_c, h2_c).item() for rate in sampling_rates])]*numIter    
dict_0['lm']=[np.mean([CorrD(sampling_rate=rate)(l_c, m_c).item()/CorrD(sampling_rate=rate)(h1_c, h2_c).item() for rate in sampling_rates])]*numIter    
dict_0['hh']=[np.mean([CorrD(sampling_rate=rate)(h1_c, h2_c).item() for rate in sampling_rates])]*numIter    
    
# 3) Find the MMD_c_pro between the same datasets for different bandwidths (if needed, we can also play with lambda_c)
dict_1={'hl':[], 'hm':[],'lm':[], 'hh':[]}
bandwidth_lst=[]

for bandwidth in np.logspace(-2, 2, num=numIter):
    loss_1= MMD_c_pro(bandwidth=[bandwidth], lambda_c=10)  
    calib_1=loss_1(h1_c, h2_c).item()
    dict_1['hl'].append(loss_1(h1_c, l_c).item()/calib_1)
    dict_1['hm'].append(loss_1(h1_c, m_c).item()/calib_1)
    dict_1['lm'].append(loss_1(l_c, m_c).item()/calib_1)
    dict_1['hh'].append(calib_1)
    bandwidth_lst.append(bandwidth)
    
dict_0['hl']=np.array(dict_0['hl']); dict_0['hm']=np.array(dict_0['hm']); dict_0['lm']=np.array(dict_0['lm']); dict_0['hh']=np.array(dict_0['hh'])
dict_1['hl']=np.array(dict_1['hl']); dict_1['hm']=np.array(dict_1['hm']); dict_1['lm']=np.array(dict_1['lm']); dict_1['hh']=np.array(dict_1['hh'])


# 4) Compute the difference between the predictions of CorrD and MMD_c_pro => 
# Find the optimal bandwidth at which the MMD_c_pro can mimic the behaviour of the CorrD
diff_hl=np.abs(dict_0['hl']-dict_1['hl'])
diff_hm=np.abs(dict_0['hm']-dict_1['hm'])
diff_lm=np.abs(dict_0['lm']-dict_1['lm'])
diff_hh=np.abs(dict_0['hh']-dict_1['hh'])
diff_sum=(diff_hl+diff_hm+diff_lm+diff_hh)/4

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
#g0=sns.scatterplot(ax=axes, x=bandwidth_lst, y=diff_hl, label='diff_hl')
#g0=sns.scatterplot(ax=axes, x=bandwidth_lst, y=diff_hm, label='diff_hm')
#g0=sns.scatterplot(ax=axes, x=bandwidth_lst, y=diff_lm, label='diff_lm')
#g0=sns.scatterplot(ax=axes, x=bandwidth_lst, y=diff_hh, label='diff_hh')
g0=sns.scatterplot(ax=axes, x=bandwidth_lst, y=diff_sum, label='diff_sum')

g0.set(xscale="log"); 
plt.show()

print('The least distance between CorrD ans MMD_c_pro happens at \n bw: {} => diff_sum: {}'.format(bandwidth_lst[np.argmin(diff_sum)],diff_sum[np.argmin(diff_sum)]))

