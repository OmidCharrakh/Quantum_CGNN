from distances import CorrD
from distances import MMD_cdt
#from distances import MMD_c_pro

import torchvision
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

import torch as th; import torch; from torch.utils.data import DataLoader; import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import random
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


class diff_block(torch.nn.Module):
    def __init__(self, bandwidth, sampling_rate=6, corr_types=[1,1]):
        super(diff_block, self).__init__()
        self.bandwidth  = nn.Parameter(th.Tensor([bandwidth]).reshape((1,1))); 
        self.criterion1 = MMD_cdt(bandwidth=self.bandwidth)
        self.criterion2 = CorrD(sampling_rate=sampling_rate, corr_types=corr_types)
        
    def forward(self, d1, d2):
        distance1=self.criterion1(d1, d2)/self.calib1
        distance2=self.criterion2(d1, d2)/self.calib2
        diff=th.abs(distance1-distance2)
        return distance1, distance2, diff
    def _calib_computer(self, calib_dataset, batch_size, num_run):
        calib_loader1 = DataLoader(calib_dataset, batch_size=batch_size, shuffle=True, drop_last=True);
        calib_loader2 = DataLoader(calib_dataset, batch_size=batch_size, shuffle=True, drop_last=True);
        calib1=th.tensor(0.0, requires_grad=True); calib2=th.tensor(0.0, requires_grad=False)
        r=0
        for run in range(num_run):
            for (c1, c2) in zip(calib_loader1, calib_loader2):
                calib1=calib1+self.criterion1(c1, c2); calib2=calib2+self.criterion2(c1, c2); r=r+1
        self.calib1=calib1/r
        self.calib2=calib2/r
        return(self.calib1,self.calib2)

def diff_optimizer(config, checkpoint_dir, datasets, batch_size, train_epochs, sample_size):
    
    
    diff_model= diff_block(sampling_rate=6, corr_types=[1,1], bandwidth=config["bandwidth"])
    optimizer = th.optim.Adam(diff_model.parameters(), lr=.01);

    try:
        model_state, optimizer_state = th.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    except:
        pass
    
    calib_dataset, train_dataset, val_dataset=datasets
    calib1, calib2=diff_model._calib_computer(calib_dataset=calib_dataset, batch_size=batch_size, num_run=2)
    
    train_loader1 = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True, drop_last=True); 
    train_loader2 = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True, drop_last=True);
    val_loader1   = DataLoader(val_dataset,    batch_size=batch_size, shuffle=True, drop_last=True); 
    val_loader2   = DataLoader(val_dataset,    batch_size=batch_size, shuffle=True, drop_last=True);
    
    #print(calib1, calib2)

    for epoch in range(train_epochs):
        for (d1, d2) in zip(train_loader1, train_loader2):
            optimizer.zero_grad()
            distance1, distance2, diff= diff_model.forward(d1, d2)
            diff.backward(retain_graph=True)
            optimizer.step()
        distance1_lst=[]; distance2_lst=[]; diff_lst=[];
        for (d1, d2) in zip(val_loader1,val_loader2):
            distance1, distance2, diff=diff_model.forward(d1, d2)
            distance1_lst.append(distance1.item());distance2_lst.append(distance2.item()); diff_lst.append(diff.item()); 
        tune.report(distance_1=np.mean(distance1_lst), distance_2=np.mean(distance2_lst), diff=np.mean(diff_lst))
            
        with tune.checkpoint_dir(epoch) as checkpoint_dir: 
            torch.save((diff_model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))
    print("Finished Training")

########################################
scheduler_paitiance=5
num_samples=100
batch_size=4000
train_epochs=50
sample_size=4000
checkpoint_dir='/Users/omid/Documents/GitHub/Causality/CGNN_EPR/Debug/Extra/phase3/'

config = {
    'bandwidth': tune.loguniform(1e-2, 1e+2)}


scheduler = ASHAScheduler(metric='diff', mode='min', max_t=scheduler_paitiance) 
#scheduler=None

search_alg= HyperOptSearch(metric="diff", mode="min")
#search_alg=BayesOptSearch(metric="diff", mode="min")

calib_dataset = th.Tensor(pd.read_csv('./Data/noCorr_c.csv').sample(sample_size).values)
train_dataset = th.Tensor(pd.read_csv('./Data/dat_train.csv').sample(sample_size).values)
val_dataset   = th.Tensor(pd.read_csv('./Data/dat_val.csv').sample(sample_size).values)
datasets=(calib_dataset, train_dataset, val_dataset)


analysis = tune.run(
    tune.with_parameters(diff_optimizer,
                         datasets=datasets,
                         checkpoint_dir=checkpoint_dir, 
                         train_epochs=train_epochs,
                         batch_size=batch_size, 
                         sample_size=sample_size), 
    config=config, 
    num_samples=num_samples, 
    search_alg=search_alg,
    scheduler=scheduler, 
    verbose=1,
    local_dir="./Extra/phase3")

df=analysis.results_df
df.sort_values(by=['diff'], ascending=True, inplace=True)
df
