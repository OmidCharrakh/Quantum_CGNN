import numpy as np
import matplotlib.pyplot as plt
import torch as th
import pandas as pd
import networkx as nx
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
import ast
import time
from tqdm import trange
from random import sample
from sklearn.model_selection import train_test_split
import torch
import pickle
import distance_functions_omid as dFun

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

import random
from torch.utils.tensorboard import SummaryWriter
os.environ["TUNE_RESULT_DELIM"] = "/"



class CGNN_block(th.nn.Module):
    def __init__(self, sizes):
        super(CGNN_block, self).__init__()
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class CGNN_model(th.nn.Module):
    def __init__(self, adjacency_matrix, batch_size, nh, lr, train_epochs, L1_w, L2_w, guassian_w, uniform_w): 
        super(CGNN_model, self).__init__()
        self.train_epochs=train_epochs
        self.adjacency_matrix = adjacency_matrix
        self.batch_size = batch_size
        self.nh=nh
        self.lr=lr
        self.L1_w=L1_w
        self.L2_w=L2_w
        self.guassian_w=guassian_w 
        self.uniform_w=uniform_w
        self.topological_order = [i for i in nx.topological_sort(nx.DiGraph(self.adjacency_matrix))]
        self.generated = [None for i in range(self.adjacency_matrix.shape[0])]
        self.blocks = th.nn.ModuleList()
        for i in range(self.adjacency_matrix.shape[0]):
            sizes=[int(self.adjacency_matrix[:, i].sum()) + 1, self.nh, 1]
            self.blocks.append(CGNN_block(sizes))
            
    def generator(self, gen_size):
        guassian_noise=th.zeros(gen_size, self.adjacency_matrix.shape[0]).normal_(0, 1)
        uniform_noise= th.zeros(gen_size, self.adjacency_matrix.shape[0]).uniform_(0, 1)
        noise=(self.guassian_w*guassian_noise+self.uniform_w*uniform_noise)/(self.guassian_w+self.uniform_w)
        for i in self.topological_order:
            self.generated[i] = self.blocks[i](th.cat([v for c in [[self.generated[j] for j in np.nonzero(self.adjacency_matrix[:, i])[0]],[noise[:, [i]]]] for v in c],axis=1))
        return th.cat(self.generated, axis=1)


def train_cgnn(config, checkpoint_dir='/Users/omid/Documents/GitHub/Causality/CGNN_EPR/Debug/Extra/', 
               L1_w=1e-3, L2_w=1e-3, batch_size=500, nh=20, lr=.01, 
               train_epochs=200, guassian_w=1, uniform_w=0, adjacency_matrix=np.reshape(np.loadtxt('./Data/candidates_debug.txt')[1], (4,4))):
    
    
    
    model=CGNN_model(adjacency_matrix=adjacency_matrix, 
                     batch_size=batch_size, 
                     nh=nh, 
                     lr=lr, 
                     train_epochs=train_epochs, 
                     L1_w=L1_w, 
                     L2_w=L2_w, 
                     guassian_w=guassian_w, 
                     uniform_w=uniform_w)
    
    kernel_names, lengthscales, variances = config["kernel_iter"]
    
    criterion_prim = dFun.MMD_pro(kernel_names=kernel_names, 
                                  variances=variances,
                                  lengthscales=lengthscales, 
                                  lambda_c=config["lambda_c"],  
                                  wg_mmd=1, 
                                  wg_corrD=0,
                                  wg_mmd_s=config["wg_mmd_s"], 
                                  wg_mmd_c=config["wg_mmd_c"],)
    
    criterion_supp = dFun.MMD_pro(sampling_rate=4, 
                                 num_std_moments=4,
                                 wg_mmd=0, 
                                 wg_corrD=1, 
                                 wg_corrD_m=1, 
                                 wg_corrD_c=1)
    
    optimizer = th.optim.Adam(model.parameters(), lr=lr);
    try:
        model_state, optimizer_state = th.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    except:
        pass
    
    train_path='/Users/omid/Documents/GitHub/Causality/CGNN_EPR/Debug/Data/train.csv' 
    val_path='/Users/omid/Documents/GitHub/Causality/CGNN_EPR/Debug/Data/val.csv'
    
    train_dataset=th.Tensor(pd.read_csv(train_path).sample(n=train_sample_size).values)
    val_dataset  =th.Tensor(pd.read_csv(val_path).sample(n=train_sample_size).values)
    datasets=(train_dataset, val_dataset)
    
    train_dataset, val_dataset= datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, drop_last=True)
    e_verbose=trange(train_epochs, disable=True)
    
    for epoch in e_verbose:
        for inx_train, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            gen_data=model.generator(batch_size)
            loss= criterion_prim(gen_data, train_data)
            L1_reg = sum(p.abs().sum() for p in model.parameters())
            L2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            train_loss = loss + L1_w * L1_reg+ L2_w * L2_reg
            train_loss.backward()
            optimizer.step()
        loss_prim=[]; loss_supp=[]; calib_prim=[]
        for inx_val, val_data in enumerate(val_loader):
            gen_data=model.generator(batch_size)
            loss = criterion_prim(gen_data, val_data)
            L1_reg = sum(p.abs().sum() for p in model.parameters())
            L2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            val_loss = loss + L1_w * L1_reg+ L2_w * L2_reg
            loss_prim.append(val_loss.item())
            loss_supp.append(criterion_supp(gen_data, val_data).item())
            calib_prim.append(criterion_prim(train_data, val_data).item())
            
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))
            loss_prim = np.mean(loss_prim)
            loss_supp = np.mean(loss_supp)
            calib_prim= np.mean(calib_prim)
            qual_prim = calib_prim/loss_prim
            
        tune.report(loss_prim=loss_prim, qual_prim=qual_prim, loss_supp = loss_supp, calib_prim=calib_prim)
    print("Finished Training")
        
    
def kernel_iter(pop_counts=[7,8,9,10], pop_names=['RBF', 'Cosine', 'Exponential'], pop_lengthscales=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3]):
    for kernel_counts in pop_counts:
        for kernel_names in [random.choices(population=pop_names, k=kernel_counts) for i in range(20)]:
            for lengthscales in [random.choices(population=pop_lengthscales, k=kernel_counts) for i in range(20)]:
                for variances in [[10*random.random() for k in range(kernel_counts)] for i in range(20)]:
                    yield(kernel_names, lengthscales, variances)
                    

                    
##############################################
from ray.tune.suggest.hebo import HEBOSearch
                    
num_kernel_iters=5
num_samples=5
train_sample_size=10000
max_epochs=20

kernel_iters=random.choices(population=list(kernel_iter()), k=num_kernel_iters)



config = {"kernel_iter": tune.grid_search(kernel_iters), 
          "wg_mmd_s": tune.uniform(1e-9, 1),
          "lambda_c": tune.choice([10,100, 300]),
          "wg_mmd_c": tune.choice([0])}

scheduler = ASHAScheduler(metric="loss_prim", mode="min", max_t=max_epochs)

search_alg = HEBOSearch(metric="loss_prim", mode="min")



result = tune.run(train_cgnn,
                  config=config, 
                  num_samples=num_samples, 
                  search_alg=search_alg,
                  scheduler=scheduler, 
                  verbose=0, 
                  local_dir="./Extra/")

df=result.results_df



