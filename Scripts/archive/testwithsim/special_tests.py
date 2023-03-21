import pandas as pd
import distance_functions as dFun
import torch as th
from torch.utils.data import DataLoader
import numpy as np
import time
import utilities_functions as uFun
import matplotlib.pyplot as plt

sample_size = 18000
batch_size = 8000

dataHigh = pd.read_csv('./Data/dat_train.csv')
dataHigh2 = pd.read_csv('./Data/dat_calib_high.csv')
dataLow = pd.read_csv('./Data/dat_calib_noCorr.csv')
dataGen = pd.read_csv('./Temp/c0000_best.prd')

dataHigh = th.Tensor(dataHigh.sample(n=sample_size).values)
dataHigh2 = th.Tensor(dataHigh2.sample(n=sample_size).values)
dataLow = th.Tensor(dataLow.sample(n=sample_size).values)
dataGen = th.Tensor(dataGen.sample(n=sample_size).values)

dataloaders = [None for k in range(4)]
dataloaders[0] = DataLoader(dataHigh,batch_size=batch_size,shuffle=True,drop_last=True)
dataloaders[1] = DataLoader(dataHigh2,batch_size=batch_size,shuffle=True,drop_last=True)
dataloaders[2] = DataLoader(dataLow,batch_size=batch_size,shuffle=True,drop_last=True)
dataloaders[3] = DataLoader(dataGen,batch_size=batch_size,shuffle=True,drop_last=True)

dTest = dFun.BinnedD('JS',4)
ref = np.zeros([len(dataloaders[0]),1])
zbm1 = np.zeros([len(dataloaders[0]),1])
zbm2 = np.zeros([len(dataloaders[0]),1])
val = np.zeros([len(dataloaders[0]),1])
val2 = np.zeros([len(dataloaders[0]),1])
for ind,(dH1,dH2,dL,dG) in enumerate(zip(dataloaders[0],dataloaders[1],dataloaders[2],dataloaders[3])):
    zbm1[ind] = dTest.forward(dH1,dL).item()
    zbm2[ind] = dTest.forward(dH2,dL).item()
    ref[ind] = dTest.forward(dH1,dH2).item()
    val[ind] = dTest.forward(dH1,dG).item()
    val2[ind] = dTest.forward(dH2,dG).item()