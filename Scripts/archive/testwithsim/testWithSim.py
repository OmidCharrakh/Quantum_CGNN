#script to test distance measures with simulated datasets
import pandas as pd
import distance_functions as distFun
import torch as th

#[CONFIG]
batch_size = 1000;

#load data
dataHigh = pd.read_csv('./Debug/tweakedSimData/high_m.csv');
dataHigh2 = pd.read_csv('./Debug/tweakedSimData/high_m_2.csv');
dataMed = pd.read_csv('./Debug/tweakedSimData/med_m.csv');
dataLow = pd.read_csv('./Debug/tweakedSimData/low_m.csv');

#reduce amount of samples
dataHigh = dataHigh.sample(n=batch_size);
dataHigh2 = dataHigh2.sample(n=batch_size);
dataMed = dataMed.sample(n=batch_size);
dataLow = dataLow.sample(n=batch_size);

#convert data to tensors
dataHigh = th.tensor(dataHigh.values);
dataHigh2 = th.tensor(dataHigh2.values);
dataMed = th.tensor(dataMed.values);
dataLow = th.tensor(dataLow.values);

#setup distance functions
dist_MMD_old = distFun.MMD_old();
dist_MMD_tot = distFun.MMD_total();
dist_CorrD = distFun.CorrD();

#MMD_old:
distOld_toHigh = dist_MMD_old.forward(dataHigh,dataHigh2).item();
distOld_toMed = dist_MMD_old.forward(dataHigh,dataMed).item();
distOld_toLow = dist_MMD_old.forward(dataHigh,dataLow).item();

#MMD_total:
distTot_toHigh = dist_MMD_tot.forward(dataHigh,dataHigh2).item();
distTot_toMed = dist_MMD_tot.forward(dataHigh,dataMed).item();
distTot_toLow = dist_MMD_tot.forward(dataHigh,dataLow).item();

#CorrD:
distCorr_toHigh = dist_CorrD.forward(dataHigh,dataHigh2).item();
distCorr_toMed = dist_CorrD.forward(dataHigh,dataMed).item();
distCorr_toLow = dist_CorrD.forward(dataHigh,dataLow).item();