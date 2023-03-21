import numpy as np
import torch as th
import matplotlib.pyplot as plt
import distance_functions as distFun
import pandas as pd

def plotLossEvolutions(lossEvs, numCand, currentDistStr):
    fig, axs = plt.subplots(2, 2); #not checking numCand
    
    #fig.title('optimization via "' + currentDistStr + '"');
    
    for k in range(numCand):
        col = k % 2;
        row = int(np.floor(k/2));
        axs[row,col].plot(lossEvs[k,:,0]);
        axs[row,col].plot(lossEvs[k,:,1]);
        axs[row,col].plot(lossEvs[k,:,2]);
        axs[row,col].legend(['old','tot','corr']);
        axs[row,col].set_title('c' + str(k));
    
    fig, axs = plt.subplots(2, 2); #not checking numCand
    
    #fig.title('optimization via "' + currentDistStr + '"');
    
    for k in range(numCand):
        col = k % 2;
        row = int(np.floor(k/2));
        axs[row,col].plot(lossEvs[k,:,0] / lossEvs[k,0,0]);
        axs[row,col].plot(lossEvs[k,:,1] / lossEvs[k,0,1]);
        axs[row,col].plot(lossEvs[k,:,2] / lossEvs[k,0,2]);
        axs[row,col].legend(['old','tot','corr']);
        axs[row,col].set_title('c' + str(k));
        
def plotMarginals(genData, dataHigh, dataLow, numCand, numBins):
    varList = ['oA','oB','sA','sB'];
    for j in range(4):
        fig, axs = plt.subplots(2, 3); #not checking numCand
        axs[0,0].hist(dataHigh[:,j], bins=numBins);
        axs[1,0].hist(dataLow[:,j], bins=numBins);
        axs[0,0].set_title(varList[j] + ' - high');
        axs[1,0].set_title('low');
        for k in range(numCand):
            col = k % 2 + 1;
            row = int(np.floor(k/2));
            axs[row,col].hist(genData[k,:,j], bins=numBins);
            axs[row,col].set_title('c' + str(k));

##############################################################################

#config
numCand = 4;
currentDistStr = 'tot';
sampling_rate = 4;
numBins = 15; #for plotting of marginal distributions

#setup distance functions
#"rbf": bandwidths = th.Tensor([0.01,0.1,1])
#"multiscale": bandwidths = th.Tensor([0.2, 0.5, 0.9, 1.3])
dist_MMD_old = distFun.MMD_old(kernel='rbf',
                               bandwidths= th.Tensor([0.9]));
dist_MMD_tot = distFun.MMD_total(bw_m=th.Tensor([1,5,10]), #[1, 5, 10]
                 bw_c=th.Tensor([2]), #[2]
                 l_c=100, #100
                 l_t=10, #10
                 only_standard=False);
dist_CorrD = distFun.CorrD(sampling_rate = 3,
                           num_std_moments = 4,
                           wgCorr = 1,
                           wgMarg = 1);

#load data
testData1 = np.loadtxt('./Debug/lossEvolution_' + currentDistStr + '_0.dat');
testData2 = np.loadtxt('./Debug/genData_' + currentDistStr + '_0_FINAL.dat');
numEpochs = testData1.shape[0];
numDist = testData1.shape[1];
batch_size = testData2.shape[0];
lossEvs = np.zeros([numCand,numEpochs,numDist]);
genData = np.zeros([numCand,batch_size,4]);
for k in range(numCand):
    lossEvs[k,:,:] = np.loadtxt('./Debug/lossEvolution_' + currentDistStr + '_' + str(k) + '.dat');
    genData[k,:,:] = np.loadtxt('./Debug/genData_' + currentDistStr + '_' + str(k) + '_FINAL.dat');

#plot evolutions
plotLossEvolutions(lossEvs,numCand,currentDistStr);

#load sim data
dataHigh = pd.read_csv('./Debug/tweakedSimData/high_m.csv');
dataHigh2 = pd.read_csv('./Debug/tweakedSimData/high_m_2.csv');
dataMed = pd.read_csv('./Debug/tweakedSimData/med_m.csv');
dataLow = pd.read_csv('./Debug/tweakedSimData/low_m.csv');

#reduce amount of sim samples
dataHigh = dataHigh.sample(n=batch_size);
dataHigh2 = dataHigh2.sample(n=batch_size);
dataMed = dataMed.sample(n=batch_size);
dataLow = dataLow.sample(n=batch_size);

#plot marginals
plotMarginals(genData,dataHigh.values,dataLow.values,numCand,numBins);

#convert sim data to tensors
dataHigh = th.tensor(dataHigh.values);
dataHigh2 = th.tensor(dataHigh2.values);
dataMed = th.tensor(dataMed.values);
dataLow = th.tensor(dataLow.values);

#plot qualification of final data sets
fig, axs = plt.subplots(2, 2);
#old
legendList = ['distLow','distMed','distHigh'];
axs[0,0].axhline(y=dist_MMD_old.forward(dataHigh,dataLow).item(), c='g')
axs[0,0].axhline(y=dist_MMD_old.forward(dataHigh,dataMed).item(), c='r')
axs[0,0].axhline(y=dist_MMD_old.forward(dataHigh,dataHigh2).item())
for k in range(numCand):
    axs[0,0].bar(k,dist_MMD_old.forward(dataHigh,th.from_numpy(genData[k,:,:])).item());
    legendList.append('cand-' + str(k));
#axs[0,0].legend(legendList);
axs[0,0].set_title('MMD_OLD');

#tot
legendList = ['distLow','distMed','distHigh'];
axs[0,1].axhline(y=dist_MMD_tot.forward(dataHigh,dataLow).item(), c='g')
axs[0,1].axhline(y=dist_MMD_tot.forward(dataHigh,dataMed).item(), c='r')
axs[0,1].axhline(y=dist_MMD_tot.forward(dataHigh,dataHigh2).item())
for k in range(numCand):
    axs[0,1].bar(k,dist_MMD_tot.forward(dataHigh,th.from_numpy(genData[k,:,:])).item());
    legendList.append('cand-' + str(k));
#axs[0,1].legend(legendList);
axs[0,1].set_title('MMD_TOT');

#corr
dist_CorrD.varying_std_mode = True;
legendList = ['distLow','distMed','distHigh'];
axs[1,0].axhline(y=dist_CorrD.forward(dataHigh,dataLow).item(), c='g')
axs[1,0].axhline(y=dist_CorrD.forward(dataHigh,dataMed).item(), c='r')
axs[1,0].axhline(y=dist_CorrD.forward(dataHigh,dataHigh2).item())
for k in range(numCand):
    axs[1,0].bar(k,dist_CorrD.forward(dataHigh,th.from_numpy(genData[k,:,:])).item());
    legendList.append('cand-' + str(k));
#axs[1,0].legend(legendList);
axs[1,0].set_title('CorrD');

corrsObs,corrSet,samples,_ = distFun.detCorrs(th.from_numpy(genData[3,:,:]),sampling_rate);
print(corrsObs);
print(corrSet);
print(samples);