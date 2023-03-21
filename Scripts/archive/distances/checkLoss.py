import numpy as np
import matplotlib.pyplot as plt
import torch as th

def corrD(dist1,dist2):
    corrs1,_,_,_ = detCorrs(dist1);
    corrs2,_,_,_ = detCorrs(dist2);
    return np.sqrt(np.mean((corrs1[1:5,1:5] - corrs2[1:5,1:5])**2));

def detCorrs(dist):
    #create correlation matrix
    minVal = -np.sqrt(np.pi);
    maxVal = np.sqrt(np.pi);
    step = np.sqrt(np.pi)/2;
    corrs = np.zeros([6,6]);
    meansA = np.zeros([6,6]);
    meansB = np.zeros([6,6]);
    samples = np.zeros([6,6]);
    for k in range(6):
        if k == 0:
            linesA = dist[:,2] < minVal;
        elif k == 5:
            linesA = dist[:,2] >= maxVal;
        else:
            linesA = np.logical_and(dist[:,2] >= (k-1)*step+minVal,dist[:,2] < k*step+minVal);
        
        for j in range (6):
            if j == 0:
                linesB = dist[:,3] < minVal;
            elif j == 5:
                linesB = dist[:,3] >= maxVal;
            else:
                linesB = np.logical_and(dist[:,3] >= (j-1)*step+minVal,dist[:,3] < j*step+minVal);
            
            pick = dist[np.logical_and(linesA,linesB),0:2];
            samples[k,j] = pick.shape[0];
            if samples[k,j] == 0:
                meansA[k,j] = 0;
                meansB[k,j] = 0;
                corrs[k,j] = 0;
            else:
                meansA[k,j] = np.mean(pick[:,0]);
                meansB[k,j] = np.mean(pick[:,1]);
                corrs[k,j] = np.mean((pick[:,0] - meansA[k,j]) * (pick[:,1]-meansB[k,j])) / (np.std(pick[:,0]) * np.std(pick[:,1]));      
                
    return corrs, meansA, meansB, samples;

#config
ep = 30;
cand = 4;

#load data
lossMMD = np.zeros([cand,ep]);
dataSim = np.zeros([cand,ep,1000,4]);
dataGen = np.zeros([cand,ep,1000,4]);

for k in range(cand):
    lossMMD[k,:] = np.loadtxt("./Debug/mmd-" + str(k) + ".dat");
    for j in range(ep):
        dataSim[k,j,:,:] = np.loadtxt("./Debug/sim-" + str(k) + "-" + str(j) + ".dat");
        dataGen[k,j,:,:] = np.loadtxt("./Debug/gen-" + str(k) + "-" + str(j) + ".dat");

#unify chunks of simulated distribution
totDataSim = np.array([]);
for k in range(cand):
    for j in range(ep):
        if totDataSim.size:
            totDataSim = np.vstack([totDataSim,dataSim[k,j,:,:]]);
        else:
            totDataSim = dataSim[k,j,:,:];        
            
#show correlation structure of simulated data
corrs,meansA,meansB,samples = detCorrs(totDataSim);
print("\ncorrelation structure of simulation:");
#print(np.around(meansA,decimals=2));
#print(np.around(meansB,decimals=2));
print(np.around(corrs,decimals=2));
# print(samples);
# print(np.sum(samples));

#calculate distances between simulated distributions and total distribution
simLossCD = np.zeros([cand,ep]);
for k in range(cand):
    for j in range(ep):
        simLossCD[k,j] = corrD(totDataSim,dataSim[k,j,:,:]);

print("\ndistances(of sims) to total simulated data");
print("dists:\n" + str(np.around(simLossCD,decimals=2)));
print("mean: " + str(np.around(np.mean(simLossCD),decimals=2)));
print("std: " + str(np.around(np.std(simLossCD),decimals=2)));

#calculate distances between generated distributions and total simulated distribution        
lossCD = np.zeros([cand,ep]);
for k in range(cand):
    for j in range(ep):
        lossCD[k,j] = corrD(totDataSim,dataGen[k,j,:,:]);
        
f = plt.figure();
plt.plot(lossCD[0,:]);
plt.plot(lossCD[1,:]);
plt.plot(lossCD[2,:]);
plt.plot(lossCD[3,:]);
plt.plot(lossMMD[0,:]);
plt.plot(lossMMD[1,:]);
plt.plot(lossMMD[2,:]);
plt.plot(lossMMD[3,:]);
plt.legend(["0-CD","1-CD","2-CD","3-CD","0-MMD","1-MMD","2-MMD","3-MMD"]);
f.savefig("./Debug/fig/lossEv.pdf",bbox_inches='tight');

#show correlation structure of good generated data
corrs,meansA,meansB,samples = detCorrs(dataGen[2,29,:,:]);
print("\ncorrelation structure of generated data (cand 2, epoch 30):");
#print(np.around(meansA,decimals=2));
#print(np.around(meansB,decimals=2));
print(np.around(corrs,decimals=2));
# print(samples);
# print(np.sum(samples));