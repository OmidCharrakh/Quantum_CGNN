import numpy as np
import matplotlib.pyplot as plt
import torch as th

def MMD(x, y, kernel='rbf', bandwidths= th.Tensor([0.01, 0.1, 1, 10, 100])):
    xx, yy, zz = th.mm(x, x.t()), th.mm(y, y.t()), th.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2. * xx 
    dyy = ry.t() + ry - 2. * yy 
    dxy = rx.t() + ry - 2. * zz 
    XX, YY, XY = (th.zeros(xx.shape), th.zeros(xx.shape), th.zeros(xx.shape))
    if kernel == "multiscale":
        bandwidths = th.Tensor([0.2, 0.5, 0.9, 1.3])
        for a in bandwidths:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
    if kernel == "rbf":
        bandwidths = bandwidths #or something like [10, 15, 20, 50]
        for a in bandwidths:
            XX += th.exp(-0.5*dxx/a)
            YY += th.exp(-0.5*dyy/a)
            XY += th.exp(-0.5*dxy/a)
    return th.mean(XX + YY - 2. * XY)

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

#load data
dataSim = np.zeros([4,2,1000,4]);
dataGen = np.zeros([4,2,1000,4]);

for k in range(4):
    dataSim[k,0,:,:] = np.loadtxt("./Debug/" + str(k) + "-sim-0.dat");
    dataSim[k,1,:,:] = np.loadtxt("./Debug/" + str(k) + "-sim-1.dat");
    dataGen[k,0,:,:] = np.loadtxt("./Debug/" + str(k) + "-gen-0.dat");
    dataGen[k,1,:,:] = np.loadtxt("./Debug/" + str(k) + "-gen-1.dat");

#calculate pairwise distances between simulated distributions
simCDrel = np.zeros([8,8]);
for k in range(4):
    for j in range(2):
        for l in range(4):
            for m in range(2):
                simCDrel[j*4+k,m*4+l] = corrD(dataSim[k,j,:,:],dataSim[l,m,:,:]);

print("pairwise distances");
print(np.around(simCDrel,decimals=2));

#unify chunks of simulated distribution
totDataSim = np.array([]);
for k in range(4):
    for j in range(2):
        if totDataSim.size:
            totDataSim = np.vstack([totDataSim,dataSim[k,j,:,:]]);
        else:
            totDataSim = dataSim[k,j,:,:];
            
#calculate distances between simulated distributions and total distribution
simCDabs = np.zeros([8]);
for k in range(4):
    for j in range(2):
        simCDabs[j*4+k] = corrD(totDataSim,dataSim[k,j,:,:]);

print("\ndistances(of sims) to total simulated data");
print("dists: " + str(np.around(simCDabs,decimals=2)));
print("mean: " + str(np.around(np.mean(simCDabs),decimals=2)));
print("std: " + str(np.around(np.std(simCDabs),decimals=2)));

#calculate distances between generated distributions and total simulated distribution        
genCDabs = np.zeros([4]);
for k in range(4):
        genCDabs[k] = corrD(totDataSim,dataGen[k,1,:,:]);

print("\ndistances(of generated - last test epoch) to total simulated data");
print("dists: " + str(np.around(genCDabs,decimals=2)));
print("mean: " + str(np.around(np.mean(genCDabs),decimals=2)));
print("std: " + str(np.around(np.std(genCDabs),decimals=2)));

#show correlation structure of generated data
corrs,meansA,meansB,samples = detCorrs(totDataSim);
print("\ncorrelation structure for simulation");
#print(np.around(meansA,decimals=2));
#print(np.around(meansB,decimals=2));
print("corrs:");
print(np.around(corrs,decimals=2));
# print(samples);
# print(np.sum(samples));

dagID = 0;
corrs,meansA,meansB,samples = detCorrs(dataGen[dagID,1,:,:]);
print("\ncorrelation structure for DAG #" + str(dagID));
print("corrs:");
print(np.around(corrs,decimals=2));

dagID = 1;
corrs,meansA,meansB,samples = detCorrs(dataGen[dagID,1,:,:]);
print("\ncorrelation structure for DAG #" + str(dagID));
print("corrs:");
print(np.around(corrs,decimals=2));

dagID = 2;
corrs,meansA,meansB,samples = detCorrs(dataGen[dagID,1,:,:]);
print("\ncorrelation structure for DAG #" + str(dagID));
print("corrs:");
print(np.around(corrs,decimals=2));

dagID = 3;
corrs,meansA,meansB,samples = detCorrs(dataGen[dagID,1,:,:]);
print("\ncorrelation structure for DAG #" + str(dagID));
print("corrs:");
print(np.around(corrs,decimals=2));
