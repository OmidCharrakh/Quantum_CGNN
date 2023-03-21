#script to test distance measures with simulated datasets
import pandas as pd

def tweakObs(data,stdTh):
    cols = data.columns;
    vals = data.values;
    vals[:,0] = vals[:,0] / stdTh;
    vals[:,1] = vals[:,1] / stdTh;
    data = pd.DataFrame(vals,columns=cols);
    return data;

def corr2std(corr):
    std = 1 / ( 4 * (1 - corr**2) )**(1/4)
    return std;

#load data
dataHigh = pd.read_csv('./Data/high_m.csv');
dataHigh2 = pd.read_csv('./Data/high_m_2.csv');
dataMed = pd.read_csv('./Data/med_m.csv');
dataLow = pd.read_csv('./Data/low_m.csv');

#tweak
daHigh = tweakObs(dataHigh,1);
dataHigh2 = tweakObs(dataHigh2,1);
dataMed = tweakObs(dataMed,corr2std(0.5)/corr2std(0.995));
dataLow = tweakObs(dataLow,corr2std(0.005)/corr2std(0.995));

#save data
dataHigh.to_csv('./Debug/tweakedSimData/high_m.csv',index=False,index_label=False);
dataHigh2.to_csv('./Debug/tweakedSimData/high_m_2.csv',index=False,index_label=False);
dataMed.to_csv('./Debug/tweakedSimData/med_m.csv',index=False,index_label=False);
dataLow.to_csv('./Debug/tweakedSimData/low_m.csv',index=False,index_label=False);