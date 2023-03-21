import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import torch as th
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set()
import math
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline, CubicSpline, pchip
from sklearn.preprocessing import Binarizer, StandardScaler, scale
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import itertools
import os
import csv
import warnings
from pathlib import Path
import urllib
from ast import literal_eval
import cdt
cdt.SETTINGS.rpath="/Library/Frameworks/R.framework/Versions/4.0/Resources/Rscript"
from cdt.metrics import (precision_recall, SID, SHD)
from cdt.data import load_dataset
from cdt.causality.graph import CGNN, PC, GES, MMPC, Inter_IAMB, Fast_IAMB, IAMB, GS, LiNGAM, CAM
from cdt.independence.stats import AdjMI, NormalizedHSIC, MIRegression, KendallTau, NormMI, PearsonCorrelation, SpearmanCorrelation
from fcit import fcit
#import Omid_Functions as OF

from itertools import chain, product, starmap
from functools import partial

def space_restrictor(is_hidden, 
                     saving_path):
    # P→Q <=> ¬P∨Q
    candidates=[]
    if is_hidden:
        nb_vars=7
        for i in itertools.product([0, 1], repeat=nb_vars*nb_vars):
            A = np.reshape(np.array(i), (nb_vars, nb_vars))
            if (#1. It should be a DAG
                np.trace(A)== 0 and nx.is_directed_acyclic_graph(nx.DiGraph(A)) and 
                #2. SA_SB independence
                A[2,3]+A[3,2]==0 and 
                #3. SA_OA and SB_OB dependence
                A[2,0]+A[3,1]==2 and 
                #4. hidden outgoing edges 0 or 2 
                (A[4,:].sum()==0 or A[4,:].sum()==2) and 
                (A[5,:].sum()==0 or A[5,:].sum()==2) and 
                (A[6,:].sum()==0 or A[5,:].sum()==2) and
                #5. hidden incomming edges=0 
                (A[:,4].sum()==0) and 
                (A[:,5].sum()==0) and 
                (A[:,6].sum()==0) and
                #6. each confounder corresponds to one pair => 4:0--1, 5:0--3, 6:1--2 
                (A[4,2]+A[4,3]+A[4,5]+A[4,6]==0) and 
                (A[5,1]+A[5,2]+A[5,4]+A[5,6]==0) and 
                (A[6,0]+A[6,3]+A[6,4]+A[6,5]==0)):
                candidates.append(A)
    given_candidates=np.empty([len(candidates), int(candidates[0].size)])
    for indx, cand in enumerate(candidates):
        given_candidates[indx,0:int(candidates[0].size)]= cand.reshape(1, candidates[0].size)
    np.savetxt(saving_path, given_candidates, fmt='%f')
    
def OI_PI(info_path, saving_path):

    df=pd.read_csv(info_path)
    features=["OI", "PI", "MI"]
    titles=["Outcome Independence", "Parameter Independence", "Measurement Independence"]
    plt.figure(figsize=(15,3))
    for indx, feature in enumerate(features):
        plt.subplot(1,3,indx+1)
        g=sns.kdeplot(data=df, x="loss_rel", hue=feature, hue_order=[0,1], multiple="fill", linewidth=1, common_norm=True, palette=['#5F5F5F', '#008001'])
        g.set_title(titles[indx], weight='bold')
        leg = g.get_legend()
        leg.set_title(''); g.set_xlabel(''); g.set_ylabel(''); plt.xticks([], []); plt.yticks([], []);
        for t, l in zip(leg.texts, ['No', 'Yes']): t.set_text(l)
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
   

import warnings
warnings.filterwarnings("ignore")

def stratified_sample(df, strata, size=None, seed=None, keep_index= True):
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    # controlling variable to create the dataframe or append to it
    first = True 
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry=''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum].copy()
            n = tmp_grpd.iloc[i]['samp_size']
            if type(value) == str:
                value = "'" + str(value) + "'"
            if s != len(strata)-1:
                qry = qry + stratum + ' == ' + str(value) +' & '
            else:
                qry = qry + stratum + ' == ' + str(value)
        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=int(n), random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=int(n), random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)
    return stratified_df

def stratified_sample_report(df, strata, size=None):
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata].copy()
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd['size']).astype(int)
    return tmp_grpd

def __smpl_size(population, size):
    if size is None:
        cochran_n = round(((1.96)**2 * 0.5 * 0.5)/ 0.02**2)
        n = round(cochran_n/(1+((cochran_n -1) /population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n

def draw_2D_samples(n_samples):
    n_list = np.linspace(1,100,100).tolist()
    w_list = np.linspace(0,2,10000).tolist()
    products = [e for e in itertools.product(*[n_list, w_list])]
    sample_ids = np.random.choice(len(products), size=n_samples, replace=False)
    samples = [products[i] for i in sample_ids]
    df = pd.DataFrame()
    df['nb_base_hu'] = [int(sample[0]) for sample in samples]
    df['wg_regularizer'] = [sample[1] for sample in samples]
    df['nb_total_hu'] = [count_hu(A,n,w,2).sum() for (n, w) in samples]
    return df