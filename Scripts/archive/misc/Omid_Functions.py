import numpy as np
from numpy.matrixlib import defmatrix
import pandas as pd
import networkx as nx
import cdt
cdt.SETTINGS.rpath="/Library/Frameworks/R.framework/Versions/4.0/Resources/Rscript"
import matplotlib.pyplot as plt; from matplotlib.colors import LinearSegmentedColormap; from matplotlib.backends.backend_pdf import PdfPages; from matplotlib.patches import Patch
import seaborn as sns; 
sns.set()
from scipy.interpolate import pchip; from scipy.stats import pearsonr
import sklearn; from sklearn.preprocessing import Binarizer; from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier; from sklearn.metrics import confusion_matrix, classification_report, accuracy_score; from sklearn.model_selection import train_test_split; from sklearn.utils import resample
import itertools; from itertools import repeat
import os
from cdt.causality.graph import PC, GES, MMPC, IAMB, GS, LiNGAM; from cdt.independence.stats import AdjMI, KendallTau, PearsonCorrelation, SpearmanCorrelation
from fcit import fcit
import pickle
import ast
######################################################################################

def adjacency_visualizer(A):
    if A.shape[0]==8:
        translator=pd.DataFrame(); translator['original_name']=[0,1,2,3,4,5,6,7]; translator['converted_name']=[r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$', r'$\lambda$', r'$\mu$', r'$\nu$', r'$\xi$'] ; translator['color']=['r', 'r', 'g', 'g', 'gold', 'gold', 'gold', 'gold']; translator['position']=[np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5]), np.array([0, +.8]), np.array([0, +0.2]), np.array([0, -0.2]), np.array([0, -.8])]
        connected_nodes= [0,1,2,3]+[idx for idx in [4,5,6,7] if A[idx,:].sum()]; adj=A[connected_nodes, :][:, connected_nodes]; df=translator.copy(); df=df.iloc[connected_nodes, :].reset_index(drop=True);
    elif A.shape[0]==4:
        translator=pd.DataFrame(); translator['original_name']=[0,1,2,3]; translator['converted_name']=[r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$']; translator['position']=[np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5])]; translator['color']=['r', 'r', 'g', 'g']; df=translator.copy(); adj=A.copy()
    plt.figure(figsize=(5,5))
    G = nx.DiGraph(adj); G = nx.relabel_nodes(G, dict(zip(df.index, df['converted_name'])))
    nx.draw_networkx(G, pos=dict(zip(df.converted_name, df['position'])), node_color=df['color'].tolist(), node_size=1000, width=3)
    plt.show(); plt.clf(); plt.close()
######################################################################################

def path_counter(G, s, t):
    counts=len([path for path in nx.all_simple_paths(G, s, t)])
    return(counts)

def literal_return(val):
    try:
        return ast.literal_eval(val)
    except ValueError:
        return (val)

def object_saver(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def object_loader(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def symmetry_checker(A):
    if A.shape[0]==4:
        if (A[1,0]+A[0,1]==0 and A[2,3]+A[3,2]==0 and A[0,2]==A[1,3] and A[2,0]==A[3,1] and A[0,3]==A[1,2] and A[3,0]==A[2,1]):
            status=1
        else:
            status=0
    else:
        if (A[1,0]+A[0,1]==0 and A[2,3]+A[3,2]==0 and A[0,2]==A[1,3] and A[2,0]==A[3,1] and A[0,3]==A[1,2] and A[3,0]==A[2,1] and A[5,0]==A[6,1]):
            status=1
        else:
            status=0
    return(int(status))

def retrocausal_checker(A):
    if (path_counter(nx.DiGraph(A), 0, 2)+path_counter(nx.DiGraph(A), 1, 3)):
        status=1
    else:
        status=0
    return(int(status))

def superluminal_checker(A):
    if (A[0,1] or A[1,0] or A[2,1] or A[3,0]):
        status=1
    else:
        status=0
    return(int(status))

def superdeterministic_checker(A):
    if A.shape[0]==4:
        if (A[0,3]+A[1,2]):
            status=1
        else:
            status=0
    else:   
        if (A[5:,:].sum()):
            status=1
        else:
            status=0
    return(int(status))

def classical_checker(A): 
    if (not superluminal_checker(A)) and (not superdeterministic_checker(A)) and (not retrocausal_checker(A)) and (A[0,3]+A[1,2]==0):
        status=1
    else:
        status=0
    return(int(status))

def OO_CC_checker(A):
    if A.shape[0]==4:
        nodes=[2,3]
    else:
        nodes=[2,3,4,5,6,7]
    if np.sum([path_counter(nx.DiGraph(A), n, 0)*path_counter(nx.DiGraph(A), n, 1) for n in nodes]):
        status=1
    else:
        status=0
    return int(status)

def OO_CE_checker(A):
    if path_counter(nx.DiGraph(A), 0, 1)+path_counter(nx.DiGraph(A), 1, 0):
        status=1
    else:
        status=0
    return int(status)

def OO_connection_checker(A):
    if nx.d_separated(nx.DiGraph(A), {0}, {1},{}):
        status='disconnect'
    elif OO_CE_checker(A) and OO_CC_checker(A):
        status='CE_CC'
    elif OO_CE_checker(A) and (not OO_CC_checker(A)):
        status='CE'
    elif OO_CC_checker(A) and (not OO_CE_checker(A)):
        status='CC'
    return status

def balancer(df, x, y, method='upsample'):
    df=df[[x, y]]
    count_dict=df[y].value_counts().to_dict()
    if method=='upsample':
        major_label=max(count_dict, key=count_dict.get)
        major_count=count_dict[major_label]
        major=df[df[y]==major_label]
        df_upsampled=major.copy()
        for label in count_dict.keys():
            if count_dict[label]<major_count:
                minor=df[df[y]==label]
                minor_upsampled=resample(minor, replace = True, n_samples = major_count, random_state =0)
                df_upsampled=pd.concat([df_upsampled, minor_upsampled])
        df_upsampled.reset_index(inplace=True, drop=True)
        return(df_upsampled)
    if method=='downsample':
        minor_label=min(count_dict, key=count_dict.get)
        minor_count=count_dict[minor_label]
        minor=df[df[y]==minor_label]
        df_downsampled=minor.copy()
        for label in count_dict.keys():
            if count_dict[label]>minor_count:
                major=df[df[y]==label]
                major_downsampled=resample(major, replace = False, n_samples = minor_count, random_state =0)
                df_downsampled=pd.concat([df_downsampled, major_downsampled])
        df_downsampled.reset_index(inplace=True, drop=True)
        return(df_downsampled)

def h_node(x):
    i,j=x[0], x[1]
    if (i==0 and j==1) or (i==1 and j==0):
        return(4)
    elif (i==0 and j==3) or (i==3 and j==0):
        return(5)
    elif (i==1 and j==2) or (i==2 and j==1):
        return(6)
    elif (i==2 and j==3) or (i==3 and j==2):
        return(7)
    else:
        return(False)
        
def versions_giver(adjacency, edge):
    versions=[]
    cand_v0=copy_giver(adjacency); versions.append(cand_v0) 
    cand_v1=copy_giver(adjacency); cand_v1[edge[0],edge[1]]=0; versions.append(cand_v1)
    cand_v2=copy_giver(adjacency); cand_v2[edge[0],edge[1]]=0; cand_v2[h_node(edge), edge[0]]=1; cand_v2[h_node(edge), edge[1]]=1; versions.append(cand_v2)
    #cand_v3=copy_giver(adjacency); cand_v3[h_node(edge), edge[0]]=1; cand_v3[h_node(edge), edge[1]]=1; versions.append(cand_v3)
    return(np.array(versions))
    
def copy_giver(arr):
    return (arr.copy())
######################################################################################

def df_info(solution_path, saving_path, is_hidden):
    if not is_hidden:
        nb_vars=4
    else:
        nb_vars=8
    df_solution = pd.DataFrame(np.loadtxt(solution_path)); df_solution.rename(columns={1:'Loss_additional', 2:'Loss'}, inplace=True); df_solution.sort_values(by=['Loss'], ignore_index=True, inplace=True)
    df = pd.DataFrame(); df['Loss']=df_solution['Loss']; df['Loss_additional']=df_solution['Loss_additional']; df['Ranking']=[inx+1 for inx in range(len(df))]; df['Ranking_additional']=[sorted(df['Loss_additional']).index(loss)+1 for loss in df['Loss_additional']]
    adjacency=[np.reshape(df_solution.values[inx,3:3+nb_vars**2],(nb_vars,nb_vars)).astype(int) for inx in range(len(df_solution))]; df['adjacency']=adjacency
    graphs=[nx.DiGraph(A) for A in adjacency]
    df['Total_Edges'] = [A.sum() for A in adjacency]
    df['Symmetric']= [symmetry_checker(A) for A in adjacency]
    df['Retrocausal'] = [retrocausal_checker(A) for A in adjacency]
    df['Superluminal'] = [superluminal_checker(A) for A in adjacency]
    df['Superdeterministic'] = [superdeterministic_checker(A) for A in adjacency]
    df['Classical'] = [classical_checker(A) for A in adjacency]
    df['OO_Connection'] = [OO_connection_checker(A) for A in adjacency]
    df['OA_OB'] = [+1 if A[0,1]!=0 else -1 if A[1,0]!=0 else 0 for A in adjacency]; 
    df['OA_SA'] = [+1 if A[0,2]!=0 else -1 if A[2,0]!=0 else 0 for A in adjacency]; 
    df['OA_SB'] = [+1 if A[0,3]!=0 else -1 if A[3,0]!=0 else 0 for A in adjacency]; 
    df['OB_SA'] = [+1 if A[1,2]!=0 else -1 if A[2,1]!=0 else 0 for A in adjacency]; 
    df['OB_SB'] = [+1 if A[1,3]!=0 else -1 if A[3,1]!=0 else 0 for A in adjacency]; 
    df['SA_SB'] = [+1 if A[2,3]!=0 else -1 if A[3,2]!=0 else 0 for A in adjacency]; 
    if is_hidden:
        df['OA_direct']=[1 if (path_counter(G, 0, 1)>0) else 0 for G in graphs];
        df['OB_direct']=[1 if (path_counter(G, 1, 0)>0) else 0 for G in graphs];
        df['lambda_cc']=[1 if (path_counter(G, 4, 0)>0) else 0 for G in graphs];#OA_lambda, OB_lambda
        df['mu_cc']=[1 if (path_counter(G, 5, 0)>0) else 0 for G in graphs];# OA_mu, SB_mu
        df['nu_cc']=[1 if (path_counter(G, 6, 1)>0) else 0 for G in graphs];#OB_nu, SA_nu
        df['xi_cc']=[1 if (path_counter(G, 7, 2)>0) else 0 for G in graphs];#SA_xi, SB_xi
        df['SA_cc']=[1 if (path_counter(G, 2, 0)*path_counter(G, 2, 1)>0) else 0 for G in graphs]; 
        df['SB_cc']=[1 if (path_counter(G, 3, 0)*path_counter(G, 3, 1)>0) else 0 for G in graphs]; 
    object_saver(df, saving_path)
    return df
######################################################################################

def dependence_plotter(data_path, saving_path):
    
    data=pd.read_csv(data_path)
    O_A, O_B, S_A, S_B = np.array(data['O_A']), np.array(data['O_B']), np.array(data['S_A']), np.array(data['S_B'])
    objects = [AdjMI(), KendallTau(), PearsonCorrelation(), SpearmanCorrelation()]
    x=['$O_A-O_B$', '$O_A-S_A$','$O_A-S_B$','$O_B-S_A$','$O_B-S_B$','$S_A-S_B$']
    y0=[obj.predict(O_A, O_B) for obj in objects]
    y1=[obj.predict(O_A, S_A) for obj in objects]
    y2=[obj.predict(O_A, S_B) for obj in objects]
    y3=[obj.predict(O_B, S_A) for obj in objects]
    y4=[obj.predict(O_B, S_B) for obj in objects]
    y5=[obj.predict(S_A, S_B) for obj in objects]
    y= [y0,y1,y2,y3,y4,y5]
    colors=['b','g','r','k']
    labels= ["Mutual Information", "Kendalls Tau", "Pearson Correlation", "Spearman Correlation"]
    plt.figure(figsize=(8,4))
    for i in range(6):
        for j in range(4):
            if i==0:
                plt.scatter(x[i], y[i][j], c=colors[j], label=labels[j])
            else:
                plt.scatter(x[i], y[i][j], c=colors[j])
    plt.title("Variouse Marginal Dependence Measures", weight='bold')
    plt.ylim(-0.05, .3) 
    plt.xticks(np.arange(len(x)), x, rotation=20, fontsize=12)
    plt.gca().xaxis.set_tick_params(pad=-1)
    plt.legend()
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
######################################################################################

def CITest_plotter(data, saving_path=None, cv_grid=None, num_perm=50):
    OA = data[:, 0:1]; 
    OB = data[:, 1:2]; 
    SA = data[:, 2:3]; 
    SB = data[:, 3:4]; 
    OA_OB = np.concatenate((OA, OB), axis=1); 
    OA_SA = np.concatenate((OA, SA), axis=1); 
    OA_SB = np.concatenate((OA, SB), axis=1); 
    OB_SA = np.concatenate((OB, SA), axis=1); 
    OB_SB = np.concatenate((OB, SB), axis=1); 
    SA_SB = np.concatenate((SA, SB), axis=1);
    
    #cv_grid=[1e-2, .2, .4, 2, 8, 64, 512]
    
    stats_return = False
    
    pval_m = [
        fcit.test(x=OA, y=OB, z=None, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OA, y=SA, z=None, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OA, y=SB, z=None, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OB, y=SA, z=None, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OB, y=SB, z=None, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=SA, y=SB, z=None, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return),
    ]  
    
    pval_c = [
        fcit.test(x=OA, y=OB, z=SA_SB, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OA, y=SA, z=OB_SB, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OA, y=SB, z=OB_SA, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OB, y=SA, z=OA_SB, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=OB, y=SB, z=OA_SA, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return), 
        fcit.test(x=SA, y=SB, z=OA_OB, cv_grid=cv_grid, num_perm=num_perm, plot_return=stats_return),
    ] 
    
    y = [
        [pval_m[0], pval_c[0]], 
        [pval_m[1], pval_c[1]], 
        [pval_m[2], pval_c[2]], 
        [pval_m[3], pval_c[3]], 
        [pval_m[4], pval_c[4]], 
        [pval_m[5], pval_c[5]],
    ]

    x = [
        '$O_A-O_B$', 
        '$O_A-S_A$', 
        '$O_A-S_B$',
        '$O_B-S_A$',
        '$O_B-S_B$',
        '$S_A-S_B$',
    ]
    colors=['#BA0100', '#008001']
    labels= ['Marginal','Conditional']
    markers= ['.', '*']
    
    plt.figure(figsize=(8,4))
    plt.subplot()
    for i in range(6):
        for j in range(2):
            if i==0:
                plt.scatter(x[i], y[i][j], c=colors[j], label=labels[j], marker=markers[j], s=200)
            else:
                plt.scatter(x[i], y[i][j], color=colors[j], marker=markers[j], s=250)         
    plt.xticks(np.arange(len(x)), x, rotation=20, fontsize=12)
    plt.gca().xaxis.set_tick_params(pad=-1)
    plt.ylabel("")
    plt.title('Marginal and Conditional Independence Tests', weight='bold')
    plt.legend(prop={'size': 10}, loc='upper right')
    if saving_path is not None: 
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

######################################################################################

def algs_plotter(data_path, saving_path):
    
    data=pd.read_csv(data_path)
    algs= ["PC()" , "GES()", "MMPC()", "IAMB()", "GS()", "LiNGAM()"]
    graphs=[eval(obj).predict(data) for obj in algs]
    fig = plt.figure(figsize=(16,3), tight_layout=True)
    mapping = {'O_A': '$O_A$', 'O_B': '$O_B$', 'S_A': '$S_A$', 'S_B': '$S_B$'}
    pos= {'$O_A$': np.array([-0.4, +0.4]),'$O_B$': np.array([+0.4, +0.4]),'$S_A$': np.array([-0.4, -0.4]),'$S_B$': np.array([+0.4, -0.4])}
    for indx, G in enumerate(graphs):
        ax = fig.add_subplot(1,6,indx+1)
        G = nx.relabel_nodes(G, mapping)
        nx.draw_networkx(G, pos=pos, node_color=['r','r', 'g', 'g'], node_size=1500)
        ax.set_title(str(algs[indx].split('()')[0]), fontweight="bold")
    plt.savefig(saving_path, dpi = 300)
    plt.show()
    plt.clf()
    plt.close()
######################################################################################

def mmd_plotter(info_path, saving_path):
    df=object_loader(info_path)
    plt.figure(figsize=(8,4))
    plt.plot(df['Ranking'], df['Loss'], color = 'b', label='Relative', linewidth=2)
    plt.xlabel('DAG Ranking', weight='bold')
    plt.ylabel('MMD Loss', weight='bold')
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
######################################################################################
def DAGs_plotter(solutions_m_path, solutions_h_path, saving_path, is_hidden=False, show_figures=False):

    node_size=1200; width=3.0; figsize=(5,5); node_size=1000; fontdict = {'size':8}; zfill=4; os.makedirs(saving_path, exist_ok=True); paths=[]
    loss_m=[float(row) for row in pd.DataFrame(np.loadtxt(solutions_m_path)[:,2]).sort_values(by=[0], ignore_index=True).values]
    loss_h=[float(row) for row in pd.DataFrame(np.loadtxt(solutions_h_path)[:,2]).sort_values(by=[0], ignore_index=True).values]
    adjacency_m=[np.reshape(row,(4,4)) for row in pd.DataFrame(np.loadtxt(solutions_m_path)[:,2:]).sort_values(by=[0], ignore_index=True).values[:,1:-1]]
    adjacency_h=[np.reshape(row,(8,8)) for row in pd.DataFrame(np.loadtxt(solutions_h_path)[:,2:]).sort_values(by=[0], ignore_index=True).values[:,1:-1]]
    if not is_hidden:
        translator=pd.DataFrame(); translator['original_name']=[0,1,2,3]; translator['converted_name']=[r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$']; translator['position']=[np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5])]; translator['color']=['r', 'r', 'g', 'g']; df=translator.copy()
        local_ranking= [rank for rank in range(1, len(loss_m)+1)]
        global_ranking= [sorted(loss_m+loss_h).index(loss)+1 for loss in loss_m]
        for inx, A in enumerate(adjacency_m):
            plt.figure(figsize=figsize)
            plt.title('Local ranking = '+ str(local_ranking[inx]).zfill(zfill), fontdict = fontdict, fontweight="bold", loc='right')
            plt.title('Loss = {:.03f}'.format(loss_m[inx]), fontdict = fontdict, fontweight="bold", loc='center')
            plt.title('Global ranking = '+ str(global_ranking[inx]).zfill(zfill), fontdict = fontdict, fontweight="bold", loc='left') 
            G = nx.DiGraph(A); G = nx.relabel_nodes(G, dict(zip(df.index, df['converted_name'])))
            nx.draw_networkx(G, node_color=['r', 'r', 'g', 'g'], node_size=node_size, width=width, pos=dict(zip(df.converted_name, df['position'])))
            plt.savefig(os.path.join(saving_path, str(inx+1).zfill(zfill)+'.png'),dpi = 300, bbox_inches='tight')
            paths.append(os.path.join(saving_path, str(inx+1).zfill(zfill)+'.png'))
            if show_figures:plt.show()
            plt.clf()
            plt.close()
    else:
        translator=pd.DataFrame(); translator['original_name']=[0,1,2,3,4,5,6,7]; translator['converted_name']=[r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$', r'$\lambda$', r'$\mu$', r'$\nu$', r'$\xi$'] ; translator['color']=['r', 'r', 'g', 'g', 'gold', 'gold', 'gold', 'gold']; translator['position']=[np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5]), np.array([0, +.8]), np.array([0, +0.2]), np.array([0, -0.2]), np.array([0, -.8])]
        local_ranking= [rank for rank in range(1, len(loss_h)+1)]
        global_ranking= [sorted(loss_m+loss_h).index(loss)+1 for loss in loss_h]
        for inx, A in enumerate(adjacency_h):
            plt.figure(figsize=figsize)
            plt.title('Local ranking = '+ str(local_ranking[inx]).zfill(zfill), fontdict = fontdict, fontweight="bold", loc='right')
            plt.title('Loss = {:.03f}'.format(loss_h[inx]), fontdict = fontdict, fontweight="bold", loc='center')
            plt.title('Global ranking = '+ str(global_ranking[inx]).zfill(zfill), fontdict = fontdict, fontweight="bold", loc='left') 
            connected_nodes= [0,1,2,3]+[idx for idx in [4,5,6,7] if A[idx,:].sum()]; adj=A[connected_nodes, :][:, connected_nodes]; df=translator.copy(); df=df.iloc[connected_nodes, :].reset_index(drop=True);
            mapping = dict(zip(df.index, df['converted_name'])); pos=dict(zip(df.converted_name, df['position'])); node_color=df['color'].tolist()
            G = nx.DiGraph(adj); G = nx.relabel_nodes(G, mapping); edgelist, edge_color = zip(*nx.get_edge_attributes(G, 'weight').items()); edge_color=[round(abs(e),2) for e in edge_color]
            nx.draw_networkx(G, pos=pos, node_color=node_color, node_size=node_size, edgelist=edgelist, width=width)
            plt.savefig(os.path.join(saving_path, str(inx+1).zfill(zfill)+'.png'),dpi = 300, bbox_inches='tight'); 
            paths.append(os.path.join(saving_path, str(inx+1).zfill(zfill)+'.png'))
            if show_figures:
                plt.show()
            plt.clf(); 
            plt.close()
    xnum=4; ynum=5; figs_num=xnum*ynum; pages_num=1+len(paths)//figs_num; 
    if not is_hidden:
        pp = PdfPages(os.path.join(saving_path, 'merged_main.pdf'))
    else:
        pp = PdfPages(os.path.join(saving_path, 'merged_hidden.pdf'))
    for page_num in range(pages_num):
        plt.figure(figsize=(11.69, 8.27))
        start_num =page_num *figs_num
        finish_num=start_num+figs_num
        for indx, path in enumerate(paths[start_num: finish_num]):
            ax = plt.subplot(xnum, ynum, indx + 1)
            fig = plt.imread(path)
            ax.imshow(fig)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(pp, format='pdf', dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()
    pp.close()
######################################################################################

def edges_plotter(info_path, saving_path, is_hidden):
    data=object_loader(info_path)
    pairs = ['OA_OB','OA_SB', 'OB_SA', 'OA_SA','OB_SB', 'SA_SB']
    titles=[r'$O_A\, vs. \, O_B$', r'$O_A\, vs. \, S_B$', r'$O_B\, vs. \, S_A$', r'$O_A\, vs. \, S_A$', r'$O_B\, vs. \, S_B$', r'$S_A\, vs. \, S_B$',]
    plt.figure(figsize=(16,5))
    if is_hidden:
        df=data[['Loss']].copy(); 
        df['OA_OB']=[2 if data['lambda_cc'][i]!=0 else data['OA_OB'][i] for i in range(len(data))];
        df['OA_SB']=[2 if data['mu_cc'][i]!=0 else -data['OA_SB'][i] for i in range(len(data))]; 
        df['OB_SA']=[2 if data['nu_cc'][i]!=0 else -data['OB_SA'][i] for i in range(len(data))]; 
        df['OA_SA']=-data['OA_SA'].copy(); 
        df['OB_SB']=-data['OB_SB'].copy(); 
        df['SA_SB']=[2 if data['xi_cc'][i]!=0 else data['SA_SB'][i] for i in range(len(data))]; 
        palettes=[['#5F5F5F', '#FFD700', '#BA0100', '#008001'], ['#5F5F5F', '#FFD700', '#BA0100', '#008001'], ['#5F5F5F', '#FFD700', '#BA0100', '#008001'], ['#5F5F5F', '#BA0100', '#008001'], ['#5F5F5F', '#BA0100', '#008001'], ['#5F5F5F', '#FFD700']]
        hue_order=[[0,2,-1,1],[0,2,-1,1],[0,2,-1,1], [0,-1,1], [0,-1,1], [0,2],] 
        labels = [['None', r'$O_A\leftarrow \lambda \rightarrow O_B$', r'$O_A\leftarrow O_B$', r'$O_A\rightarrow O_B$'], ['None', r'$O_A\leftarrow \mu \rightarrow S_B$',     r'$O_A\rightarrow S_B$', r'$O_A\leftarrow S_B$'], ['None', r'$O_B\leftarrow \nu \rightarrow S_A$',     r'$O_B\rightarrow S_A$', r'$O_B\leftarrow S_A$'],['None', r'$O_A\rightarrow S_A$', r'$O_A\leftarrow S_A$'], ['None', r'$O_B\rightarrow S_B$', r'$O_B\leftarrow S_B$'], ['None', r'$S_A\leftarrow \xi \rightarrow S_B$'],]
    else:
        df=data[['Loss']].copy(); df['OA_OB']= data['OA_OB'].copy(); df['OA_SB']=-data['OA_SB'].copy();df['OB_SA']=-data['OB_SA'].copy(); df['OA_SA']=-data['OA_SA'].copy(); df['OB_SB']=-data['OB_SB'].copy(); df['SA_SB']= data['SA_SB'].copy(); 
        palettes=[['#5F5F5F', '#BA0100', '#008001'], ['#5F5F5F', '#BA0100', '#008001'],['#5F5F5F', '#BA0100', '#008001'],['#5F5F5F', '#BA0100', '#008001'],['#5F5F5F', '#BA0100', '#008001'],['#5F5F5F'],]
        hue_order=[[0,-1,1],[0,-1,1],[0,-1,1], [0,-1,1],[0,-1,1],[0]] 
        labels = [['None', r'$O_A\leftarrow O_B$', r'$O_A\rightarrow O_B$'], ['None', r'$O_A\rightarrow S_B$', r'$O_A\leftarrow S_B$'], ['None', r'$O_B\rightarrow S_A$', r'$O_B\leftarrow S_A$'], ['None', r'$O_A\rightarrow S_A$', r'$O_A\leftarrow S_A$'], ['None', r'$O_B\rightarrow S_B$', r'$O_B\leftarrow S_B$'], ['None'],]
    for indx, pair in enumerate(pairs):
        ax=plt.subplot(2,3, indx+1)
        g = sns.kdeplot(data=balancer(df=df, x="Loss", y=pair, method='upsample'), x="Loss", hue=pair, hue_order=hue_order[indx], multiple="fill", common_norm=True, linewidth=1, palette= palettes[indx])
        leg = g.get_legend(); leg.remove(); g.set_xlabel(''); g.set_ylabel(''); g.set_xticks([]); g.set_yticks([]); handles=[Patch(color=palettes[indx][i], label=labels[indx][i]) for i in range(len(labels[indx]))]
        ax.legend(handles=handles, ncol=4+indx//3, fancybox=True, shadow=True, frameon=True, framealpha=.8, prop={'size': 9}, loc='upper center')
        ax.text(x=0.5, y=0.02, s= titles[indx], fontsize = 14, fontweight='bold', horizontalalignment='center', verticalalignment='baseline', transform=ax.transAxes, color='black')
        plt.tight_layout()
    plt.tight_layout()
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
######################################################################################

def total_edges_plotter(info_path, saving_path):

    df=object_loader(info_path); 
    plt.figure(figsize=(8,4))
    g=sns.kdeplot(data=df, x='Loss', hue="Total_Edges", multiple="fill", linewidth=1, common_norm=True, palette=sns.color_palette("rocket", len(df["Total_Edges"].value_counts()))[::-1], warn_singular=False)
    leg = g.get_legend(); leg.set_title('')
    g.set_ylabel('')
    g.set_xlabel('MMD Loss', weight='bold')
    g.set_yticks([0, 0.5, 1.0])
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()

######################################################################################
def interpretations_plotter(info_path, saving_path, separate_plots=False):

    df=object_loader(info_path); 
    fontsize=12; x='Loss'
    if separate_plots:
        palette=['#5F5F5F', '#008001']; labels = ['No', 'Yes']
        plt.figure(figsize=(16,3))
        ########################################
        plt.subplot(131)
        g=sns.kdeplot(data=balancer(df, x=x, y='Superluminal', method='downsample'),
                      x=x, hue="Superluminal", hue_order=[0,1], multiple="fill", linewidth=1, common_norm=True, palette=palette)
        g.set_title('Superluminal', weight='bold', fontsize = fontsize)
        leg = g.get_legend(); leg.set_title(''); g.set_xlabel(''); g.set_ylabel(''); 
        for t, l in zip(leg.texts, labels): t.set_text(l)
        ########################################
        plt.subplot(132)
        g=sns.kdeplot(data=balancer(df, x=x, y='Superdeterministic', method='downsample'),
                      x=x, hue="Superdeterministic", hue_order=[0,1], multiple="fill", linewidth=1, common_norm=True, palette=palette)
        g.set_title('Superdeterministic', weight='bold', fontsize = fontsize)
        leg = g.get_legend(); leg.set_title(''); g.set_xlabel(''); g.set_ylabel(''); plt.yticks([], []);  
        for t, l in zip(leg.texts, labels): t.set_text(l)
        ########################################
        plt.subplot(133)
        g=sns.kdeplot(data=balancer(df, x=x, y='Retrocausal', method='downsample'), 
                      x=x, hue="Retrocausal", hue_order=[0,1], multiple="fill", linewidth=1, common_norm=True, palette=palette)
        g.set_title('Retrocausal', weight='bold', fontsize = fontsize)
        leg = g.get_legend(); leg.set_title(''); g.set_xlabel(''); g.set_ylabel(''); plt.yticks([], []); 
        for t, l in zip(leg.texts, labels): t.set_text(l)
        ########################################
        plt.tight_layout()
        plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()
    else:
        mask1=df['Retrocausal']==1; mask2=df['Superluminal']==1; mask3=df['Superdeterministic']==1
        df1=df[ mask1 & ~mask2 & ~mask3]; df2=df[~mask1 &  mask2 & ~mask3]; df3=df[~mask1 & ~mask2 &  mask3]; df_c=df[~mask1 & ~mask2 & ~mask3]; 
        Bell_A=np.array([[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0],[1, 1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]])
        Bell_Loss=[row.Loss for inx, row in df.iterrows() if np.array_equal(row.adjacency, Bell_A)][0]
        df_concat=pd.concat([df1,df2,df3]).reset_index(drop=True); df_concat.sort_values(by=['Loss'], ignore_index=True, inplace=True)
        qm_interpretation=['None']*len(df_concat)
        for inx, row in df_concat.iterrows():
            if (row.Retrocausal==1):
                qm_interpretation[inx]='Retrocausal'
            elif (row.Superluminal==1):
                qm_interpretation[inx]='Superluminal'
            elif (row.Superdeterministic==1):
                qm_interpretation[inx]='Superdeterministic'
        df_concat['qm_interpretation']=qm_interpretation
        plt.figure(figsize=(8,4))
        g=sns.kdeplot(data=balancer(df=df_concat, x="Loss", y='qm_interpretation', method='downsample'), x='Loss', hue="qm_interpretation", hue_order=['Retrocausal','Superluminal', 'Superdeterministic'], multiple="fill", linewidth=1, common_norm=True,palette=sns.color_palette("rocket", len(df_concat["qm_interpretation"].value_counts()))[::-1])
        leg = g.get_legend(); leg.set_title(''); g.set_ylabel(''); g.set_xlabel('MMD Loss', weight='bold'); g.set_yticks([0, 0.5, 1.0]); 
        plt.axvline(x=Bell_Loss, color='black', linestyle='dotted', linewidth = '2')
        plt.text(x=Bell_Loss, y=1.01, s="Bell's Model" ,   fontsize = 10, horizontalalignment='center', verticalalignment='baseline',color='black')
        plt.text(x=Bell_Loss, y=-.05, s=round(Bell_Loss,2),fontsize = 10, horizontalalignment='center', verticalalignment='baseline',color='black')
        plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

######################################################################################

def importance_plotter(info_path, saving_path, do_categorization=True, catg_counts=3, sorting_criterion='RF'):
    
    df= object_loader(info_path)
    figsize=(16,5); labelsize=11; top=.25
    features= ['Total_Edges', 'Symmetric', 'Retrocausal', 'Superluminal', 'Superdeterministic', 'Classical', 'OO_Connection','lambda_cc', 'mu_cc', 'nu_cc', 'xi_cc','OA_OB', 'OA_SA', 'OA_SB', 'OB_SA', 'OB_SB', 'OA_direct', 'OB_direct', 'SA_cc', 'SB_cc']
    f_ticks = ['Total\nEdges','Symmetric', 'Retrocausal', 'Superluminal', 'Superdeterministic', 'Classical', 'Outcomes\nConnection',r'$\lambda$', r'$\mu$', r'$\nu$', r'$\xi$','$O_A$-$O_B$', '$O_A$-$S_A$', '$O_A$-$S_B$', '$O_B$-$S_A$', '$O_B$-$S_B$', '$O_A$ direct', '$O_B$ direct', '$S_A$ CC', '$S_B$ CC']
    df=df[['Loss']+features]; df=df.replace({"OO_Connection": {'CE':0, 'CC':1, 'disconnect':2, 'CE_CC':3}}); df["OO_Connection"]=df["OO_Connection"].astype("category")
    
    if do_categorization:
        df['Loss']=pd.qcut(df['Loss'], q=np.linspace(0, 1, catg_counts+1), labels=list(range(catg_counts))); df=df.astype("category")
        X=df[features]; y=df['Loss'].values; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
        RF = RandomForestClassifier(n_estimators=1000); RF.fit(X_train, y_train); y_pred=RF.predict(X_test); accuracy_score=sklearn.metrics.accuracy_score(y_pred, y_test); RF_importances= RF.feature_importances_; print('RF Accuracy:{:.03f}'.format(accuracy_score))
    else:
        X=df[features]; y=df['Loss'].values
        RF = RandomForestRegressor(n_estimators=1000); RF.fit(X, y); RF_importances= RF.feature_importances_;
    importance = pd.DataFrame({'f_ticks': f_ticks, 'Features': features, 'RF': RF_importances})
    criteria = ['AdjMI', 'KendallTau', 'Pearson', 'Spearman']
    for inx, criterion in enumerate([AdjMI(), KendallTau(), PearsonCorrelation(), SpearmanCorrelation()]):importance[criteria[inx]] = importance['Features'].apply(lambda x: criterion.predict(df['Loss'], df[x]))
    importance.sort_values(by=[sorting_criterion], key=abs, ignore_index=True, inplace=True)
    plt.figure(figsize=figsize)
    plt.ylim(top = top)
    bar = plt.bar(np.arange(len(features)), np.abs(importance[sorting_criterion]), .3, align='center', alpha=1, color=sns.color_palette("rocket", len(features))[::-1])
    for indx, rect in enumerate(bar):
        if importance['Pearson'][indx]>0:
            s='+'
            fontsize=15
        else:
            s='-'
            fontsize=25
        plt.text(rect.get_x() + rect.get_width()/2, .99* rect.get_height(), s=s, ha='center', fontsize=fontsize, weight='bold')
    plt.gca().yaxis.set_tick_params(pad=-2)
    plt.xticks(np.arange(len(features)), importance['f_ticks'])
    plt.gca().xaxis.set_tick_params(rotation=20, direction='inout', pad=-1, labelsize=labelsize)
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
    return(importance)

#################################################################################################################
def nonlocality_plotter(info_path, knots, saving_path):
                       
    f_ticks = ['Observable CE','Hidden CC', 'Observable CC']
    color_map = sns.color_palette("rocket", len(f_ticks))
    features=['OA_direct', 'OB_direct', 'lambda_cc', 'mu_cc', 'nu_cc', 'xi_cc', 'SA_cc', 'SB_cc']
    df = object_loader(info_path)[features]; df = df[df.sum(axis=1)!=0].reset_index(drop=True)
    Q0 = [(row.OA_direct+row.OB_direct)/row.sum() for i, row in df.iterrows()]
    Q1 = [(row.lambda_cc+row.mu_cc+row.nu_cc+row.xi_cc)/row.sum() for i, row in df.iterrows()]
    Q2 = [(row.SA_cc+row.SB_cc)/row.sum() for i, row in df.iterrows()]
    Q=[Q0, Q1, Q2]
    x = np.arange(1, len(Q0)+1); xnew = np.linspace(x.min(), x.max(), knots)
    y=[pchip(x, np.array(q))(xnew) for q in Q[:-1]]; y.insert(len(y), 1-np.array([sum([y[i][j] for i in range(len(y))]) for j in range(len(y[0]))]))
    plt.figure(figsize=(8,4))
    plt.stackplot(xnew, y, labels=f_ticks, colors =color_map)
    plt.title('', weight='bold')
    plt.xlabel("DAGs Ranking", weight='bold')
    plt.ylabel("Relative Contribution", weight='bold')
    plt.ylim(top=1)
    plt.xlim(left=xnew.min(), right=xnew.max())
    plt.legend(ncol=3,loc='upper center', fancybox=True, shadow=True)
    plt.gca().yaxis.set_tick_params(direction='in')
    plt.xticks([], [])
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
#################################################################################################################

def space_restrictor(saving_path_m, saving_path_h):
    # P→Q <=> ¬P∨Q
    ################################################
    # No hidden:
    A_m=[] 
    #0. To include the completely disconected DAG:
    A_m.append(np.zeros(shape=(4,4)))
    for i in itertools.product([0, 1], repeat=4*4):
        A = np.reshape(np.array(i), (4, 4))
        #1. To be a DAG:
        if (np.trace(A) == 0) and (nx.is_directed_acyclic_graph(nx.DiGraph(A))):
            #2. To disconnect SA_SB:
            if (A[2,3]+A[3,2]==0): 
                A_m.append(A)
    given_candidates_m=np.empty([len(A_m), int(A_m[0].size)])
    for indx, cand in enumerate(A_m):
        given_candidates_m[indx,0:int(A_m[0].size)]= cand.reshape(1, A_m[0].size)
    given_candidates_m=np.unique(given_candidates_m, axis=0)
    np.savetxt(saving_path_m, given_candidates_m, fmt='%f')
    ################################################
    # Hidden Variables
    B_m=[np.pad(np.reshape(np.array(i),(4,4)), [(0,4)], mode='constant', constant_values=0) for i in itertools.product([0,1], repeat=4*4) if (np.trace(np.reshape(np.array(i),(4,4)))==0 and nx.is_directed_acyclic_graph(nx.DiGraph(np.reshape(np.array(i),(4,4)))))]
    A_h = np.empty([0, 8, 8])
    for B in B_m:
        try:
            element_indx=[(i,j) for j in range(B.shape[0]) for i in range(B.shape[0]) if B[i,j]!=0 and h_node((i,j))]
            versions=[np.empty([0, 8, 8]) for i in range(len(element_indx))]
            for inx, r in enumerate(repeat(None, len(element_indx))):
                if inx==0:
                    for e in element_indx:
                        versions[inx]=np.concatenate((versions[inx], versions_giver(adjacency=B, edge=e)), axis=0)
                else:
                    for e in element_indx:
                        for v in versions[inx-1]:
                            versions[inx]=np.concatenate((versions[inx], versions_giver(adjacency=v, edge=e)), axis=0)
            A_h=np.concatenate((A_h, versions[-1]), axis=0)
        except:
            A_h=np.concatenate((A_h, np.expand_dims(B, axis=0)), axis=0)
    A_h = np.unique(A_h, axis=0)
    A_h=[A for A in A_h if (
        ((A[4,0]+A[4,1])*(A[0,1]+A[1,0])==0) and 
        ((A[5,0]+A[5,3])*(A[0,3]+A[3,0])==0) and 
        ((A[6,1]+A[6,2])*(A[1,2]+A[2,1])==0) and
        ((A[2,3]+A[3,2])==0) and A[4,:].sum()+A[5,:].sum()+A[6,:].sum()+A[7,:].sum())]
    given_candidates_h=np.empty([len(A_h), int(A_h[0].size)])
    for indx, cand in enumerate(A_h):
        given_candidates_h[indx,0:int(A_h[0].size)]= cand.reshape(1, A_h[0].size)
    unique_candidates_h=np.unique(given_candidates_h, axis=0)
    np.savetxt(saving_path_h, unique_candidates_h, fmt='%f')
#################################################################################################################

def dist_plotter(data_path, saving_path):
    data=pd.read_csv(data_path)[0:50000]
    sns.set_style("dark")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 3))
    sns.histplot(ax=axes[0], data=data, x='O_A', stat='density', kde=True, bins=20, line_kws=dict(linewidth=1))
    sns.histplot(ax=axes[1], data=data, x='O_B', stat='density', kde=True, bins=20, line_kws=dict(linewidth=1))
    sns.histplot(ax=axes[2], data=data, x='S_A', stat='density', kde=True, bins=50, line_kws=dict(linewidth=1))
    sns.histplot(ax=axes[3], data=data, x='S_B', stat='density', kde=True, bins=50, line_kws=dict(linewidth=1))
    axes[0].set_ylabel(''); axes[1].set_ylabel(''); axes[2].set_ylabel(''); axes[3].set_ylabel('')
    axes[0].set_xlabel('$O_A$', fontsize=15); axes[1].set_xlabel('$O_B$', fontsize=15); axes[2].set_xlabel('$S_A$', fontsize=15); axes[3].set_xlabel('$S_B$', fontsize=15)
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
    sns.set()
#################################################################################################################

def possible_dag(saving_path):
    fig = plt.figure(figsize=(10,4))
    node_size=1200; width=3.0; 
    ax = fig.add_subplot(1,2, 1)
    translator=pd.DataFrame(); translator['original_name']=[0,1,2,3]; translator['converted_name']=[r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$'];  translator['position']=[np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5])]; translator['color']=['r', 'r', 'g', 'g']; df=translator.copy()
    G = nx.DiGraph(np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]])); G = nx.relabel_nodes(G, dict(zip(df.index, df['converted_name'])))           
    edgelist, edge_color = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw_networkx(G, node_color=['r', 'r', 'g', 'g'], node_size=node_size, edgelist=edgelist, width=width, pos=dict(zip(df.converted_name, df['position'])))
    
    ax = fig.add_subplot(1,2, 2)
    translator=pd.DataFrame(); translator['original_name']=[0,1,2,3,4,5,6,7]; translator['converted_name']=[r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$', r'$\lambda$', r'$\mu$', r'$\nu$', r'$\xi$'] ; translator['color']=['r', 'r', 'g', 'g', 'gold', 'gold', 'gold', 'gold']
    translator['position']=[np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5]), np.array([0, +.8]), np.array([0, +0.2]), np.array([0, -0.2]), np.array([0, -.8])]
    A=np.zeros([8,8]); A[0,1]=0; A[0,2]=0; A[0,3]=0; A[0,4]=0; A[0,5]=0; A[1,0]=0; A[1,2]=0; A[1,3]=0; A[1,4]=0; A[1,6]=0; A[2,0]=1; A[2,1]=0; A[2,6]=0; A[2,7]=0; A[3,0]=0; A[3,1]=1; A[3,5]=0; A[3,7]=0; A[4,0]=1; A[4,1]=1; A[5,0]=1; A[5,3]=1; A[6,1]=1; A[6,2]=1; A[7,2]=1; A[7,3]=1; 
    connected_nodes= [0,1,2,3]+[idx for idx in [4,5,6,7] if A[idx,:].sum()]; adj=A[connected_nodes, :][:, connected_nodes]; df=translator.copy(); df=df.iloc[connected_nodes, :].reset_index(drop=True); 
    mapping = dict(zip(df.index, df['converted_name'])); pos=dict(zip(df.converted_name, df['position'])); node_color=df['color'].tolist()
    G = nx.DiGraph(adj); G = nx.relabel_nodes(G, mapping); edgelist, edge_color = zip(*nx.get_edge_attributes(G, 'weight').items()); edge_color=[round(abs(e),2) for e in edge_color]
    nx.draw_networkx(G, pos=pos, node_color=node_color, node_size=node_size, edgelist=edgelist, width=width)
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show(); plt.clf(); plt.close()
#################################################################################################################

def best_DAGS_plotter(DAGs_path, DAGs_num, saving_path):
    paths=sorted([DAGs_path+file for file in os.listdir(DAGs_path) if file.endswith('.png')])[:DAGs_num]
    fig = plt.figure(figsize=(16,3), tight_layout=True)
    for indx, path in enumerate(paths):
        ax = plt.subplot(1, DAGs_num, indx + 1)
        img = plt.imread(path)[100:, :]
        ax.text(x=0.5, y=1, s= str(indx+1), fontweight='bold', horizontalalignment='center', verticalalignment='baseline',  transform=ax.transAxes)
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
    plt.tight_layout()
    plt.savefig(saving_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.close()

def outcomes_connection_plotter(info_path, saving_path):
                       
    df = object_loader(info_path)
    plt.figure(figsize=(8,4))
    palette= ['#5F5F5F', '#FFD700', '#BA0100', '#008001']
    g=sns.kdeplot(data=balancer(df, x='Loss', y='OO_Connection', method='downsample'), x='Loss', hue="OO_Connection", multiple="fill", linewidth=1, common_norm=True, palette=palette, warn_singular=False, hue_order=['disconnect', 'CC', 'CE', 'CE_CC'])
    leg = g.get_legend(); leg.remove()
    labels=['Disconnected','CC','CE', r'CE$\,\wedge\,$CC']
    handles=[Patch(color=palette[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=handles, ncol=4, loc='upper center', fancybox=True, shadow=False, frameon=True, framealpha=.7)
    g.set_ylabel('')
    g.set_xlabel('MMD Loss', weight='bold')
    g.set_yticks([0, 0.5, 1.0])
    plt.xticks([], [])
    plt.savefig(saving_path, dpi = 300, bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
