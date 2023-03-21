import cgnn_functions as cgnnFun
import distance_functions as dFun
##############################################################################
import numpy as np
import ast
import os
import pandas as pd; pd.set_option('max_columns', 100); pd.set_option('max_rows', 100)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


# Define relavent variables 
candidates_main_path='./Data/candidates_m.txt'
candidates_tuning_path='./Data/candidates_tuning.txt'


# Load the set of all candidate DAGs (with 207 elements), and Draw 16 sample DAGs:
array_cands=np.loadtxt(candidates_main_path)
sample_cands_inx=[0, 6, 9, 13, 15, 29, 36, 45, 109, 111, 112, 116, 119, 138, 156, 192]
np.savetxt(candidates_tuning_path,array_cands[sample_cands_inx, :], fmt='%f')

# Choose a range for testing nh:
nh_range=range(1,60,3)

# Run the CGNN algorithm for the 16 sample DAGs and for the given nh_range => save the resulting txt files on disk
for nh in nh_range: 
    batch_size=200
    criterion_1 = dFun.MMD_cdt(batch_size)
    criterion_2 = dFun.CorrD(sampling_rate=4, wg_penalty=1)
    CGNN_obj=cgnnFun.CGNN(
        batch_size=batch_size, nh=nh, lr=0.01, guassian_w=1, uniform_w=0, patience=20, criterion_1=criterion_1, criterion_2=criterion_2, wg_criterion_2=1, train_epochs=500, test_epochs=20, nruns=1, save_training_details=False)
    CGNN_obj.create_graph_from_data(
        train_sample_size=2000, test_sample_size=1000, gen_sample_size=1000, data_paths=['./Data/train.csv', './Data/val.csv', './Data/test.csv'], candidates_path=candidates_tuning_path,
        saving_dir='./nh_tuning/{}/'.format(nh))


nh_values=sorted([ast.literal_eval(filename) for filename in os.listdir('./Tuning') if not filename.startswith('.')])
metrics=['loss_1', 'calib_1', 'qual_1', 'loss_2', 'calib_2', 'qual_2', 'qual']
inx_nh_threshold=len(nh_values)-1
inx_cand_threshold=16


# Set details of Visualization
colors=sns.color_palette("hls", inx_cand_threshold); ci=None; n_boot=10000; order=3; scatter_kws={'s': 1}; line_kws={'lw': 1.2}; 

# Create a meta-dataframe (df) containg the preformance of each candidate DAG for each possible nh 
# => rows and columns of df correspond to nh and cands respectively
# => each cell of df is a dictionary whose keys are ['loss_1', 'calib_1', 'qual_1', 'loss_2', 'calib_2', 'qual_2', 'qual']

df=pd.DataFrame()
df['nh']=nh_values
for cand in range(len(sample_cands_inx)): 
    df['cand_{}'.format(cand)]=None
cands_cols=[column for column in df.columns if column.startswith('cand_')]
cands_metric=[[] for cand in range(len(cands_cols))]

for inx_nh, nh in enumerate(nh_values):
    if inx_nh<inx_nh_threshold:
        solution_path="./nh_tuning/{}/solutions.txt".format(nh)
        df_solution=pd.DataFrame(np.loadtxt(solution_path)[:, 1:8])
        df_solution.rename(columns=dict(zip(range(len(metrics)), metrics)), inplace=True) 
        for inx_cand, cand in enumerate(cands_cols):
            df.loc[inx_nh, cand]=[{metric: df_solution.loc[inx_cand, metric] for metric in metrics}]

# Define 3 monitoring metrics ['qual_1', 'qual_2', 'qual'] for each pair of cand_nh
# => Visualize these metrics separately for each pair of cand_nh 

monitoring_metrics=['qual_1', 'qual_2', 'qual']
fig, axes = plt.subplots(nrows=len(monitoring_metrics), ncols=1, figsize=(10, 10))
for inx_metric, monitoring_metric in enumerate(monitoring_metrics):
    df_monitoring=pd.DataFrame(); df_monitoring['nh']=nh_values
    for inx_nh, nh in enumerate(nh_values):
        if inx_nh<inx_nh_threshold:
            for inx_cand, cand in enumerate(cands_cols):
                df_monitoring.loc[inx_nh, cand] = df.loc[inx_nh, cand][monitoring_metric]
    for inx_cand, cand in enumerate(cands_cols[:inx_cand_threshold]):
        sns.regplot(ax=axes[inx_metric], x="nh", y=cand, data=df_monitoring, label=str(cand), ci=ci, n_boot=n_boot, order=order, scatter_kws=scatter_kws, line_kws=line_kws, color=colors[inx_cand])
        axes[inx_metric].text(x=.5, y=.1, s=monitoring_metric + '  vs. nh', weight='bold',  ha='center', va='center', transform=axes[inx_metric].transAxes, fontsize=20);
        axes[inx_metric].set_ylabel(''); axes[inx_metric].set_xlabel(''); 

fig.tight_layout() 
plt.savefig('./Plots/tuner_multivariate.png', dpi=300)
plt.show()
plt.close()