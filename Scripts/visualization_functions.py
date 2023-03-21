import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th
import pandas as pd
import networkx as nx
import utilities_functions as uFun
import distance_functions as dFun
import fcit
# import cdt
# cdt.SETTINGS.rpath="/Library/Frameworks/R.framework/Versions/4.0/Resources/Rscript"
# from cdt.causality.graph import (PC,GES,MMPC,IAMB,GS,LiNGAM,)
sns.set()

latex_dict = {
    'dataTR_measTR': r'$\mathcal{M}_{tr}(\mathscr{D}_{tr}, \hat{\mathscr{D}})$',
    'dataTE_measTR': r'$\mathcal{M}_{tr}(\mathscr{D}_{te}, \hat{\mathscr{D}})$',
    'dataTR_measUN': r'$\mathcal{M}_{un}(\mathscr{D}_{tr}, \hat{\mathscr{D}})$',
    'dataTE_measUN': r'$\mathcal{M}_{un}(\mathscr{D}_{te}, \hat{\mathscr{D}})$',
    'dataTR_measTE': r'$\mathcal{M}_{te}(\mathscr{D}_{tr}, \hat{\mathscr{D}})$',
    'dataTE_measTE': r'$\mathcal{M}_{te}(\mathscr{D}_{te}, \hat{\mathscr{D}})$',
}

def customBarPlot(
    axes, data, labelsSets, labelsVals,
    errorBars=[], errorAnchorPoints=[],
    logPlot=False, enforceZero=False, norm_mode='none',
):
    # config
    bar_space = 0.8
    # determine size of data
    numSets = data.shape[0]
    numVals = data.shape[1]
    if not numVals == len(labelsVals):
        print('ERROR mismatched numbers of sets for customBarPlot')
        return
    if not numSets == len(labelsSets):
        print('ERROR mismatched numbers of values for customBarPlot')
        return
    # prepare x-values
    x_vals = list(range(numVals))
    axes.set_xticks(x_vals)
    axes.set_xticklabels(labelsVals)
    x_vals = np.array(x_vals)
    # prepare bar parameters
    bar_width = bar_space / numSets
    # normalize data to dataset
    normalize = True
    if isinstance(norm_mode, str):
        if norm_mode == 'min':
            try:
                norm_vec = np.min(data, 0).reshape([1, numVals])
            except:
                norm_vec = th.min(data, 0).reshape([1, numVals])
        elif norm_mode == 'max':
            try:
                norm_vec = np.max(data, 0).reshape([1, numVals])
            except:
                norm_vec = th.max(data, 0).reshape([1, numVals])
        elif norm_mode == 'mean':
            try:
                norm_vec = np.mean(data, 0).reshape([1, numVals])
            except:
                norm_vec = th.mmean(data, 0).reshape([1, numVals])
        if norm_mode == 'minAnch' and len(errorAnchorPoints) > 0:
            try:
                norm_vec = np.min(errorAnchorPoints, 0).reshape([1, numVals])
            except:
                norm_vec = th.min(errorAnchorPoints, 0).reshape([1, numVals])
        else:
            normalize = False
    elif norm_mode >= 0:
        norm_vec = data[[norm_mode], :]
    else:
        normalize = False
    if normalize:
        pltDat = data / norm_vec
        if len(errorBars) > 0:
            pltErr = errorBars / norm_vec
            if len(errorAnchorPoints) > 0:
                pltErrAnch = errorAnchorPoints / norm_vec
    else:
        pltDat = data
        pltErr = errorBars
        pltErrAnch = errorAnchorPoints
    # plot
    for k in range(numSets):
        if len(errorBars) == 0:
            axes.bar(x_vals - bar_space/2 + k*bar_width, pltDat[k, :], width=bar_width)
        else:
            if len(errorAnchorPoints) == 0:
                axes.bar(x_vals - bar_space/2 + (k+0.5)*bar_width, pltDat[k, :], width=bar_width, yerr=pltErr[k, :])
            else:
                axes.bar(x_vals - bar_space/2 + (k+0.5)*bar_width, pltDat[k, :], width=bar_width)
                axes.bar(x_vals - bar_space/2 + (k+0.5)*bar_width, pltErrAnch[k, :], width=bar_width/3, yerr=pltErr[k, :], color='black', label='_nolegend_')
        if logPlot:
            axes.set_yscale('log')
        elif enforceZero:
            try:
                axes.set_ylim(0, np.max(pltDat)*1.05)
            except:
                axes.set_ylim(0, th.max(pltDat)*1.05)
    axes.legend(labelsSets, prop={'size': 7.5})


def plot_progress(df, cr_names, axes):
    limitUpp = 20
    if cr_names is None:
        cr_names = df.columns.tolist()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, cr_name in enumerate(cr_names):
        data = df.loc[:, cr_name]
        ax = axes[k]
        ax.plot(range(len(df)), data, ls='-', marker='*', ms=1, label=cr_name, c=colors[k])
        ax.axhline(1, c='black')
        ax.axhline(10, c='black')
        if len(data[data <= limitUpp]) >= np.floor(len(data)/2):
            ax.set_ylim(0, limitUpp)
        ax.set_xticks([])
        ax.set_yticks([])


'''
def plot_progress(df_loss, saving_path):
    n_epochs = df_loss.shape[0]
    cr_names = df_loss.columns.tolist()
    loss_values = df_loss.values

    x_range = list(range(n_epochs))
    max_plot_crit = 8
    valBnchmrk = 10
    limitUpp = 20
    limitUppMarg = 20

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), tight_layout=True)
    for k, cr_name in enumerate(cr_names):
        if k == max_plot_crit:
            break
        row = int(np.floor(k/4))
        col = k % 4
        data = loss_values[:n_epochs, k]
        axes[row, col].plot(x_range, data, ls='-', marker='.', ms=10, label=cr_name, c=colors[k])
        axes[row, col].axhline(1, c='black')
        axes[row, col].axhline(valBnchmrk, c='black')
        if not cr_name == 'EDF_Mrg':
            if len(data[data <= limitUpp]) >= np.floor(len(data)/2):  # check if at least half of the data is displayed in y-Range
                axes[row, col].set_ylim(0, limitUpp)
        else:
            if len(data[data <= limitUppMarg]) >= np.floor(len(data)/2):  # check if at least half of the data is displayed in y-Range
                axes[row, col].set_ylim(0, limitUppMarg)
        axes[row, col].set_ylabel('')
        axes[row, col].set_xlabel(cr_name, fontsize=12)
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
'''


def plot_marginals(data, saving_path=None):
    # [CONFIG]
    xlim_obs = [-5, 5]
    xlim_set = [-0.5, 1.5]
    labels = ['$O_A$', '$O_B$', '$S_A$', '$S_B$']
    xlims = [xlim_obs, xlim_obs, xlim_set, xlim_set]
    data = pd.DataFrame(data, columns=labels)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        ax = axes[i]
        sns.histplot(
            data=data,
            x=labels[i],
            stat='density',
            kde=True,
            bins=20,
            line_kws=dict(linewidth=1),
            ax=ax,
            )
        ax.set_ylabel('')
        ax.set_xlabel(labels[i], fontsize=15)
        ax.set_xlim(xlims[i])
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_corrs(df_data, axes):
    pass


def plot_grad_flow(named_parameters):
    # copied from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()
    plt.close()


def plot_citests(data, saving_path=None, cv_grid=None, num_perm=50):
    # cv_grid=[1e-2, .2, .4, 2, 8, 64, 512]

    OA = data[:, 0:1]
    OB = data[:, 1:2]
    SA = data[:, 2:3]
    SB = data[:, 3:4]
    OA_OB = np.concatenate((OA, OB), axis=1)
    OA_SA = np.concatenate((OA, SA), axis=1)
    OA_SB = np.concatenate((OA, SB), axis=1)
    OB_SA = np.concatenate((OB, SA), axis=1)
    OB_SB = np.concatenate((OB, SB), axis=1)
    SA_SB = np.concatenate((SA, SB), axis=1)
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
    colors = ['#BA0100', '#008001']
    labels = ['Marginal', 'Conditional']
    markers = ['.', '*']
    plt.figure(figsize=(8, 4))
    plt.subplot()
    for i in range(6):
        for j in range(2):
            if i == 0:
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


def plot_indep_tests(data, test_type=2, ax=None, num_perm=50, label='orig', marker='*', color='#008001', s=200):
    pval_dict = uFun.perform_independence_test(
        data=data, test_type=test_type, num_perm=num_perm, cv_grid=None)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.set_xticks(np.arange(len(pval_dict)), pval_dict.keys(), rotation=20, fontsize=12)
    for k, pval in enumerate(pval_dict.values()):
        ax.scatter(
            k, pval, c=color, marker=marker, s=s,
            label=label if k==0 else None)
    return pval_dict


def plot_heatmaps(data, axes, sampling_rate=25):
    corr_type = '(2,2)'
    legend_dict = {
        '01_23': '$(O_A, O_B)|(S_A, S_B)$',
        '02_13': '$(O_A, S_A)|(O_B, S_B)$',
        '03_12': '$(O_A, S_B)|(O_B, S_A)$',
        '12_03': '$(O_B, S_A)|(O_A, S_B)$',
        '13_02': '$(O_B, S_B)|(O_A, S_A)$',
        '23_01': '$(S_A, S_B)|(O_A, O_B)$',
    }
    self = uFun.CorrPatterns(sampling_rate=sampling_rate)
    patterns = self.get_patterns(data, corr_type)

    for i, (conditional, label) in enumerate(legend_dict.items()):
        ax = axes[i]
        ax.pcolormesh(
            patterns[conditional]['cv0'],
            patterns[conditional]['cv1'],
            patterns[conditional]['corr'],
            label=label,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    return legend_dict


def _visualize_adjacency(A, ax, for_cgnn):
    n_nodes = A.shape[0]
    if for_cgnn:
        orig_names = [i for i in range(n_nodes)]
        conv_names = [r'$O_A$', r'$O_B$', r'$S_A$', r'$S_B$'] + [r'$\Lambda_{}$'.format(i-4) for i in range(4, n_nodes)]
        colors = ['green']*4 + ['gold']*(n_nodes-4)
        positions = [np.array([-1, +0.5]), np.array([+1, +0.5]), np.array([-1, -0.5]), np.array([+1, -0.5])] + [np.array([0, y]) for y in np.linspace(+.8, -.8, n_nodes-4)]
        G = nx.DiGraph(A)
        G = nx.relabel_nodes(
            G,
            dict(zip(orig_names, conv_names)))
        nx.draw_networkx(
            G,
            pos=dict(zip(conv_names, positions)),
            node_color=colors,
            node_size=1000,
            width=2,
            ax=ax,
        )
    else:
        G = nx.DiGraph(A)  # G = nx.MultiDiGraph(A)
        G = nx.relabel_nodes(
            G,
            {i: r'${}$'.format(i) for i in range(n_nodes)})
        nx.draw_networkx(
            G,
            pos=nx.circular_layout(G),
            node_color='gold',
            node_size=1000,
            width=2,
            ax=ax,
        )


def visualize_adjacency(As, for_cgnn=True, figsize_single=(3, 3), ax=None, saving_path=None):
    if not isinstance(As, list):
        As = [As]
    n_graphs = len(As)
    if ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=n_graphs,
            figsize=(figsize_single[0]*n_graphs, figsize_single[1]),
            tight_layout=True,
        )
    for index_A, A in enumerate(As):
        _visualize_adjacency(
            A=A,
            ax=ax if n_graphs == 1 else ax[index_A],
            for_cgnn=for_cgnn,
        )
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')


'''
def plot_candidate_losses(
    df, cand_names, labels=None, datasets=['tr', 'va', 'te'],
    show_errors=False, ax=None, saving_path=None,
):
    df = df.copy().set_index('cand_name').loc[cand_names].reset_index()
    if labels is None:
        labels = cand_names
    df['labels'] = labels
    n_cands = len(cand_names)
    n_datasets = len(datasets)
    x = np.arange(n_cands)
    width = 0.2
    shift_counter = 0
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(n_cands*n_datasets*0.5, 4),
            tight_layout=True,
        )

    if 'tr' in datasets:
        ax.bar(
            x=x+width*shift_counter,
            height=df['loss_tr'],
            yerr=df['loss_tr_std'] if show_errors else None,
            width=width,
            color='#165C8D',
            label='Training Loss',
        )
        shift_counter = shift_counter+1

    if 'va' in datasets:
        ax.bar(
            x=x+width*shift_counter,
            height=df['loss_va'],
            yerr=df['loss_va_std'] if show_errors else None,
            width=width,
            color='#167F0F',
            label='Validation Loss',
        )
        shift_counter = shift_counter+1

    if 'te' in datasets:
        ax.bar(
            x=x+width*shift_counter,
            height=df['loss_te'],
            yerr=df['loss_te_std'] if show_errors else None,
            width=width,
            color='#CE0917',
            label='Test Loss',
        )
        shift_counter = shift_counter+1

    ax.set_xticks(
        x+width*shift_counter/2,
        df['labels'],
        weight='bold',
        fontsize=14,
        )
    ax.legend(loc='best')

    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    if ax is None:
        plt.show()
        plt.close()
'''

def plot_varying_rate_performance(
    data_t, data_q, data_c, sampling_rates=list(range(3, 30)), saving_path=None,
):
    distances_tt = []
    distances_tq = []
    distances_tc = []
    for sampling_rate in sampling_rates:
        ev = uFun.Measure_Distance(
            criterion=dFun.CorrD(
                corr_types=[1, 1],
                sampling_rate=sampling_rate),
            n_epochs=2,
        )
        distances_tt.append(ev.forward(data_t, data_t))
        distances_tq.append(ev.forward(data_t, data_q))
        distances_tc.append(ev.forward(data_t, data_c))
    distances_tt = np.array(distances_tt)
    distances_tq = np.array(distances_tq)
    distances_tc = np.array(distances_tc)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), tight_layout=True)
    ax.plot(sampling_rates, (distances_tq/distances_tt)-1, label='qCC')
    ax.plot(sampling_rates, (distances_tc/distances_tt)-1, label='cCC')
    ax.legend()
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_loss_portions(df_loss, ax):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    portions = df_loss.mean()/df_loss.mean().sum()
    ax.pie(
        x=portions,
        colors=colors,
    )
    sns.set_style("darkgrid")

def plot_candidate_losses(df, candidates_list, comparison_strings_list, colors_list, show_errors=True, saving_path=None):

    df = df.loc[candidates_list, :].copy()
    x = np.arange(len(candidates_list))
    width = 0.2
    shift_counter = 0
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), tight_layout=True)

    for index_string, string in enumerate(comparison_strings_list):
        ax.bar(
            x=x+width*shift_counter,
            height=df[f'avg_{string}'],
            yerr=df[f'std_{string}'] if show_errors else None,
            width=width,
            color=colors_list[index_string],
            label=latex_dict[string],
        )
        shift_counter = shift_counter+1
    ax.set_xticks(
        x+width*shift_counter/2,
        candidates_list,
        weight='bold',
        fontsize=14,
    )
    ax.legend(loc='upper left')
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_stacked_loess(df, candidates_list, cr_names, saving_path=None):
    df = df.copy().loc[candidates_list, cr_names]
    x = np.arange(len(candidates_list))
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    width = 0.3
    for i, cr in enumerate(cr_names):
        ax.bar(
            x=x,
            height=df.loc[:, cr],
            width=width, label=cr,
            bottom=df.loc[:, :cr_names[i-1]].sum(1) if i > 0 else None,
            )
    ax.set_xticks(
        x,
        candidates_list,
        weight='bold',
        fontsize=14,
        )
    ax.legend(loc='best', fontsize=8)
    ax.set_yticks([])
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


'''
def plot_stacked_loess(df, cand_names, cand_labels, cr_names, weight_cr, saving_path):
    wg_dict_all = {'MMD_cdt': 1.0, 'MMD_Fr': 4.90, 'CorrD': 23.75, 'BinnedD': 10.89, 'CorrD_N': 2.15, 'NpMom': 1.03, 'CndD': 6.85, 'EDF_Mrg': 4.28}
    if weight_cr:
        cr_weights = {k: v for k, v in wg_dict_all.items() if k in cr_names}
    else:
        cr_weights = {k: 1 for k, v in wg_dict_all.items() if k in cr_names}
    cr_weights = pd.Series(cr_weights)/pd.Series(cr_weights).sum()

    df = df.copy().set_index('cand_name').loc[cand_names].reset_index()
    if cand_labels is None:
        cand_labels = cand_names
    df['cand_label'] = cand_labels
    n_cands = len(df)
    x = np.arange(n_cands)
    #cr_names = pd.read_csv(df['loss_te_link'].tolist()[0]).columns.tolist()
    df_te = pd.DataFrame(index=cr_names, columns=cand_names)
    for index_cand, cand in enumerate(cand_names):
        loss_mean = pd.read_csv(df.loc[index_cand, 'loss_te_link']).mean()
        df_te[cand] = loss_mean*cr_weights
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    width = 0.3
    for i, cr in enumerate(cr_names):
        ax.bar(
            x=x,
            height=df_te.loc[cr, :],
            width=width,
            label=cr,
            bottom=df_te.loc[:cr_names[i-1], :].sum() if i > 0 else None,
            )
    ax.set_xticks(
        x,
        df['cand_label'],
        weight='bold',
        fontsize=10,
        )
    ax.legend(loc='best')
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def visualize_separate_performances(
    df_sep_perfs,
    cand_names,
    labels=None,
    criteria_sets=['seen', 'unseen', 'mixed'],
    show_errors=False,
    ax=None,
    saving_path=None,
):
    df = df_sep_perfs.copy().set_index('cand_name').loc[cand_names].reset_index()
    if labels is None:
        labels = cand_names
    df['labels'] = labels
    n_cands = len(cand_names)
    n_datasets = len(criteria_sets)
    x = np.arange(n_cands)
    width = 0.2
    shift_counter = 0
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(n_cands*n_datasets*0.5, 4),
            tight_layout=True,
        )
    if 'seen' in criteria_sets:
        ax.bar(
            x=x+width*shift_counter,
            height=df['loss_te_seen'],
            yerr=df['loss_te_std_seen'] if show_errors else None,
            width=width,
            color='#165C8D',
            label='seen criteria',
        )
        shift_counter = shift_counter+1
    if 'unseen' in criteria_sets:
        ax.bar(
            x=x+width*shift_counter,
            height=df['loss_te_unseen'],
            yerr=df['loss_te_std_unseen'] if show_errors else None,
            width=width,
            color='#167F0F',
            label='unseen criteria',
        )
        shift_counter = shift_counter+1
    if 'mixed' in criteria_sets:
        ax.bar(
            x=x+width*shift_counter,
            height=df['loss_te_mixed'],
            yerr=df['loss_te_std_mixed'] if show_errors else None,
            width=width,
            color='#CE0917',
            label='mixed criteria',
        )
        shift_counter = shift_counter+1
    ax.set_xticks(
        x+width*shift_counter/2,
        df['labels'],
        weight='bold',
        fontsize=14,
        )
    ax.legend(loc='best')

    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    if ax is None:
        plt.show()
        plt.close()
'''


def plot_conditionals(data_true, data_pred, sampling_rate, corr_type, ax):
    legend_dict = {
        '01_23': f'$(O_A, O_B)|(S_A, S_B)$',
        '02_13': f'$(O_A, S_A)|(O_B, S_B)$',
        '03_12': f'$(O_A, S_B)|(O_B, S_A)$',
        '12_03': f'$(O_B, S_A)|(O_A, S_B)$',
        '13_02': f'$(O_B, S_B)|(O_A, S_A)$',
        '23_01': f'$(S_A, S_B)|(O_A, O_B)$',
        '0_123': f'$O_A |(O_B, S_A, S_B)$',
        '1_023': f'$O_B |(O_A, S_A, S_B)$',
        '2_013': f'$S_A |(O_A, O_B, S_B)$',
        '3_012': f'$S_B |(O_A, O_B, S_A)$',
    }

    distance_dict = uFun.decompose_conditionals(data_true, data_pred, sampling_rate, corr_type)
    distances = {}
    for k, v in distance_dict.items():
        if k in legend_dict.keys():
            distances[legend_dict[k]] = v
    ax.pie(
        x=list(distances.values()),
        # colors=plt.rcParams['axes.prop_cycle'].by_key()['color'],
    )
    return distances


def visualize_marginal_distance(data, ax):
    import cdt
    cdt.SETTINGS.rpath="/Library/Frameworks/R.framework/Versions/4.0/Resources/Rscript"
    from cdt.independence.stats import (AdjMI, KendallTau, PearsonCorrelation, SpearmanCorrelation)
    measures = {
        'Mutual Information': AdjMI(),
        'Kendall Tau': KendallTau(),
        'Pearson Correlation': PearsonCorrelation(),
        'Spearman Correlation': SpearmanCorrelation(),
    }
    metrics = list(measures.values())
    labels = list(measures.keys())
    colors = ['b', 'g', 'r', 'k', 'y']
    # sns.color_palette("husl", len(measures))

    meas_dict = uFun.measure_marginal_distance(data, metrics)

    for i, values in enumerate(meas_dict.values()):
        for j, value in enumerate(values):
            ax.scatter(
                i, value, color=colors[j], label=labels[j] if i == 0 else None,
            )
    return meas_dict


def plot_cndDX(df, candidates_list, relation_types, saving_path=None):
    sns.set_style("dark")
    df = df.loc[candidates_list, relation_types]
    n_cands = len(candidates_list)
    x = np.arange(n_cands)
    fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    for i, relation in enumerate(relation_types):
        ax.bar(
            x=x,
            height=df.loc[:, relation],
            width=.3,
            label=relation,
            bottom=df.loc[:, :relation_types[i-1]].sum(1) if i > 0 else None,
            )
    ax.set_xticks(
        x,
        candidates_list,
        weight='bold',
        fontsize=10,
        )
    ax.legend(loc='best', fontsize=6)
    ax.set_yticks([])
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
