import os
import numpy as np
import matplotlib.pyplot as plt
import time
from visualization_functions import customBarPlot
######################################################################################

class Profiler:
    def __init__(self, nb_cand, nb_runs, epochs, save_dir=''):
        # [CONFIG]
        self.save_dir = save_dir
        self.max_plot_cand = 5
        self.bar_space = 0.8
        # [HISTORICAL DURATIONS]
        self.hist_durations = np.array([[0.389, 0,0.592, 0,0.602], [0.365, 0,0.545, 0, 0.569], [0.360, 0,0.552, 0, 0.574], [0.406, 0.013, 0.560, 0.005, 0.111]])
        self.hist_legend = ['phaseA', 'phaseB', 'phaseC', 'phaseD', 'current']
        # [CODE]
        self.inds = {'bp': 0, 'gen_tr': 1, 'ev_tr': 2, 'gen_va': 3, 'ev_va': 4, 'gen_te': 5, 'ev_te': 6,'sav': 7,'rest': 8}
        # LEGEND: [backprop, gen_train, eval_train, gen_valid, eval_valid, gen_test, eval_test, saving]
        self.labels = 'Backprop', 'Gen-Train', 'Eval-Train', 'Gen-Valid', 'Eval-Valid', 'Gen-Test', 'Eval-Test', 'Save', 'Rest'
        self.nb_cand = nb_cand
        self.nb_runs = nb_runs
        self.epochs = epochs
        self.reset()

    def reset(self):
        self.durations_gen = np.zeros([self.nb_cand, 4]) # testing, saving & rest
        self.durations_tra = np.zeros([self.nb_cand, self.nb_runs, self.epochs, 5]) #backprop,training and validation

    def set_candidate(self, cand_ind):
        self.cand_ind = cand_ind

    def set_run(self, run_ind):
        self.run_ind = run_ind

    def start(self):
        self.start_time = time.time()

    def start_global(self):
        self.start_time_global = time.time()

    def stop(self, val_str, epoch=None):
        if val_str == 'gen_va' or val_str == 'ev_va':
            self.stop_tr(val_str, epoch)
        else:
            val_ind = self.inds[val_str]-5
            self.durations_gen[self.cand_ind, val_ind] = self.durations_gen[self.cand_ind,val_ind] + (time.time() - self.start_time)

    def stop_tr(self, val_str, epoch):
        val_ind = self.inds[val_str]
        self.durations_tra[self.cand_ind, self.run_ind, epoch,val_ind] = self.durations_tra[self.cand_ind,self.run_ind,epoch,val_ind] + (time.time() - self.start_time)

    def stop_global(self):
        self.durations_gen[self.cand_ind,3] = time.time() - self.start_time_global

    def get_current_tot_time(self):
        return self.durations_gen[self.cand_ind,3]

    def plot_results(self):
        #training times for all candidates over epochs (mean over runs)
        durations_cand_mean = np.mean(self.durations_tra,axis=1)
        fig_cand, ax_cand = plt.subplots(2,3)
        for k in range(self.nb_cand):
            if k == self.max_plot_cand:
                break
            row = int(np.floor(k/3));
            col = k % 3;
            for l in range(5):
                ax_cand[row, col].plot(durations_cand_mean[k,:,l])
            ax_cand[row, col].legend(['bckpr', 'gen_train', 'ev_train', 'gen_valid','ev_valid'], fontsize=8)    
            ax_cand[row, col].set_title('c' + str(k))
            ax_cand[row, col].set_ylim([0, 1])
        plt.savefig(os.path.join(self.save_dir, 'dur_epochs.png'),dpi=150, bbox_inches='tight')

        #mean training durations per epoch
        fig_tra, ax_tra = plt.subplots()
        durations_cand_mean = np.mean(self.durations_tra, axis=(0,1,2))
        plot_data = np.concatenate((self.hist_durations, durations_cand_mean.reshape([1,len(durations_cand_mean)])),axis=0)
        customBarPlot(ax_tra, plot_data, self.hist_legend, ['bckpr', 'gen_tr', 'ev_tr', 'gen_val', 'ev_val'])
        plt.savefig(os.path.join(self.save_dir, 'dur_training.png'), dpi=150, bbox_inches='tight')

        print('Current Training Durations:')
        print('bp: {:.03f}, gen_tr: {:.03f}, ev_tr: {:.03f}, gen_va: {:.03f}, ev_va: {:.03f}'.format(durations_cand_mean[0],durations_cand_mean[1],durations_cand_mean[2],durations_cand_mean[3],durations_cand_mean[4]))
        #duration ratios for all candidates and global
        durations_tra_sum = np.sum(self.durations_tra,axis=(1,2))
        sum_durations = np.concatenate((durations_tra_sum,self.durations_gen),axis=1)
        tot_durations = sum_durations[:,8];
        sum_durations[:,8] = tot_durations - np.sum(sum_durations[:,0:8],axis=1)
        mean_durations = np.mean(sum_durations,0)
        fig_pies, ax_pies = plt.subplots(2,3)
        for k in range(self.nb_cand):
            if k == self.max_plot_cand:
                break
            row = int(np.floor(k/3));
            col = k % 3;
            ax_pies[row,col].pie(sum_durations[k,:],labels=self.labels,autopct='%1.1f%%',shadow=False,startangle=90,textprops=dict(size=8))
            ax_pies[row,col].set_title('c' + str(k))
        ax_pies[1,2].pie(mean_durations,labels=self.labels,autopct='%1.1f%%',shadow=False,startangle=90,textprops=dict(size=8))
        ax_pies[1,2].set_title('global')
        plt.savefig(os.path.join(self.save_dir, 'dur_ratios.png'), dpi=150, bbox_inches='tight')
