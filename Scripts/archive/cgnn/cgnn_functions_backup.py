import utilities_functions as uFun
import distance_functions as dFun
import numpy as np
import pandas as pd #; pd.set_option('max_columns', 100); pd.set_option('max_rows', 100)
import networkx as nx
import torch as th; from torch.utils.data import DataLoader
from tqdm import tqdm #library for showing progress bars
import os; from pathlib import Path #for creation of directories and saving data

##############################################################################
class CGNN_block(th.nn.Module):
    def __init__(self, ind_node, col_inc):
        super().__init__()
        #[CONFIG]
        self.enforce_bounds_for_settings = False
        self.bounds = [0,1]
        #process input
        self.ind_inc = np.nonzero(col_inc)[0]
        self.has_inc = (self.ind_inc.size > 0)
        self.ind_node = ind_node
        self.is_setting_var = (ind_node == 2 or ind_node == 3)
        
    def build(self, hu_base_nb, hu_wg_noise, hu_wg_conn):
        #calculate number of hidden units
        nb_inc = len(self.ind_inc)
        nb_hu = np.round(hu_base_nb * (hu_wg_noise + nb_inc*hu_wg_conn))
        #create network
        layers = []
        layers.append(th.nn.Linear(nb_inc+1,nb_hu)) #transformation from incoming connections and noise (+1) to hidden units            
        layers.append(th.nn.ReLU()) #ReLU on each hidden unit
        layers.append(th.nn.Linear(nb_hu,1)) #transformation from hidden units to output
        self.layers = th.nn.Sequential(*layers)
        return nb_hu
    
    def forward(self, gen_data, noise):
        inc_data = th.cat((noise[:,[self.ind_node]],gen_data[:,self.ind_inc]),axis=1) #collect all incoming data
        result = self.layers(inc_data) #pass data through layers
        if self.enforce_bounds_for_settings and self.is_setting_var:
            result = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * (result - result.min()) / (result.max() - result.min()) 
        gen_data[:,self.ind_node] = th.squeeze(result) #update gen_data tensor
    
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
##############################################################################
                             
class CGNN_model(th.nn.Module):
    def __init__(self, adjacency_matrix, hu_base_nb = 20, hu_wg_noise = 1, hu_wg_conn = 0): 
        super().__init__()
        self.build(adjacency_matrix,hu_base_nb,hu_wg_noise,hu_wg_conn)
        self.adjacency_matrix = adjacency_matrix #temp

    def build(self, adjacency_matrix, hu_base_nb, hu_wg_noise, hu_wg_conn):
        self.nb_vars = adjacency_matrix.shape[0]
        self.topological_order = list(nx.topological_sort(nx.DiGraph(adjacency_matrix)))
        self.blocks = th.nn.ModuleList()
        nb_hu_list = th.zeros([4])
        for k in range(4):
            self.blocks.append(CGNN_block(k,adjacency_matrix[:,k]))
            nb_hu_list[k] = self.blocks[-1].build(hu_base_nb,hu_wg_noise,hu_wg_conn)
        print('Model constructed! hidden untis:',nb_hu_list.tolist(),'tot:',str(th.sum(nb_hu_list).item()))
        return nb_hu_list
        
    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()
        print('Model paremeters reset.')
        
    def generate(self, gen_size):
        #generate noise
        noise_obs = th.zeros(gen_size,2).normal_(0, 1) #for setting variables
        noise_set = th.zeros(gen_size,2).uniform_(0, 1) #for setting variables
        #hidden_noise = th.zeros(gen_size,self.nb_vars-4).normal_(0, 1) #for observables and hidden variables
        noise = th.cat((noise_obs,noise_set),axis=1)
        #noise = th.zeros(gen_size, self.nb_vars).normal_(0, 1) #temp
        #generate data
        gen_data = th.zeros([gen_size,4])
        for node in self.topological_order:
            if node < 4:
                self.blocks[node].forward(gen_data,noise)
        return gen_data
    
##############################################################################
class CGNN:
    def __init__(self, nh, lr, patience, criteria_train, criteria_train_wgs, data_paths, data_paths_calib, data_path_saving, show_cand_plot, evaluate_training):
        super().__init__()
        #store parameters
        self.nh = nh
        self.lr = lr
        self.patience = patience
        self.data_path_saving = data_path_saving
        self.eval_training = evaluate_training
        #sort criteria into regular and mmd
        self.cr = []
        self.cr_wgs = []
        self.cr_MMD = []
        self.cr_wgs_MMD = []
        for c,wg in zip(criteria_train,criteria_train_wgs):
            if c.is_MMD():
                self.cr_MMD.append(c)
                self.cr_wgs_MMD.append(wg)
            else:
                self.cr.append(c)
                self.cr_wgs.append(wg)
        self.cr_wgs = th.tensor(self.cr_wgs)
        self.cr_wgs_MMD = th.tensor(self.cr_wgs_MMD)
        #initialize evaluator (evaluator loads all data)
        self.ev = Evaluator(data_paths,data_paths_calib,data_path_saving,show_cand_plot)
        #store metadata in panda dataframe for saving
        self.df_meta = pd.DataFrame()
        self.df_meta['nh'] = [self.nh]; self.df_meta['lr'] = [self.lr]; self.df_meta['patience'] = [self.patience]
        self.df_meta['criteria_names'] = [[c.get_name(False) for c in self.cr]]; self.df_meta['criteria_weights'] = [[self.cr_wgs]]
        self.df_meta['criteria_names_MMD'] = [[c.get_name(False) for c in self.cr_MMD]]; self.df_meta['criteria_weights_MMD'] = [[self.cr_wgs_MMD]]

    def evaluate_all_candidates(self, nb_runs, train_epochs, test_epochs, batch_size, batch_size_MMD, train_sample_size, valid_sample_size, test_sample_size, calib_sample_size, gen_sample_size, candidates_path): 
        #preallocate memory for criteria
        for c in self.cr:
            c.preallocate_memory(batch_size)
        for c in self.cr_MMD:
            c.preallocate_memory(batch_size_MMD)
        self.ev.preallocate_memory(batch_size,batch_size_MMD)
        #load candidates file, setup profiler and pass parameters to evaluator
        candidates = np.loadtxt(candidates_path)
        if candidates.ndim == 1:
            #convert to 2Dim array if only one candidate present
            candidates = candidates[None,...]
        nb_vars = np.sqrt(candidates.shape[1]).astype(int)
        self.prof = uFun.Profiler(len(candidates),nb_runs,train_epochs) #[PROFILING]
        #prepare evaluator and sample data
        self.ev.set_containers(self.prof,len(candidates),nb_runs,train_epochs,test_epochs)
        print('Sampling data...')
        self.dataset_train = self.ev.sample_data(train_sample_size,valid_sample_size,test_sample_size,calib_sample_size,gen_sample_size)
        #store metadata in panda dataframe for saving
        self.df_meta['nruns'] = [nb_runs]; self.df_meta['train_epochs'] = [train_epochs]
        self.df_meta['batch_size'] = [batch_size]; self.df_meta['batch_size_MMD'] = [batch_size_MMD]
        self.df_meta['train_sample_size'] = [train_sample_size]; self.df_meta['valid_sample_size'] = [valid_sample_size]
        self.df_meta['test_sample_size'] = [test_sample_size]; self.df_meta['calib_sample_size'] = [calib_sample_size]
        self.df_meta['gen_sample_size'] = [gen_sample_size]
        #save parameters and remove old solutions file if it exists
        self.df_meta.to_csv(os.path.join(self.data_path_saving,'df_meta.csv'),index=False)
        try:
            os.remove(os.path.join(self.data_path_saving,'results.dat'))
        except:
            pass
        #calculate reference values for loss functions
        print('Calculating reference values...')
        self.ev.calibrate_distances(self.cr,self.cr_MMD,batch_size,batch_size_MMD)
        #loop over candidates
        print('\n[TRAINING]:')
        for cand_ind,candidate in enumerate(candidates):
            self.prof.set_candidate(cand_ind) #[PROFILING]
            self.prof.start_global() #[PROFILING]
            print('Candidate ' + str(cand_ind) + ':')
            #initialize next model
            adjacency_matrix = np.reshape(candidate,(nb_vars,nb_vars))
            self.current_model = CGNN_model(adjacency_matrix)
            self.ev.set_candidate(cand_ind,adjacency_matrix)
            #begin training of current model (several times if nruns > 1)
            for run_ind in range(nb_runs):
                self.prof.set_run(run_ind) #[PROFILING]
                self.ev.set_run(run_ind) #tell evaluator where to store current results
                print('[Cand-' + str(cand_ind) + ', Run-' + str(run_ind) + ']:')
                #train (and validate)!
                self._train_current_model(train_epochs,batch_size,batch_size_MMD,gen_sample_size) #including validation of model
                #test!
                print('[TESTING]:')
                self.ev.test_model(self.current_model)
                #save training results for current run
                self.ev.save_run_results()
            #save training results for current candidate
            self.ev.save_candidate_results()
            self.prof.stop_global()
            print('FINISHED training candidate ', str(cand_ind), '(Needed Time: ' + str(round((self.prof.get_current_tot_time()),2)) + ')\n\n')
        self.ev.save_eval_results()
        self.prof.plot_results() #[PROFILING]
        
    def _train_current_model(self,train_epochs,batch_size,batch_size_MMD,gen_sample_size):
        #initialize early stopping control object
        early_stopping = uFun.EarlyStopping(patience=self.patience)
        #reset model parameters
        self.current_model.reset_parameters()
        #setup optimizer
        optimizer = th.optim.Adam(self.current_model.parameters(),lr=self.lr)
        #initialize DataLoaders
        train_loader = DataLoader(self.dataset_train,batch_size=batch_size,shuffle=True,drop_last=True)
        train_loader_MMD = DataLoader(self.dataset_train,batch_size=batch_size_MMD,shuffle=True,drop_last=True)
        #loop over training epochs
        for epoch in tqdm(range(train_epochs)):
            #[EVAL] 
            if self.eval_training:
                uFun.plot_marginals(self.current_model.generate(gen_sample_size)) #,saving_path='./Temp/marg_' + str(epoch) + '.png'
            #training: loop over batches in train loader and train_loader_MMD
            lenReg = len(train_loader)
            lenMMD = len(train_loader_MMD)
            factorMMD = np.floor(lenMMD / lenReg).astype(int)
            iter_train_loader_MMD = iter(train_loader_MMD)
            #indMMD = 0
            for data in train_loader:
                #nb_cr = len(self.cr)
                #nb_cr_MMD = len(self.cr_MMD)
                #train_loss = th.zeros(nb_cr + nb_cr_MMD * factorMMD,1)
                #[non MMD]
                #calculate training loss - non MMD
                self.prof.start() #[PROFILING]
                gen_data = self.current_model.generate(batch_size)
                self.prof.stop_tr('gen_tr',epoch) #[PROFILING]
                self.prof.start() #[PROFILING]#
                if len(self.cr):
                    train_loss = self._apply_criterion(gen_data,data)
                else:
                    train_loss = th.zeros([1,0],requires_grad=True)
                self.prof.stop_tr('ev_tr',epoch) #[PROFILING]
                #calculate training loss - MMD
                for k in range(factorMMD):
                    self.prof.start() #[PROFILING]
                    gen_data = self.current_model.generate(batch_size_MMD)
                    self.prof.stop_tr('gen_tr',epoch) #[PROFILING]
                    self.prof.start() #[PROFILING]
                    train_loss_MMD = self._apply_criterion_MMD(gen_data,next(iter_train_loader_MMD)) / factorMMD
                    train_loss = th.cat((train_loss,train_loss_MMD),axis=1)
                    self.prof.stop_tr('ev_tr',epoch) #[PROFILING]
                #backpropagation
                train_loss = th.sum(train_loss)
                #train_loss = th.sum(train_loss**2)
                self.prof.start() #[PROFILING]
                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                if self.eval_training:
                    uFun.plot_grad_flow(self.current_model.named_parameters())
                optimizer.step()
                self.prof.stop_tr('bp',epoch) #[PROFILING]
            #validation
            loss = self.ev.validate_model(self.current_model,epoch)
            #check if early stop should occur
            if early_stopping(loss):
                print('EARLY-STOPPING')
                break
            
    def _apply_criterion(self, d1, d2):
        loss_list = th.zeros([1,0],requires_grad=True)
        for ind,c in enumerate(self.cr):
            loss_list = th.cat((loss_list,th.atleast_2d(c.forward(d1,d2,True))),dim=1)
        return loss_list * self.cr_wgs / th.sum(self.cr_wgs)
    
    def _apply_criterion_MMD(self, d1, d2):
        loss_list = th.zeros([1,0],requires_grad=True)
        for ind,c in enumerate(self.cr_MMD):
            loss_list = th.cat((loss_list,th.atleast_2d(c.forward(d1,d2,True))),dim=1)
        return loss_list * self.cr_wgs_MMD / th.sum(self.cr_wgs_MMD)
    
#Evaluator
##############################################################################   
class Evaluator():
    def __init__(self, data_paths, data_paths_calib, data_path_saving, show_cand_plot):
        #[CONFIG]
        self.cr = [dFun.MMD_cdt(bandwidth=[1]),
                   dFun.MMD_s_pro(kernels=['Cos'],bandwidths=[1,10],variances=[0.1,1,10]),
                   dFun.CorrD([1,1],3),
                   dFun.BinnedD('JS',4),
                   dFun.CorrD_N(),
                   dFun.EDF_Marg('AD'),
                   dFun.Binned_Marg('He',sampling_rate=4),
                   dFun.MargD(num_std_moments=4)]
        self.cr_val_inds = [0,1,2,3,4,5,6,7] #which if the test criteria should be used for validation (in each training epoch)
        self.display_calib_digits = 3
        #store settings
        self.show_cand_plot = show_cand_plot
        #determine criteria parameters
        self.nb_cr_val = len(self.cr_val_inds)
        self.nb_cr_test = len(self.cr)
        self.cr_names = [c.get_name(True) for c in self.cr]
        #load data
        print('Loading data...')
        self.full_data_train = pd.read_csv(data_paths[0])
        self.full_data_valid = pd.read_csv(data_paths[1])
        self.full_data_test = pd.read_csv(data_paths[2])
        self.full_data_calib_high = pd.read_csv(data_paths_calib[0])
        self.full_data_calib_no = pd.read_csv(data_paths_calib[1])
        #create directories for saving results if they do not exist
        self.data_path_saving = data_path_saving
        Path(os.path.join(data_path_saving,'progress')).mkdir(parents=True,exist_ok=True);
        Path(os.path.join(data_path_saving,'test')).mkdir(parents=True,exist_ok=True);
        Path(os.path.join(data_path_saving,'trained_models')).mkdir(parents=True,exist_ok=True); 
        Path(os.path.join(data_path_saving,'generated_data')).mkdir(parents=True,exist_ok=True); 
        Path(os.path.join(data_path_saving,'plot')).mkdir(parents=True,exist_ok=True)
        
    def preallocate_memory(self, batch_size, batch_size_MMD):
        self.batch_size = batch_size
        self.batch_size_MMD = batch_size_MMD
        for c in self.cr:
            if c.is_MMD():
                c.preallocate_memory(batch_size_MMD)
            else:
                c.preallocate_memory(batch_size)
        
    def set_containers(self, prof, nb_cand, nb_runs, train_epochs, test_epochs):
        self.prof = prof
        self.nb_runs = nb_runs
        self.test_epochs = test_epochs
        self.dist_val = np.zeros([nb_cand,nb_runs,train_epochs,self.nb_cr_val]) #distance values from validation
        self.dist_test = np.zeros([nb_cand,nb_runs,test_epochs,self.nb_cr_test]) #distance values from testing
        self.dist_test_std = np.zeros([nb_cand,nb_runs,self.nb_cr_test]) #std deviation of test distance values
        self.results = np.zeros([nb_cand,1 + self.nb_cr_test]) #best mean distance values for each candidate (+1 for index of best candidate)
        
    def sample_data(self, train_sample_size, valid_sample_size, test_sample_size, calib_sample_size, gen_sample_size):
        self.gen_sample_size = gen_sample_size
        self.dataset_train = th.Tensor(self.full_data_train.sample(n=train_sample_size).values)
        self.dataset_valid = th.Tensor(self.full_data_valid.sample(n=valid_sample_size).values)
        self.dataset_test = th.Tensor(self.full_data_test.sample(n=test_sample_size).values)
        self.dataset_calib_high = th.Tensor(self.full_data_calib_high.sample(n=calib_sample_size).values)
        self.dataset_calib_no = th.Tensor(self.full_data_calib_no.sample(n=calib_sample_size).values)
        return self.dataset_train
    
    def set_candidate(self, cand_ind, adjacency_matrix):
        self.cand_ind = cand_ind
        self.cand_model_list = [None for r in range(self.nb_runs)]
        self.current_adj_matrix = adjacency_matrix
        
    def set_run(self, run_ind):
        self.run_ind = run_ind
    
    def calibrate_distances(self, criteria_train, criteria_train_MMD, batch_size_train, batch_size_train_MMD):
        print('[Calculating reference-values]');
        print('training (non-MMD):')
        for ind,c in enumerate(criteria_train):
            print('- ' + c.get_name(True))
            self._calibrate_distance(c,batch_size_train)
        print('training (MMD):')
        for ind,c in enumerate(criteria_train_MMD):
            print('- ' + c.get_name(True))
            self._calibrate_distance(c,batch_size_train_MMD)
        #ref values for validation/test criteria
        print('validation/test:')
        for ind,c in enumerate(self.cr):
            print('- ' + c.get_name(True))
            self._calibrate_distance(c)
        return
    
    def validate_model(self, model, epoch):
        #initialize DataLoaders
        loader_valid = DataLoader(self.dataset_valid,batch_size=self.batch_size,shuffle=True,drop_last=True)
        loader_valid_MMD = DataLoader(self.dataset_valid,batch_size=self.batch_size_MMD,shuffle=True,drop_last=True)
        #loop over validation distances
        for ind in self.cr_val_inds:
            if self.cr[ind].is_MMD():
                val_mean,_ = self._average_model_dist_over_batches(self.cr[ind],model,self.batch_size_MMD,loader_valid_MMD,True,'va',epoch)
            else:
                val_mean,_ = self._average_model_dist_over_batches(self.cr[ind],model,self.batch_size,loader_valid,True,'va',epoch)
            self.dist_val[self.cand_ind,self.run_ind,epoch,ind] = val_mean
        return val_mean
            
    def test_model(self, model):
        #save model
        self.cand_model_list[self.run_ind] = model
        #prepare container to calculate standard deviations
        full_list = [[] for c in range(len(self.cr))]
        #loop over test epochs
        for epoch in range(self.test_epochs):
            #initialize DataLoaders
            loader_test = DataLoader(self.dataset_test,batch_size=self.batch_size,shuffle=True,drop_last=True)
            loader_test_MMD = DataLoader(self.dataset_test,batch_size=self.batch_size_MMD,shuffle=True,drop_last=True)
            #loop over test distances
            for ind,cr in enumerate(self.cr):
                if cr.is_MMD():
                    val_mean, val_list = self._average_model_dist_over_batches(cr,model,self.batch_size_MMD,loader_test_MMD,True,'te')
                else:
                    val_mean, val_list = self._average_model_dist_over_batches(cr,model,self.batch_size,loader_test,True,'te')
                self.dist_test[self.cand_ind,self.run_ind,epoch,ind] = val_mean
                full_list[ind] = full_list[ind] + val_list
        #calculate standard deviations
        for ind,vals in enumerate(full_list):
            self.dist_test_std[self.cand_ind,self.run_ind,ind] = np.std(vals)
        
    def save_run_results(self):
        #check if current run is best so far
        mean_test_loss = np.mean(self.dist_test[self.cand_ind,self.run_ind,:,:])
        if self.run_ind == 0:
            self.best_loss = mean_test_loss
            self.best_run_ind = self.run_ind
            print('   mean-test-loss: {:.02f} [CURRENT BEST]'.format(mean_test_loss.item()))
        elif mean_test_loss < self.best_loss:
            self.best_loss = mean_test_loss
            self.best_run_ind = self.run_ind
            print('   mean-test-loss: {:.02f} [CURRENT BEST]'.format(mean_test_loss.item()))
        else:
            print('   mean-test-loss: {:.02f} '.format(mean_test_loss.item()))
        #save run results
        self.prof.start() #[PROFILING]
        data_prog,data_test,data_test_std = self._store_run_to_dataFrames(self.cand_ind,self.run_ind)
        file_id = 'c{}_r{}'.format(str(self.cand_ind).zfill(4),str(self.run_ind).zfill(2))
        data_prog.to_csv(os.path.join(self.data_path_saving,'progress',file_id + '.csv'),index=False)
        data_test.to_csv(os.path.join(self.data_path_saving,'test',file_id + '.csv'),index=False)
        data_test_std.to_csv(os.path.join(self.data_path_saving,'test',file_id + '_std' + '.csv'),index=False)
        self.prof.stop('sav') #[PROFILING]
        
    def save_candidate_results(self):
        self.prof.start() #[PROFILING]
        best_model = self.cand_model_list[self.best_run_ind]
        prog_dists = self.dist_val[self.cand_ind,self.best_run_ind,:,:]
        data_prog,_,_ = self._store_run_to_dataFrames(self.cand_ind,self.best_run_ind)
        file_id = 'c{}_best'.format(str(self.cand_ind).zfill(4))
        results = np.mean(self.dist_test[self.cand_ind,self.best_run_ind,:,:],axis=0) #mean over test epochs
        self.results[self.cand_ind,:] = np.concatenate((np.reshape(self.best_run_ind,[1,1]),np.reshape(results,[1,len(results)])),axis=1)
        #model
        path = os.path.join(self.data_path_saving,'trained_models',file_id+'.pkl')
        uFun.object_saver(best_model,path)
        #data
        gen_data = best_model.generate(self.gen_sample_size)
        path = os.path.join(self.data_path_saving,'generated_data',file_id+'.dat')
        np.savetxt(path,gen_data.detach().numpy())
        #info-graphs
        path = os.path.join(self.data_path_saving,'plot',file_id) #no '.png'
        labels = [ self.cr_names[k] for k in self.cr_val_inds ]
        marg_bools = [ self.cr[k].is_Marg() for k in self.cr_val_inds ]
        uFun.dist_progress_dag_plotter(data=gen_data,prog_dists=prog_dists,prog_labels=labels,prog_marg_bools=marg_bools,adjacency_matrix=self.current_adj_matrix,saving_path=path,show_plot=self.show_cand_plot)
        #leftover:
        #self.current_model.load_state_dict(th.load(early_stopping.path)) #no idea what this does
        self.prof.stop('sav') #[PROFILING]
        
    def save_eval_results(self):
        self.prof.start() #[PROFILING]
        f = open(os.path.join(self.data_path_saving,'results.dat'),'a')
        np.savetxt(f,self.results,fmt='%f',newline=" ")
        f.write("\n")
        f.close()
        self.prof.stop('sav') #[PROFILING]
        
    def _calibrate_distance(self, criterion, custom_batch_size = None):
        #config
        nb_marg_batches = 10;
        #decide which batch_size to use
        if not custom_batch_size == None:
            bs = custom_batch_size
        elif criterion.is_MMD():
            bs = self.batch_size_MMD
        else:
            bs = self.batch_size
        if criterion.is_Marg():
            ref_val, ref_val_std, benchmark, benchmark_std = self._average_marg_dist_over_simData(criterion,bs,nb_marg_batches)
        else:
            #initialize DataLoaders
            loader_train = DataLoader(self.dataset_train,batch_size=bs,shuffle=True,drop_last=True)
            loader_calib_high = DataLoader(self.dataset_calib_high,batch_size=bs,shuffle=True,drop_last=True)
            loader_calib_no = DataLoader(self.dataset_calib_no,batch_size=bs,shuffle=True,drop_last=True)
            #calculate reference value
            ref_val, ref_val_std = self._average_samples_dist_over_batches(criterion,loader_train,loader_calib_high,False)
            #calculate benchmark
            benchmark, benchmark_std = self._average_samples_dist_over_batches(criterion,loader_train,loader_calib_no,False)
        #print results
        print('\trefVal: ' + str(np.round(ref_val.item(),decimals=self.display_calib_digits)) + ' (+-' + str(np.round(ref_val_std.item(),decimals=self.display_calib_digits)) + '), bchmrk: ' + str(np.round(benchmark.item(),decimals=self.display_calib_digits)) + ' (+-' + str(np.round(benchmark_std.item(),decimals=self.display_calib_digits)) + ')')
        print('\tSNR: ' + str(np.round((benchmark.item() - ref_val.item())/benchmark_std.item(),decimals=self.display_calib_digits)) + ', DIFF: ' + str(np.round((benchmark.item() - ref_val.item())/ref_val.item(),decimals=self.display_calib_digits)))
        criterion.save_calib(ref_val,benchmark)
        return
        
    def _average_samples_dist_over_batches(self, criterion, loader1, loader2, normalize):
        dist_list = th.zeros(len(loader1))
        for ind,(d1,d2) in enumerate(zip(loader1,loader2)):
            dist_list[ind] = criterion.forward(d1,d2,normalize)
        return th.mean(dist_list), th.std(dist_list)
    
    def _average_model_dist_over_batches(self, criterion, model, bs, loader, normalize, eval_str, epoch = None):
        dist_list = th.zeros(len(loader))
        for ind,data in enumerate(loader):
            self.prof.start() #[PROFILING]
            gen_data = model.generate(bs)
            self.prof.stop('gen_' + eval_str,epoch) #[PROFILING]
            self.prof.start() #[PROFILING]
            dist_list[ind] = criterion(gen_data,data,normalize)
            self.prof.stop('ev_' + eval_str,epoch) #[PROFILING]
        return th.mean(dist_list), dist_list.detach().tolist()
    
    def _average_marg_dist_over_simData(self, criterion, batch_size, nb_marg_batches):
        list_ref_vals_n = th.zeros(nb_marg_batches,1)
        list_ref_vals_u = th.zeros(nb_marg_batches,1)
        list_benchmarks_n = th.zeros(nb_marg_batches,1)
        list_benchmarks_u = th.zeros(nb_marg_batches,1)
        for k in range(nb_marg_batches):
            mdat_prop,mdat_prop2,mdat_comp_n,mdat_comp_u = uFun.create_marginal_test_data(batch_size)
            #ref-vals
            list_ref_vals_n[k] = criterion.forward(mdat_prop[0],mdat_prop2[0])
            list_ref_vals_u[k] = criterion.forward(mdat_prop[1],mdat_prop2[1])
            subList_BM_n = th.zeros(2*len(mdat_comp_n),1)
            for j in range(len(mdat_comp_n)):
                subList_BM_n[2*j] = criterion.forward(mdat_prop[0],mdat_comp_n[j])
                subList_BM_n[2*j+1] = criterion.forward(mdat_prop[1],mdat_comp_n[j])
            subList_BM_u = th.zeros(2*len(mdat_comp_u),1)
            for j in range(len(mdat_comp_u)):
                subList_BM_u[2*j] = criterion.forward(mdat_prop[0],mdat_comp_u[j])
                subList_BM_u[2*j+1] = criterion.forward(mdat_prop[1],mdat_comp_u[j])
            list_benchmarks_n[k] = th.mean(subList_BM_n)
            list_benchmarks_u[k] = th.mean(subList_BM_u)
        list_ref_vals = th.concat((list_ref_vals_n,list_ref_vals_u))
        list_benchmarks = th.concat((list_benchmarks_n,list_benchmarks_u))
        return th.mean(list_ref_vals), th.std(list_ref_vals), th.mean(list_benchmarks), th.std(list_benchmarks)
    
    def _store_run_to_dataFrames(self, cand_ind, run_ind):
        data_prog = pd.DataFrame()
        data_test = pd.DataFrame()
        data_test_std = pd.DataFrame()
        for ind,cr in enumerate(self.cr):
            data_prog[cr.get_name(True)] = self.dist_val[cand_ind,run_ind,:,ind]
        for ind in self.cr_val_inds:
            data_test[self.cr[ind].get_name(True)] = np.mean(self.dist_test[cand_ind,run_ind,:,ind])
            data_test_std[self.cr[ind].get_name(True)] = self.dist_test_std[cand_ind,run_ind,ind]
        return data_prog, data_test, data_test_std