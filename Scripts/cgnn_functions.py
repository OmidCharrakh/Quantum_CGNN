import utilities_functions as uFun
from profiler import Profiler
from evaluator import Evaluator
import numpy as np
import networkx as nx
import torch as th; from torch.utils.data import DataLoader
import os; from pathlib import Path
import collections


class CGNN_block(th.nn.Module):
    def __init__(self, hidden_layer_sizes, nb_incomings_c, nb_incomings_q=0, node_type='c', nb_ns=2, dim_q=3):
        super().__init__()
        if node_type=='c':
            dim_in = (1+nb_incomings_c) + dim_q*(0+nb_incomings_q)
            dim_out = 1
        elif node_type=='q':
            dim_in = (0+nb_incomings_c) + dim_q*(1+nb_incomings_q)
            dim_out = dim_q
        elif node_type=='ns':
            dim_in = (1+nb_incomings_c) + dim_q*(0+nb_incomings_q)
            dim_out = nb_ns
        dim_first = hidden_layer_sizes[0]
        dim_last  = hidden_layer_sizes[-1]
        nb_layers = len(hidden_layer_sizes)
        layers    = th.nn.ModuleList()
        layers.append(th.nn.Linear(dim_in, dim_first)) #transformation from incoming connections and noise (+1) to first layer
        layers.append(th.nn.ReLU()) #ReLU on each hidden unit (of first layer)
        for i in range(1, nb_layers):
            layers.append(th.nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i])) #transformation from layer "layer_ind-1" to layer "layer_ind"
            layers.append(th.nn.ReLU()) #ReLU on each hidden unit (of layer "layer_ind")
        layers.append(th.nn.Linear(dim_last, dim_out)) #transformation from hidden units (of last layer) to output
        self.layers = th.nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
                
##############################################################################
class CGNN_model(th.nn.Module):
    def __init__(self, adjacency_matrix, nb_base_hu, nb_layers, wg_conn, nodes_type=None, dim_q=3): 
        super().__init__()
        self.dim_q = dim_q
        self.adjacency_matrix = adjacency_matrix
        self.nb_vars = adjacency_matrix.shape[0]
        self.nodes_type = ['c']*self.nb_vars if nodes_type is None else nodes_type
        self.topological_order = list(nx.topological_sort(nx.DiGraph(adjacency_matrix)))
        self.nb_tot_conn = np.sum(adjacency_matrix) - np.sum(adjacency_matrix[np.diag_indices(self.nb_vars)]) 
        self.legend = ['{}'.format(n) for n in range(self.nb_vars)] 
        self.parents_ids = {n: [i for i in np.where(adjacency_matrix[:, n]!=0)[0]] for n in range(self.nb_vars)}
        hidden_layer_sizes_all = uFun.count_hu(adjacency_matrix, nb_base_hu, wg_conn, nb_layers)
        self.ns_ids = [k for k in range(self.nb_vars) if self.nodes_type[k]=='ns']
        self.blocks_list = th.nn.ModuleList([None for _ in range(self.nb_vars)])
        for n in range(self.nb_vars):
            incoming_ids = np.nonzero(adjacency_matrix[:, n])[0]
            incoming_types = [self.nodes_type[i] for i in incoming_ids]
            counter = collections.Counter(incoming_types)
            block = CGNN_block(
                hidden_layer_sizes=hidden_layer_sizes_all[n], 
                nb_incomings_c=counter['c'], 
                nb_incomings_q=counter['q'], 
                node_type=self.nodes_type[n],
                dim_q = self.dim_q,
            )
            self.blocks_list[n] = block
        self._print_hu(hidden_layer_sizes_all)

    def generate_noise(self, gen_size):
        noise_dict = {}
        for n in range(self.nb_vars):
            if n in [0, 1]:
                noise_dict[n] = th.zeros(gen_size,1).normal_(0,1) #for the outcome variables
            elif n in [2, 3]:
                noise_dict[n] = th.zeros(gen_size,1).uniform_(0,1) #for the setting variables
            elif self.nodes_type[n]=='c':
                noise_dict[n] = th.zeros(gen_size,1).normal_(0,1) #for classical latent variables
            elif self.nodes_type[n]=='q':
                noise_dict[n] = th.zeros(gen_size, self.dim_q).normal_(0,1) #for quantum latent variables
        return noise_dict

    def forward(self, gen_size, only_observables=True):
        local_noises = self.generate_noise(gen_size)
        gen_data     = {n: None for n in range(self.nb_vars)}
        for n in self.topological_order:
            local_noise = [local_noises[n]]
            parent_data = [gen_data[p] for p in self.parents_ids[n]]
            input_data  = th.cat([v for c in [parent_data, local_noise] for v in c], axis=1)
            gen_data[n] = self.blocks_list[n](input_data) 
        ns_counter = 0
        data = []
        for k in range(4 if only_observables else self.nb_vars):
            if k not in self.ns_ids:
                data.append(gen_data[k])
            elif (k in self.ns_ids) and (ns_counter == 0):
                data.append(gen_data[k])
                ns_counter += 1
        return th.cat(data, axis=1)
    
    def _print_hu(self, hidden_layer_sizes_all):
        total_hu = 0
        print('Model constructed!')
        for k in range(self.nb_vars):
            incStr = '';
            for j in range(self.nb_vars):
                if self.adjacency_matrix[j,k]:
                    if len(incStr):
                        incStr += ','
                    incStr += self.legend[j]
            if not len(incStr):
                incStr = 'X'
            incStr += ' --' + str(hidden_layer_sizes_all[k]) + '--> ' + self.legend[k] + '   (' + str(np.sum(hidden_layer_sizes_all[k])) + ')'
            total_hu += np.sum(hidden_layer_sizes_all[k])
            print(incStr)
        print('Total HU: (' + str(total_hu) + ')\n')
    def reset_parameters(self):
        for block in self.blocks_list:
            block.reset_parameters()
        print('Model paremeters reset.')

##############################################################################

class CGNN:
    def __init__(self, arc_nb_base_hu, arc_nb_layers, arc_wg_conn, lr, patience, loss_sum_pow, criteria_train, criteria_train_wgs, data_paths, data_paths_calib, results_path, mode_sgl_backprop, show_cand_plot, evaluate_training, nb_cpus=8):
        super().__init__()
        #store parameters
        th.set_num_threads(nb_cpus)
        print('The process will use {} CPUs!'.format(th.get_num_threads()))
        #   architecture
        self.arc_nb_base_hu = arc_nb_base_hu
        self.arc_nb_layers = arc_nb_layers
        self.arc_wg_conn = arc_wg_conn
        #   early stopping
        self.lr = lr
        self.patience = patience
        self.loss_sum_pow = loss_sum_pow
        #   misc
        self.mode_sgl_backprop = mode_sgl_backprop
        self.results_path = results_path
        self.eval_training = evaluate_training
        #sort criteria into regular and mmd
        self.cr_REG = []
        self.cr_MMD = []
        self.cr_REG_wgs = []
        self.cr_MMD_wgs = []
        if len(criteria_train_wgs) == 0:
            #generate weights from SNR
            self.weightsFromSNR = True
            for c in criteria_train:
                if c.is_MMD():
                    self.cr_MMD.append(c)
                else:
                    self.cr_REG.append(c)
        else:
            #use user given weights
            self.weightsFromSNR = False
            for c, wg in zip(criteria_train, criteria_train_wgs):
                if c.is_MMD():
                    self.cr_MMD.append(c)
                    self.cr_MMD_wgs.append(wg)
                else:
                    self.cr_REG.append(c)
                    self.cr_REG_wgs.append(wg)
            self.cr_REG_wgs = th.tensor(self.cr_REG_wgs)
            self.cr_MMD_wgs = th.tensor(self.cr_MMD_wgs)
        #initialize evaluator (evaluator loads all data)
        self.ev = Evaluator(data_paths, data_paths_calib, results_path, show_cand_plot)

        #create directories for saving results if they do not exist
        for folder in ['synthetic', 'profile', 'losses/train', 'losses/valid', 'losses/test']:
            saving_path = os.path.join(results_path, folder)
            Path(saving_path).mkdir(parents=True, exist_ok=True)

    def evaluate_candidates(self, nb_runs, train_epochs, test_epochs, batch_size_REG, batch_size_MMD, train_sample_size, valid_sample_size, test_sample_size, calib_sample_size, gen_sample_size, candidates_path, eval_name='test', candidates_ids=None, nodes_type=None, dim_q=3): 
        self.dim_q = dim_q
        #preallocate memory for criteria
        for cr in self.cr_REG:
            cr.preallocate_memory(batch_size_REG)
        for cr in self.cr_MMD:
            cr.preallocate_memory(batch_size_MMD)
        self.ev.set_current_name(eval_name)
        self.ev.preallocate_memory(batch_size_REG, batch_size_MMD)
        #load candidates file, setup profiler and pass parameters to evaluator
        candidates = np.loadtxt(candidates_path)
        if candidates.ndim == 1:
            #convert to 2Dim array if only one candidate present
            candidates = candidates[None,...]
        if candidates_ids is None:
            candidates_ids = list(range(len(candidates)))
        candidates = candidates[candidates_ids, :]
        nb_cands = len(candidates)

        self.prof = Profiler(nb_cands, nb_runs, train_epochs, self.results_path + '/profile/') #[PROFILING]
        #prepare evaluator and sample data
        self.ev.set_containers(self.prof, nb_cands, nb_runs, train_epochs, test_epochs)
        print('Sampling data...')
        self.dataset_train = self.ev.sample_data(train_sample_size, valid_sample_size, test_sample_size, calib_sample_size, gen_sample_size)
        #calculate reference values for loss functions
        print('Calculating reference values...')
        if self.weightsFromSNR:
            self.cr_REG_wgs, self.cr_MMD_wgs = self.ev.calibrate_distances(self.cr_REG, self.cr_MMD, batch_size_REG, batch_size_MMD)
        else:
            self.ev.calibrate_distances(self.cr_REG, self.cr_MMD, batch_size_REG, batch_size_MMD)

        #loop over candidates
        print('\n[TRAINING]:')
        for cand_ind, candidate in enumerate(candidates):
            cand_name = candidates_ids[cand_ind]
            self.prof.set_candidate(cand_ind) #[PROFILING]
            self.prof.start_global() #[PROFILING]
            print('Started training candidate with cand_ind={} and cand_name={}'.format(cand_ind, cand_name))
            #initialize next model
            nb_vars = np.sqrt(len(candidate)).astype(int)
            adjacency_matrix = np.reshape(candidate, (nb_vars, nb_vars)).astype(int)
            adjacency_matrix = uFun.prune_adjacency(adjacency_matrix)
            self.current_model = CGNN_model(adjacency_matrix, self.arc_nb_base_hu, self.arc_nb_layers, self.arc_wg_conn, nodes_type, self.dim_q)
            self.ev.set_candidate(cand_ind, adjacency_matrix)
            #begin training of current model (several times if nruns > 1)
            for run_ind in range(nb_runs):
                self.prof.set_run(run_ind) #[PROFILING]
                self.ev.set_run(run_ind) #tell evaluator where to store current results
                print('[Cand-' + str(cand_ind) + ', Run-' + str(run_ind) + ']:')
                #train (and validate)!
                self._train_current_model(train_epochs, batch_size_REG, batch_size_MMD, gen_sample_size)
                #test!
                self.ev.test_model(self.current_model)
                #set the best run of the current cand 
                self.ev.set_best_run()
                #save the loss values (train & valid & test) for the present cand-run 
                self.ev.save_losses(cand_name)
            #save synethic data from generated by the best run for the present cand-run 
            self.ev.save_synthetic(cand_name)
            self.prof.stop_global()
            print('Fnished training candidate with cand_ind={} and cand_name={}'.format(cand_ind, cand_name))
            print('(Needed Time: = {}\n\n'.format(round((self.prof.get_current_tot_time()), 2)))
        self.prof.plot_results() #[PROFILING]
        
    def _train_current_model(self, train_epochs, batch_size_REG, batch_size_MMD, gen_sample_size):
        #initialize early stopping control object
        early_stopping = uFun.EarlyStopping(patience=self.patience)
        #nan_stopping = uFun.NanStopping(patience=20)
        #reset model parameters
        self.current_model.reset_parameters()
        #setup optimizer
        optimizer = th.optim.Adam(self.current_model.parameters(), lr=self.lr)
        #initialize DataLoaders
        train_loader_REG = DataLoader(self.dataset_train, batch_size=batch_size_REG, shuffle=True, drop_last=True)
        train_loader_MMD = DataLoader(self.dataset_train, batch_size=batch_size_MMD, shuffle=True, drop_last=True)
        #loop over training epochs
        train_loss_epoch_list = []
        for epoch in range(train_epochs):
            if self.eval_training:
                uFun.plot_marginals(self.current_model(gen_sample_size)) 
            #training: loop over batches in train loader and train_loader_MMD
            lenREG = len(train_loader_REG)
            lenMMD = len(train_loader_MMD)
            factorMMD = np.floor(lenMMD/lenREG).astype(int)
            iter_train_loader_MMD = iter(train_loader_MMD)
            train_loss = th.zeros([1,0],requires_grad=True)
            train_loss_batch_list = []
            for data in train_loader_REG:
                #[non MMD]
                #calculate training loss - non MMD
                self.prof.start() #[PROFILING]
                gen_data = self.current_model(batch_size_REG)
                self.prof.stop_tr('gen_tr',epoch) #[PROFILING]
                self.prof.start() #[PROFILING]#
                train_loss = th.cat((train_loss, self._apply_criterion_REG(gen_data,data)),axis=1)
                self.prof.stop_tr('ev_tr',epoch) #[PROFILING]
                #calculate training loss - MMD
                for _ in range(factorMMD):
                    self.prof.start() #[PROFILING]
                    gen_data = self.current_model(batch_size_MMD)
                    self.prof.stop_tr('gen_tr',epoch) #[PROFILING]
                    self.prof.start() #[PROFILING]
                    train_loss_MMD = self._apply_criterion_MMD(gen_data, next(iter_train_loader_MMD))/factorMMD
                    train_loss = th.cat((train_loss, train_loss_MMD), axis=1)
                    self.prof.stop_tr('ev_tr',epoch) #[PROFILING]
                train_loss = th.sum(train_loss**self.loss_sum_pow)
                train_loss_batch_list.append(train_loss.item())
                self.prof.start() #[PROFILING]
                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                if self.eval_training:
                    uFun.plot_grad_flow(self.current_model.named_parameters())
                optimizer.step()
                self.prof.stop_tr('bp',epoch) #[PROFILING]
                train_loss = th.zeros([1,0], requires_grad=True)
            #train_loss
            lossTr = np.mean(train_loss_batch_list)
            train_loss_epoch_list.append(lossTr)
            #valid_loss
            valid_loss = self.ev.validate_model(self.current_model,epoch)
            lossVa = valid_loss.item()

            if epoch%10==0:
                print('Epoch {}: train_loss: {:.02f}, valid_loss: {:.02f}'.format(epoch, lossTr, lossVa))
            if early_stopping(valid_loss):
                print('EARLY-STOPPING')
                break
        self.ev.dist_train = np.array(train_loss_epoch_list)

    def _apply_criterion_REG(self, d1, d2):
        loss_list = th.zeros([1,0], requires_grad=True)
        for cr in self.cr_REG:
            loss = cr.forward(d1, d2, True)
            loss_list = th.cat((loss_list, th.atleast_2d(loss)), dim=1).abs()
        return loss_list * self.cr_REG_wgs / th.sum(th.concat((self.cr_REG_wgs, self.cr_MMD_wgs)))
    
    def _apply_criterion_MMD(self, d1, d2):
        loss_list = th.zeros([1,0], requires_grad=True)
        for cr in self.cr_MMD:
            loss = cr.forward(d1, d2, True)
            loss_list = th.cat((loss_list, th.atleast_2d(loss)), dim=1).abs()
        return loss_list * self.cr_MMD_wgs / th.sum(th.concat((self.cr_REG_wgs, self.cr_MMD_wgs)))
