import numpy as np
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import pandas as pd
import networkx as nx
import torch as th
import distance_functions as dFun
from copy import deepcopy
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, TensorDataset
import ast
import time
from tqdm import trange
from random import sample
from sklearn.model_selection import train_test_split


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        th.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


#####################################################

class CGNN_block(th.nn.Module):
    def __init__(self, structure):
        super().__init__()
        layers = []
        for i, j in zip(structure[:-2], structure[1:-1]):
            layers.append(th.nn.Linear(i, j))
            layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(structure[-2], structure[-1])) #linear connection between final layer and output (usually single number output)
        self.layers = th.nn.Sequential(*layers) #"*layers" allows to give a list of arguments as positional arguments
    def forward(self, x):
        return self.layers(x)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"): #no parameters to reset for the ReLU layers
                layer.reset_parameters()
                
class CGNN(th.nn.Module):
    def __init__(self, adj_matrix, nh):
        super().__init__()
        self.adjacency_matrix = adj_matrix
        self.nh=nh
        self.topological_order = list(nx.topological_sort(nx.DiGraph(self.adjacency_matrix)))
        self.num_var = len(self.topological_order)
        self.generated = [None for i in range(self.num_var)]
        self.register_buffer('noise', th.zeros(self.batch_size, self.num_var))
        self.register_buffer('score', th.FloatTensor([0]))
        #[initialize blocks]
        self.blocks = th.nn.ModuleList()
        for i in range(4): #only the non-hidden variables (first four) get blocks
            num_incoming = int(self.adjacency_matrix[:,i].sum()) + 1; #+1 to include the incoming local randomness
            block_structure = [num_incoming, self.nh, 1]
            self.blocks.append(CGNN_block(block_structure))

    def forward(self):
        self.noise.data.normal_()
        for i in self.topological_order:
            if i>=4:
                self.generated[i]=self.noise[:, [i]]
        for i in self.topological_order:
            if i<4:
                self.generated[i] = self.blocks[i](th.cat([v for c in [[self.generated[j] for j in np.nonzero(self.adjacency_matrix[:, i])[0]],[self.noise[:, [i]]]] for v in c],1))
        try:
            return th.cat(self.generated[0:4], 1)
        except:
            return th.cat(self.generated, 1)

    def run(self, data, train_epochs, test_epochs, lr, sample_size, training_eval_mode, index):
        optim = th.optim.Adam(self.parameters(), lr=lr); 
        self.score.zero_()
        early_stopping = EarlyStopping(patience=self.patience)
        data=data.sample(n= sample_size)
        train_val, test_data = train_test_split(data, test_size=0.20); 
        train_data, val_data = train_test_split(train_val, test_size=0.20);
        train_dataset, val_dataset, test_dataset = th.Tensor(train_data.values), th.Tensor(val_data.values), th.Tensor(test_data.values)
        train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True), DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True), DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)

        if training_eval_mode: #training_eval_mode
            lossEvList = np.zeros([train_epochs,3]);
            critOld = dFun.MMD_old();
            critTot = dFun.MMD_total();
            critCorr = dFun.CorrD();
            
        for epoch in range(train_epochs):
            lst_train, lst_val = [], []
            self.train()
            for data in train_loader:
                optim.zero_grad()
                mmd = self.criterion(self.forward()[:,0:4], data[:,0:4])
                mmd.backward()
                optim.step()
                lst_train.append(mmd.item())
            self.eval() 
            for data in val_loader:
                mmd=self.criterion(self.forward()[:,0:4], data[:,0:4])
                lst_val.append(mmd.item())
            train_loss = np.mean(lst_train); valid_loss = np.mean(lst_val); epoch_len = len(str(train_epochs))
            
            if training_eval_mode: #training_eval_mode
                #lossEvList[epoch,0] = mmd.item();
                lossEvList[epoch,0] = critOld.forward(self.forward()[:,0:4],data[:,0:4]);
                lossEvList[epoch,1] = critTot.forward(self.forward()[:,0:4],data[:,0:4]);
                lossEvList[epoch,2] = critCorr.forward(self.forward()[:,0:4],data[:,0:4]);
            
            if not epoch % 1:
                print_msg = (f'[{epoch:>{epoch_len}}/{train_epochs:>{epoch_len}}] ' + f'Train Loss: {train_loss:.4f} ' + f', Validation Loss: {valid_loss:.4f}')
                print(print_msg) 
                
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print('EarlyStopping')
                break 
        self.load_state_dict(th.load('checkpoint.pt'))
        
        if training_eval_mode: #training_eval_mode
            np.savetxt('./Debug/lossEvolution_' + self.criterionString + '_' + str(index) + '.dat',lossEvList);
            genData = self.forward()[:,0:4];
            genData = genData.detach().numpy();
            np.savetxt('./Debug/genData_' + self.criterionString + '_' + str(index) + '_FINAL.dat',genData);
            
        self.eval()
        lst_test=[self.criterion(self.forward()[:,0:4], data[:,0:4]).detach().numpy() for data in test_loader for epoch in range(test_epochs)]
        return np.mean(lst_test)
    
    def reset_parameters(self):
        for block in self.blocks:
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()
                
class CGNN_trainer:
    def __init__(self, nh, batch_size, train_epochs, test_epochs, sample_size, patience=10000, nruns=1, lr=0.01):
        self.nh = nh
        self.nruns = nruns
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.patience=patience
        self.index = -1;
        self.activate_training_eval_mode = True;
    
    def create_graph_from_data(self, data_path, given_candidates_path, saving_path, do_pruning=False, return_weights=False):
        data=pd.read_csv(data_path);
        nb_vars = len(list(data.columns))
        names = list(data.columns)
        cands=np.loadtxt(given_candidates_path)

        for idx, cand in enumerate(cands):
            self.index = idx;
            t_i=time.time()
            row=np.zeros((4+nb_vars**2,), dtype=float)
            candidate=np.reshape(cand,(nb_vars,nb_vars))
            scores = CGNN_trainer.parallel_graph_evaluation(self, data =data , adj_matrix= candidate, training_eval_mode=self.activate_training_eval_mode)
            output = np.ones(candidate.shape)
            prediction_array=candidate * output
            cand_o= prediction_array.reshape(1, nb_vars**2)
            row[0]=idx
            row[1]=scores
            row[2]=scores
            row[3:3+nb_vars**2]=cand_o
            f=open(saving_path,'a')
            np.savetxt(f, row, fmt='%f', newline=" ")
            f.write("\n")
            f.close()
            t_f=time.time()
            print('Candidate ID:', str(idx), 'Needed Time:', round((t_f-t_i),2))

    def parallel_graph_evaluation(self, data, adj_matrix, training_eval_mode=False):
        output=[]
        for run in range(self.nruns):
            model = CGNN_old(adj_matrix= adj_matrix, batch_size=self.batch_size, nh=self.nh, patience= self.patience)
            #model = CGNN(adj_matrix= adj_matrix, batch_size=self.batch_size, nh=self.nh, patience= self.patience)
            model.reset_parameters()
            scores = model.run(data=data, train_epochs=self.train_epochs, test_epochs=self.test_epochs, lr=self.lr, sample_size=self.sample_size ,training_eval_mode=training_eval_mode, index=self.index)
            output.append(scores)
        return np.mean(output)
                
class CGNN_processor:
    def __init__(self, nh, batch_size, train_epochs, test_epochs, sample_size, patience=10000, nruns=1, lr=0.01):
        self.nh = nh
        self.nruns = nruns
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.patience=patience
        self.index = -1;
        self.activate_training_eval_mode = True;
    
    def create_graph_from_data(self, data_path, given_candidates_path, saving_path, do_pruning=False, return_weights=False):
        data=pd.read_csv(data_path);
        nb_vars = len(list(data.columns))
        names = list(data.columns)
        cands=np.loadtxt(given_candidates_path)

        for idx, cand in enumerate(cands):
            self.index = idx;
            t_i=time.time()
            row=np.zeros((4+nb_vars**2,), dtype=float)
            candidate=np.reshape(cand,(nb_vars,nb_vars))
            scores = CGNN.parallel_graph_evaluation(self, data =data , adj_matrix= candidate, training_eval_mode=self.activate_training_eval_mode)
            output = np.ones(candidate.shape)
            prediction_array=candidate * output
            cand_o= prediction_array.reshape(1, nb_vars**2)
            row[0]=idx
            row[1]=scores
            row[2]=scores
            row[3:3+nb_vars**2]=cand_o
            f=open(saving_path,'a')
            np.savetxt(f, row, fmt='%f', newline=" ")
            f.write("\n")
            f.close()
            t_f=time.time()
            print('Candidate ID:', str(idx), 'Needed Time:', round((t_f-t_i),2))

    def parallel_graph_evaluation(self, data, adj_matrix, training_eval_mode=False):
        output=[]
        for run in range(self.nruns):
            model = CGNN(adj_matrix= adj_matrix, batch_size=self.batch_size, nh=self.nh, patience= self.patience)
            model.reset_parameters()
            scores = model.run(data=data, train_epochs=self.train_epochs, test_epochs=self.test_epochs, lr=self.lr, sample_size=self.sample_size ,training_eval_mode=training_eval_mode, index=self.index)
            output.append(scores)
        return np.mean(output)

class CGNN_old(th.nn.Module):
    def __init__(self, adj_matrix, batch_size, nh, patience):
        super().__init__()
        self.adjacency_matrix = adj_matrix
        self.batch_size = batch_size
        self.patience=patience
        self.nh=nh
        self.topological_order = list(nx.topological_sort(nx.DiGraph(self.adjacency_matrix)));
        self.num_var = len(self.topological_order);
        self.generated = [None for i in range(self.num_var)];
        #self.topological_order = [i for i in nx.topological_sort(nx.DiGraph(self.adjacency_matrix))]
        #self.generated = [None for i in range(self.adjacency_matrix.shape[0])]
        self.register_buffer('noise', th.zeros(self.batch_size, self.num_var))
        self.register_buffer('score', th.FloatTensor([0]))
        #[choose criterion]
        self.criterion = dFun.MMD_total()
        #self.criterion = dFun.MMD_old()
        #self.criterion = dFun.CorrD()
        #self.criterion.wgMarg = 1;
        #self.criterion.num_std_moments = 4;
        self.criterionString = 'tot';
        #[initialize blocks]
        self.blocks = th.nn.ModuleList()
        for i in range(4): #only the non-hidden variables (first four) get blocks
            num_incoming = int(self.adjacency_matrix[:,i].sum()) + 1; #+1 to include the incoming local randomness
            block_structure = [num_incoming, self.nh, 1]
            self.blocks.append(CGNN_block(block_structure))

    def forward(self):
        self.noise.data.normal_()
        for i in self.topological_order:
            if i>=4:
                self.generated[i]=self.noise[:, [i]]
        for i in self.topological_order:
            if i<4:
                f1 = [self.generated[j] for j in np.nonzero(self.adjacency_matrix[:, i])[0]];
                f2 = [self.noise[:, [i]]];
                e1 = [f1,f2];
                e2 = [v for c in e1 for v in c]
                e3 = th.cat(e2,1)
                self.generated[i] = self.blocks[i]( e3 )
        try:
            return th.cat(self.generated[0:4], 1)
        except:
            return th.cat(self.generated, 1)

    def run(self, data, train_epochs, test_epochs, lr, sample_size, training_eval_mode, index):
        optim = th.optim.Adam(self.parameters(), lr=lr); 
        self.score.zero_()
        early_stopping = EarlyStopping(patience=self.patience)
        data=data.sample(n= sample_size)
        train_val, test_data = train_test_split(data, test_size=0.20); 
        train_data, val_data = train_test_split(train_val, test_size=0.20);
        train_dataset, val_dataset, test_dataset = th.Tensor(train_data.values), th.Tensor(val_data.values), th.Tensor(test_data.values)
        train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True), DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True), DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True)

        if training_eval_mode: #training_eval_mode
            lossEvList = np.zeros([train_epochs,3]);
            critOld = dFun.MMD_old();
            critTot = dFun.MMD_total();
            critCorr = dFun.CorrD();
            
        for epoch in range(train_epochs):
            lst_train, lst_val = [], []
            self.train()
            for data in train_loader:
                optim.zero_grad()
                mmd = self.criterion(self.forward()[:,0:4], data[:,0:4])
                mmd.backward()
                optim.step()
                lst_train.append(mmd.item())
            self.eval() 
            for data in val_loader:
                mmd=self.criterion(self.forward()[:,0:4], data[:,0:4])
                lst_val.append(mmd.item())
            train_loss = np.mean(lst_train); valid_loss = np.mean(lst_val); epoch_len = len(str(train_epochs))
            
            if training_eval_mode: #training_eval_mode
                #lossEvList[epoch,0] = mmd.item();
                lossEvList[epoch,0] = critOld.forward(self.forward()[:,0:4],data[:,0:4]);
                lossEvList[epoch,1] = critTot.forward(self.forward()[:,0:4],data[:,0:4]);
                lossEvList[epoch,2] = critCorr.forward(self.forward()[:,0:4],data[:,0:4]);
            
            if not epoch % 1:
                print_msg = (f'[{epoch:>{epoch_len}}/{train_epochs:>{epoch_len}}] ' + f'Train Loss: {train_loss:.4f} ' + f', Validation Loss: {valid_loss:.4f}')
                print(print_msg) 
                
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print('EarlyStopping')
                break 
        self.load_state_dict(th.load('checkpoint.pt'))
        
        if training_eval_mode: #training_eval_mode
            np.savetxt('./Debug/lossEvolution_' + self.criterionString + '_' + str(index) + '.dat',lossEvList);
            genData = self.forward()[:,0:4];
            genData = genData.detach().numpy();
            np.savetxt('./Debug/genData_' + self.criterionString + '_' + str(index) + '_FINAL.dat',genData);
            
        self.eval()
        lst_test=[self.criterion(self.forward()[:,0:4], data[:,0:4]).detach().numpy() for data in test_loader for epoch in range(test_epochs)]
        return np.mean(lst_test)
    
    def reset_parameters(self):
        for block in self.blocks:
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()