import numpy as np
import pandas as pd
import networkx as nx
import torch as th
from copy import deepcopy
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, TensorDataset
import ast
import time
from tqdm import trange
from random import sample
from sklearn.model_selection import train_test_split
#####################################################

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
#####################################################

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

class GNN_model(th.nn.Module):
    def __init__(self, batch_size, nh=20, lr=0.01, train_epochs=100, test_epochs=100, dataloader_workers=0, **kwargs):
        super(GNN_model, self).__init__()
        self.l1 = th.nn.Linear(2, nh)
        self.l2 = th.nn.Linear(nh, 1)
        self.register_buffer('noise', th.Tensor(batch_size, 1))
        self.act = th.nn.ReLU()
        self.layers = th.nn.Sequential(self.l1, self.act, self.l2)
        self.batch_size = batch_size
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.dataloader_workers = dataloader_workers
    def forward(self, x):
        self.noise.normal_()
        return self.layers(th.cat([x, self.noise], 1))
    def run(self, dataset):
        optim = th.optim.Adam(self.parameters(), lr=self.lr)
        pbar = trange(self.train_epochs + self.test_epochs, disable=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.dataloader_workers)
        test_loss = 0
        for epoch in pbar:
            for i, (x, y) in enumerate(dataloader):
                optim.zero_grad()
                pred = self.forward(x)
                mmd = MMD(th.cat([x, pred], 1), th.cat([x, y], 1))
                if epoch < self.train_epochs:
                    mmd.backward()
                    optim.step()
                else:
                    test_loss = test_loss + mmd.data
        return test_loss.numpy()/self.test_epochs

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

def GNN_instance(data, batch_size=-1, device=None, nh=20, **kwargs):
    if batch_size == -1:
        batch_size = data.__len__()
    device = 'cpu'
    GNNXY = GNN_model(batch_size, nh=nh, **kwargs).to(device)
    GNNYX = GNN_model(batch_size, nh=nh, **kwargs).to(device)
    GNNXY.reset_parameters()
    GNNYX.reset_parameters()
    XY = GNNXY.run(TensorDataset(data[0].to(device), data[1].to(device)))
    YX = GNNYX.run(TensorDataset(data[1].to(device), data[0].to(device)))
    return [XY, YX]

class GNN:
    def __init__(self, nh=20, lr=0.01, nruns=6, batch_size=-1, train_epochs=100, test_epochs=100, dataloader_workers=0):
        super(GNN, self).__init__()
        self.nh = nh
        self.lr = lr
        self.nruns = nruns
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.dataloader_workers = dataloader_workers
    def predict_proba(self, dataset):
        data = [th.Tensor(scale(th.Tensor(i).view(-1, 1))) for i in dataset]
        AB = []; BA = []
        result_pair = [GNN_instance(data, device='cpu', train_epochs=self.train_epochs, test_epochs=self.test_epochs, batch_size=self.batch_size, dataloader_workers=self.dataloader_workers) for run in range(self.nruns)]
        AB.extend([runpair[0] for runpair in result_pair]) 
        BA.extend([runpair[1] for runpair in result_pair])
        loss_AB = np.mean(AB); loss_BA = np.mean(BA)
        return (loss_AB/ (loss_BA + loss_AB))

def tunner_generator(n_points=100, 
                     n_pairs=100,
                     data_path='./Data/data_main.csv', 
                     saving_path='./Data/gnn_data.csv'):
    df = pd.read_csv(data_path)
    gnn_data=pd.DataFrame(columns=['A', 'B'])
    for pair in range(n_pairs):
        rand_id=sample(set(df.index), n_points)
        A1=[df.iloc[i,2] for i in rand_id]; B1=[df.iloc[i,0] for i in rand_id];
        A2=[df.iloc[i,3] for i in rand_id]; B2=[df.iloc[i,1] for i in rand_id];
        gnn_data = gnn_data.append({'A': A1, 'B': B1}, ignore_index=True) 
        gnn_data = gnn_data.append({'A': A2, 'B': B2}, ignore_index=True)
    gnn_data.to_csv(saving_path, index=False)
    
def nh_tunner(nh_range=range(1, 100,1), 
              data_path='./Data/gnn_data.csv', 
              saving_path='./Data/hyper.txt'):
    gnn_data=pd.read_csv(data_path, converters={"A": ast.literal_eval, "B": ast.literal_eval})
    for idx, nh in enumerate(nh_range):
        t_i=time.time()
        obj = GNN(nh=nh, lr=0.01, nruns=15, train_epochs=20, test_epochs=20, batch_size=len(gnn_data.iloc[0,0]))
        loss_AB_1=np.array([obj.predict_proba((gnn_data.iloc[i])) for i in range(0, len(gnn_data),2)])
        loss_AB_2=np.array([obj.predict_proba((gnn_data.iloc[i])) for i in range(1, len(gnn_data),2)])
        loss_AB=loss_AB_1+loss_AB_2
        sum_loss= np.sum(loss_AB)
        pred_1=np.array([1 if l<=.5 else 0 for l in loss_AB_1])
        pred_2=np.array([1 if l<=.5 else 0 for l in loss_AB_2])
        pred=pred_1*pred_2
        wrong_predictions=len(pred)-np.sum(pred)
        total_predictions= len(pred)
        accuracy=1-1*wrong_predictions/total_predictions
        row=np.zeros((6,), dtype=float)
        row[0]=idx
        row[1]=nh
        row[2]=accuracy
        row[3]=sum_loss
        row[4]=wrong_predictions
        row[5]=total_predictions
        f=open(saving_path,'a')
        np.savetxt(f, row, fmt='%f', newline=" ")
        f.write("\n")
        f.close()
        t_f=time.time()
        print('Completed Round:', str(idx+1), 'Needed Time:', round((t_f-t_i),2))
#####################################################

class CGNN_block(th.nn.Module):
    def __init__(self, sizes):
        super(CGNN_block, self).__init__()
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(th.nn.Linear(i, j))
            layers.append(th.nn.ReLU())
        layers.append(th.nn.Linear(sizes[-2], sizes[-1]))
        self.layers = th.nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
                
class CGNN_model(th.nn.Module):
    def __init__(self, adj_matrix, batch_size, nh, patience):
        super(CGNN_model, self).__init__()
        self.adjacency_matrix = adj_matrix
        self.batch_size = batch_size
        self.patience=patience
        self.nh=nh
        self.topological_order = [i for i in nx.topological_sort(nx.DiGraph(self.adjacency_matrix))]
        self.generated = [None for i in range(self.adjacency_matrix.shape[0])]
        self.register_buffer('noise', th.zeros(self.batch_size, self.adjacency_matrix.shape[0]))
        self.register_buffer('score', th.FloatTensor([0]))
        self.blocks = th.nn.ModuleList()
        for i in range(self.adjacency_matrix.shape[0]):
            sizes=[int(self.adjacency_matrix[:, i].sum()) + 1, self.nh, 1]
            self.blocks.append(CGNN_block(sizes))

    def forward(self):
        self.noise.data.normal_()
        for i in self.topological_order:
            self.generated[i] = self.blocks[i](th.cat([v for c in [ [self.generated[j] for j in np.nonzero(self.adjacency_matrix[:, i])[0]],[self.noise[:, [i]]]] for v in c],1))
        return th.cat(self.generated, 1)


    def run(self, data, train_epochs, test_epochs, lr, sample_size):
        optim = th.optim.Adam(self.parameters(), lr=lr); self.score.zero_()
        early_stopping = EarlyStopping(patience=self.patience)
        data=data.sample(n= sample_size)
        train_val, test_data = train_test_split(data, test_size=0.20); 
        test_dataset = th.Tensor(scale(test_data.values));
        test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, drop_last=True);

        train_data, val_data = train_test_split(train_val, test_size=0.20); 
        train_dataset, val_dataset = th.Tensor(scale(train_data.values)), th.Tensor(scale(val_data.values)) 
        train_loader, val_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True), DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True)

        for epoch in range(train_epochs):
            lst_train, lst_val = [], []
            self.train()
            for data in train_loader:
                optim.zero_grad()
                mmd = MMD(self.forward()[:,0:4], data[:,0:4])
                mmd.backward()
                optim.step()
                lst_train.append(mmd.item())
            self.eval() 
            for data in val_loader:
                mmd = MMD(self.forward()[:,0:4], data[:,0:4])
                lst_val.append(mmd.item())
            train_loss = np.mean(lst_train); valid_loss = np.mean(lst_val); epoch_len = len(str(train_epochs))
            if not epoch % 10:
                print_msg = (f'[{epoch:>{epoch_len}}/{train_epochs:>{epoch_len}}] ' + f'Train Loss: {train_loss:.4f} ' + f', Validation Loss: {valid_loss:.4f}')
                print(print_msg) 
            
            early_stopping(valid_loss, self)
            if early_stopping.early_stop:
                print('EarlyStopping')
                break 
        self.load_state_dict(th.load('checkpoint.pt'))
        
        loss_abs, loss_rel=[], []
        # eCnt = 0; #DEBUG
        for epoch in range(test_epochs):
            self.eval()
            # dCnt = 0; #DEBUG
            for data in test_loader:
                # if (eCnt == test_epochs-1 or eCnt == 0) and dCnt == 0: #DEBUG
                #     simData = data[:,0:4];
                #     simData = simData.numpy();
                #     genData = self.forward()[:,0:4];
                #     genData = genData.detach().numpy();                    
                #     if eCnt == 0:
                #         np.savetxt('./Debug/X-sim-0.dat',simData);
                #         np.savetxt('./Debug/X-gen-0.dat',genData);
                #     else:
                #         np.savetxt('./Debug/X-sim-1.dat',simData);
                #         np.savetxt('./Debug/X-gen-1.dat',genData);
                mmd_abs  = MMD(data[:,0:4], self.forward()[:,0:4]).item()
                mmd_base = np.mean([MMD(data[:,0:4], _data[:,0:4]).item() for indx, _data in enumerate(test_loader) if indx<30])
                #mmd_base=1
                mmd_rel  = mmd_abs/mmd_base
                loss_abs.append(mmd_abs)
                loss_rel.append(mmd_rel)
                # dCnt += 1; #DEBUG
            # eCnt += 1; #DEBUG
        return (np.mean(loss_abs), np.mean(loss_rel))


    def reset_parameters(self):
        for block in self.blocks:
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()
class CGNN:
    def __init__(self, nh, batch_size, train_epochs, test_epochs, sample_size, patience=10, nruns=1, lr=0.01):
        super(CGNN, self).__init__()
        self.nh = nh
        self.nruns = nruns
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.patience=patience
    
    def create_graph_from_data(self, data, given_candidates_path, saving_path, do_pruning=False, return_weights=False):
        nb_vars = len(list(data.columns))
        names = list(data.columns)
        cands=np.loadtxt(given_candidates_path)

        for idx, cand in enumerate(cands):
            t_i=time.time()
            if do_pruning:
                row=np.zeros((5+2*nb_vars**2,), dtype=float)
            else:
                row=np.zeros((4+nb_vars**2,), dtype=float)
            candidate=np.reshape(cand,(nb_vars,nb_vars))
            scores = CGNN.parallel_graph_evaluation(self, data =data , adj_matrix= candidate)
            score_abs, score_rel = scores[0], scores[1]
            if return_weights:
                output = np.zeros(candidate.shape)
                for (i, j), x in np.ndenumerate(candidate):
                    if x > 0:
                        cand_copy = np.copy(candidate) 
                        cand_copy[i, j] = 0
                        output[i, j]= score_rel - CGNN.parallel_graph_evaluation(self, data=data, adj_matrix=cand_copy)[1]
            else:
                output = np.ones(candidate.shape)
    
            prediction_array=candidate * output
            cand_o= prediction_array.reshape(1, nb_vars**2)
            if do_pruning:
                prediction_graph = nx.DiGraph(prediction_array)
                nx.relabel_nodes(prediction_graph, {idx: i for idx, i in enumerate(names)}, copy=False)
                pruned_graph, score_p = CGNN.hill_climbing(self, data=data, graph=prediction_graph)
                pruned_array = nx.to_numpy_array(pruned_graph)
                cand_p= pruned_array.reshape(1, nb_vars**2)
                row[0]=idx
                row[1]=score_abs
                row[2]=score_rel
                row[3]=score_p
                row[4:4+nb_vars**2]=cand_o
                row[4+nb_vars**2:4+2**nb_vars**2]=cand_p
            else:
                row[0]=idx
                row[1]=score_abs
                row[2]=score_rel
                row[3:3+nb_vars**2]=cand_o
            f=open(saving_path,'a')
            np.savetxt(f, row, fmt='%f', newline=" ")
            f.write("\n")
            f.close()
            t_f=time.time()
            print('Candidate ID:', str(idx), 'Needed Time:', round((t_f-t_i),2))

    def parallel_graph_evaluation(self, data, adj_matrix):
        output_abs = []; output_rel = []
        for run in range(self.nruns):
            model = CGNN_model(adj_matrix= adj_matrix, batch_size=self.batch_size, nh=self.nh, patience= self.patience)
            model.reset_parameters()
            scores = model.run(data=data, train_epochs=self.train_epochs, test_epochs=self.test_epochs, lr=self.lr, sample_size=self.sample_size)
            output_abs.append(scores[0]); output_rel.append(scores[1]); 
        return (np.mean(output_abs), np.mean(output_rel))

    def hill_climbing(self, data, graph):
        nodelist = list(data.columns)
        tested_candidates = [nx.adjacency_matrix(graph, nodelist=nodelist, weight=None)]
        best_score = CGNN.parallel_graph_evaluation(self, data=data, adj_matrix=tested_candidates[0].todense())[1]
        best_candidate = graph
        can_improve = True
        while can_improve:
            can_improve = False
            for (i, j) in best_candidate.edges():
                test_graph = deepcopy(best_candidate)
                test_graph.add_edge(j, i, weight=test_graph[i][j]['weight'])
                test_graph.remove_edge(i, j)
                tadjmat = nx.adjacency_matrix(test_graph, nodelist=nodelist, weight=None)
                if (nx.is_directed_acyclic_graph(test_graph) and not any([(tadjmat != cand).nnz == 0 for cand in tested_candidates])):
                    tested_candidates.append(tadjmat)
                    score = CGNN.parallel_graph_evaluation(self, data=data, adj_matrix=tadjmat.todense())[1]
                    if score < best_score:
                        can_improve = True
                        best_candidate = test_graph
                        best_score = score
                        break
        return (best_candidate, best_score)
####################################################################################################################################################################################################################

# Debug
obj_main = CGNN(nh=20, batch_size=1000, train_epochs=4, test_epochs=4, sample_size=10000)
obj_main.create_graph_from_data(data=pd.read_csv('./Data/high_m.csv'), 
                                given_candidates_path='./Data/candidates_debug.txt', 
                                saving_path='./Models/solutions_debug.txt',
                                return_weights=False)

# Scenario main
# obj_main = CGNN(nh=20, batch_size=1000, train_epochs=500, test_epochs=20, sample_size=30000)
# obj_main.create_graph_from_data(data=pd.read_csv('./Data/high_m.csv'), 
#                                 given_candidates_path='./Data/candidates_m.txt', 
#                                 saving_path='./Models/solutions_m.txt',
#                                 return_weights=False)

###############################################
# Scenario hidden
# obj_hidden = CGNN(nh=20, batch_size=1000, train_epochs=500, test_epochs=20, sample_size=30000)
# obj_hidden.create_graph_from_data(data=pd.read_csv('./Data/high_h.csv'), 
#                                   given_candidates_path='./Data/candidates_h.txt', 
#                                   saving_path='./Models/solutions_h.txt', 
#                                   return_weights=False)
