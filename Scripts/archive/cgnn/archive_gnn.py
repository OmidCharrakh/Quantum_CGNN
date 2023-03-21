import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import distance_functions as dFun
#####################################################

class GNN_model(th.nn.Module):
    def __init__(self, batch_size, nh, lr, train_epochs, test_epochs):
        super(GNN_model, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.criterion = dFun.MMD_cdt()
        self.l1 = th.nn.Linear(2, nh)
        self.l2 = th.nn.Linear(nh, 1)
        self.register_buffer('noise', th.Tensor(batch_size, 1))
        self.act = th.nn.ReLU()
        self.layers = th.nn.Sequential(self.l1, self.act, self.l2)
    def forward(self, x):
        self.noise.normal_()
        return self.layers(th.cat([x, self.noise], 1))
    def run(self, dataset):
        optim = th.optim.Adam(self.parameters(), lr=self.lr)
        pbar = trange(self.train_epochs + self.test_epochs, disable=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
        test_loss = 0
        for epoch in pbar:
            for i, (x, y) in enumerate(dataloader):
                optim.zero_grad()
                pred = self.forward(x)
                mmd = self.criterion(th.cat([x, pred], 1), th.cat([x, y], 1))
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
#####################################################
def GNN_instance(data, nh, batch_size, **kwargs):
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
#####################################################
class GNN:
    def __init__(self, nh, batch_size, train_epochs, test_epochs, nruns=1, lr=0.01):
        super(GNN, self).__init__()
        self.nh = nh
        self.lr = lr
        self.nruns = nruns
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
    def predict_proba(self, dataset):
        data = [th.Tensor(th.Tensor(i).view(-1, 1)) for i in dataset]
        AB = []; BA = []
        result_pair = [GNN_instance(data, nh=self.nh, batch_size=self.batch_size, train_epochs=self.train_epochs, test_epochs=self.test_epochs, lr=self.lr) for run in range(self.nruns)]
        AB.extend([runpair[0] for runpair in result_pair]) 
        BA.extend([runpair[1] for runpair in result_pair])
        loss_AB = np.mean(AB); loss_BA = np.mean(BA)
        return (loss_AB, loss_BA)
#####################################################

def tuner_generator(nodes='OA_OB',
                     n_points=100, 
                     n_pairs=100,
                     data_path='./Data/high_m.csv', 
                     saving_path='OA_OB.csv'):
    df = pd.read_csv(data_path)
    gnn_data=pd.DataFrame(columns=['A', 'B'])
    pair_dict={'OA_OB':[0,1],'OA_SA':[0,2],'OA_SB':[0,3],'OB_SA':[1,2],'OB_SB':[1,3],'SA_SB':[2,3]}
    for pair in range(n_pairs):
        rand_idx=sample(set(df.index), n_points)
        A=[df.iloc[inx, pair_dict[nodes][0]] for inx in rand_idx]
        B=[df.iloc[inx, pair_dict[nodes][1]] for inx in rand_idx]
        gnn_data = gnn_data.append({'A': A, 'B': B}, ignore_index=True) 
    gnn_data.to_csv(saving_path, index=False)
#####################################################
def nh_tuner(nh_range=range(1,10,1), 
              nb_runs=32, 
              train_epochs=20, 
              test_epochs=3, 
              batch_size=-1,
              data_path='./Data/gnn_data.csv',
              saving_path_1= 'saving_path_1.txt', 
              saving_path_2= 'saving_path_2.txt'):
    gnn_data=pd.read_csv(data_path, converters={"A": ast.literal_eval, "B": ast.literal_eval})
    for idx_nh, nh in enumerate(nh_range):
        t_i=time.time()
        loss_AB=np.zeros(nb_runs, dtype=float)
        loss_BA=np.zeros(nb_runs, dtype=float)
        for run in range(nb_runs):
            obj = gnnFun.GNN(nh=nh, train_epochs=train_epochs, test_epochs=test_epochs, batch_size=batch_size, nruns=1)
            losses=[obj.predict_proba(gnn_data.iloc[i]) for i in range(0, len(gnn_data))] 
            loss_AB[run]=np.sum([losses[i][0] for i in range(len(losses))])
            loss_BA[run]=np.sum([losses[i][1] for i in range(len(losses))])
        f1=open(saving_path_1,'a'); f2=open(saving_path_2,'a')
        np.savetxt(f1, loss_AB, fmt='%f', newline=" "); np.savetxt(f2, loss_BA, fmt='%f', newline=" ")
        f1.write("\n"); f2.write("\n")
        f1.close(); f2.close()
        t_f=time.time()
        print('Completed Round:', str(idx_nh+1), 'Needed Time:', round((t_f-t_i),2))
##################################################### 
def tuner_plotter(nh_range, 
                  nodes, 
                  loss_AB_path, 
                  loss_BA_path, 
                  saving_path):
    
    loss_AB=np.loadtxt(loss_AB_path)
    loss_BA=np.loadtxt(loss_BA_path)
    df=pd.DataFrame(data= {'nh': nh_range})
    df['loss_AB']= np.mean(loss_AB, axis=1)
    df['loss_BA']= np.mean(loss_BA, axis=1)
    df['Diff']   = df['loss_AB']-df['loss_BA']
    df['std_AB'] = np.std(loss_AB, axis=1)
    df['std_BA'] = np.std(loss_BA, axis=1)
    label_dict={'OA_OB':[r'$O_A\rightarrow O_B$', r'$O_B\rightarrow O_A$'],
            'OA_SA':[r'$O_A\rightarrow S_A$', r'$S_A\rightarrow O_A$'],
            'OA_SB':[r'$O_A\rightarrow S_B$', r'$S_B\rightarrow O_A$'],
            'OB_SA':[r'$O_B\rightarrow S_A$', r'$S_A\rightarrow O_B$'],
            'OB_SB':[r'$O_B\rightarrow S_B$', r'$S_B\rightarrow O_B$'],
            'SA_SB':[r'$S_A\rightarrow S_B$', r'$S_B\rightarrow S_A$']}
    plt.figure(figsize=(8,5))
    plt.errorbar(df['nh'], df['loss_AB'], color='r', yerr=df['std_AB'], label=label_dict[nodes][0])
    plt.errorbar(df['nh'], df['loss_BA'], color='b', yerr=df['std_BA'], label=label_dict[nodes][1])
    plt.xlabel('Number of hidden units', weight='bold')
    plt.ylabel('Loss', weight='bold')
    plt.xticks(df['nh'], df['nh'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(saving_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.close()
    return df
#####################################################