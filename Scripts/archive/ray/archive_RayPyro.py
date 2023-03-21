
# Ray_Pyro

def train_cgnn(config, datasets, train_epochs, start_val_epochs, checkpoint_dir):
    
    batch_size=1000; nh=20; lr=.001; adjacency_matrix=np.array([[0., 0, 0., 0.], [0., 0., 0., 0.], [0, 0, 0., 0.], [0, 0, 0., 0.]])
    #adjacency_matrix=np.array([[0., 1., 0., 0.], [0., 0., 0., 0.], [1., 1., 0., 0.], [1., 1., 0., 0.]])

    criterion_1 = dFun.MMD(
        kernel_counts=config["kernel_counts"],
        kernel_name=config["kernel_name"], 
        lambda_c=0,
        wg_mmd_s=1,
        wg_mmd_c=0, 
        variances=th.tensor([0.1483, 0.1958, 0.0329, 0.2135, 0.0034, 0.0656, 0.0266, 0.0116, 0.0108, 0.2915]),
        lengthscales=None)
    
    criterion_2 = dFun.corrD(
        sampling_rate=4, 
        num_std_moments=4, 
        wg_corrD_m=config["wg_corrD_m"], 
        wg_corrD_c=1,)

    model=cgnn.CGNN_model(
        adjacency_matrix=adjacency_matrix, 
        batch_size=batch_size, 
        nh=nh, 
        lr=lr, 
        train_epochs=train_epochs, 
        guassian_w=config["guassian_w"], 
        uniform_w=config["uniform_w"],
        activation_tresh=config["activation_tresh"])
    
    model.reset_parameters()
    optimizer = th.optim.Adam(model.parameters(), lr=lr);
    
    try:
        model_state, optimizer_state = th.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    except:
        pass
    
    train_dataset, val_dataset= datasets
    train_loader1 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader1   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader2   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, drop_last=True)

    
    parameters={
        'criterion_1': relevent_attributes(criterion_1), 
        'criterion_2': relevent_attributes(criterion_2), 
        'CGNN'       : relevent_attributes(model), 
        'other'      : {'start_val_epochs':start_val_epochs, 'train_sample_size': len(train_dataset)}
    }
    
    for inx_epoch, epoch in enumerate(range(train_epochs)):
        for inx_train, (train_data1, train_data2) in enumerate(zip(train_loader1, train_loader2)):
            optimizer.zero_grad()
            gen_data=model.generator(batch_size)
            loss_1 = criterion_1(train_data1, gen_data); calib_1= criterion_1(train_data1, train_data2); 
            qual_1 = calib_1/loss_1; qual_1 = qual_1/qual_1 if qual_1>1 else qual_1
            loss_2 = criterion_2(train_data1, gen_data); calib_2= criterion_2(train_data1, train_data2); 
            qual_2 = calib_2/loss_2; qual_2 = qual_2/qual_2 if qual_2>1 else qual_2
            
            qual= (qual_1+ config["wg_q2"]*qual_2)/(1+config["wg_q2"])
            train_loss = 1/qual 
            train_loss.backward()
            optimizer.step()
        if inx_epoch>start_val_epochs:
            loss_1=[]; calib_1=[]; qual_1=[]; loss_2=[]; calib_2=[]; qual_2=[]; qual=[]
            for inx_val, (val_data1, val_data2) in enumerate(zip(val_loader1, val_loader2)):
                gen_data = model.generator(batch_size); 
                _loss_1  = criterion_1(val_data1, gen_data); _calib_1 = criterion_1(val_data1, val_data2); 
                _qual_1  = _calib_1/_loss_1; _qual_1 = _qual_1/_qual_1 if _qual_1>1 else _qual_1
                _loss_2  = criterion_2(val_data1, gen_data); _calib_2 = criterion_2(val_data1, val_data2); 
                _qual_2  = _calib_2/_loss_2; _qual_2 = _qual_2/_qual_2 if _qual_2>1 else _qual_2
                _qual=_qual_1+_qual_2
                _qual= (_qual_1+ config["wg_q2"]*_qual_2)/(1+config["wg_q2"])
                loss_1.append(_loss_1.item()); calib_1.append(_calib_1.item()); qual_1.append(_qual_1.item()); loss_2.append(_loss_2.item()); calib_2.append(_calib_2.item()); qual_2.append(_qual_2.item()); 
                qual.append(_qual.item())
                
                
            tune.report(qual=np.mean(qual), 
                        qual_1=np.mean(qual_1), 
                        qual_2=np.mean(qual_2),
                        loss_1=np.mean(loss_1), 
                        calib_1=np.mean(calib_1),
                        loss_2=np.mean(loss_2), 
                        calib_2=np.mean(calib_2), 
                        parameters=parameters)
            
            with tune.checkpoint_dir(epoch) as checkpoint_dir: torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))
    print("Finished Training")
    
        
num_samples=8
train_sample_size=10000
train_epochs=100
start_val_epochs=5
scheduler_epochs=100

checkpoint_dir='/Users/omid/Documents/GitHub/Causality/CGNN_EPR/Debug/Extra/'
train_dataset=th.Tensor(pd.read_csv('./Data/train.csv').sample(n=train_sample_size).values)
val_dataset  =th.Tensor(pd.read_csv('./Data/val.csv').sample(n=train_sample_size).values)
datasets=(train_dataset, val_dataset)

config = {
    'kernel_counts': tune.choice([10]),
    'kernel_name'  : tune.choice(['RBF']),
    'uniform_w'    : tune.choice([0]), #tune.uniform(0,1), 
    'guassian_w'   : tune.choice([1]), #tune.uniform(0,1), 
    'wg_q2'        : tune.choice([10]), 
    'wg_corrD_m'   : tune.choice([0]), 
    'activation_tresh':tune.choice([1])}#0.875087

scheduler = ASHAScheduler(metric='qual', mode='max', max_t=scheduler_epochs) 
#scheduler=None

search_alg= HyperOptSearch(metric="qual", mode="max")
#search_alg=None

analysis = tune.run(
    tune.with_parameters(train_cgnn, datasets=datasets, checkpoint_dir=checkpoint_dir, train_epochs=train_epochs, start_val_epochs=start_val_epochs), 
    config=config, 
    num_samples=num_samples, 
    search_alg=search_alg,
    scheduler=scheduler, 
    verbose=1,
    local_dir="./Extra/")

df=analysis.results_df
df.sort_values(by=['qual'], ascending=False, inplace=True)
df


def kernel_iter(pop_counts=[7,8,9,10], pop_names=['RBF', 'Cosine', 'Exponential'], pop_lengthscales=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3]):
    for kernel_counts in pop_counts:
        for kernel_names in [random.choices(population=pop_names, k=kernel_counts) for i in range(20)]:
            for lengthscales in [random.choices(population=pop_lengthscales, k=kernel_counts) for i in range(20)]:
                for variances in [[10*random.random() for k in range(kernel_counts)] for i in range(20)]:
                    yield(kernel_names, lengthscales, variances)
                    


##############################################################################
# multi_generator
##############################################################################

def hyperloss_evaluator(l_c, l_t, l_td, sampling_rate, num_std_moments, wgCorr, wgMarg,
                        L1_w, L2_w, guassian_w, uniform_w, adjacency_matrix, nh, lr, train_epochs, batch_size, train_sample_size, test_sample_size,
                        criterion_index, model_saving_path, generated_saving_path, plot_saving_path, eval_df_saving_path, meta_saving_path, 
                        bw_m, bw_c, patience):
    
    criteria_list=[dFun.MMD_CorrD(bw_m=th.Tensor(bw_m), bw_c=th.Tensor(bw_c), l_c=l_c, l_t=l_t, l_td=l_td, only_standard=False, sampling_rate=sampling_rate, num_std_moments=num_std_moments, wgCorr=wgCorr, wgMarg=wgMarg),
                   dFun.MMD_s_m(lengthscales=th.Tensor(bw_m), wgMarg=wgMarg), 
                   dFun.MMD_s(th.Tensor(bw_m)),
                   dFun.MMD_s_c(l_c=l_c, bw_m=th.Tensor(bw_m), bw_c=th.Tensor(bw_c)),
                   dFun.CorrD(sampling_rate=sampling_rate, num_std_moments=num_std_moments, wgCorr=wgCorr, wgMarg=wgMarg)]
    
    trained_model= CGNN_trainer(adjacency_matrix=adjacency_matrix, nh=nh, lr=lr, train_epochs=train_epochs, sample_size=train_sample_size, batch_size=batch_size, criterion=criteria_list[criterion_index], 
                                patience=patience, L1_w=L1_w, L2_w=L2_w, guassian_w=guassian_w,uniform_w=uniform_w,saving_path=model_saving_path)
    generated_data=CGNN_generator(model=trained_model, gen_size=test_sample_size, saving_path=generated_saving_path)
    dist_plotter(generated_data, saving_path=plot_saving_path, show_plot=True)
    df_evl=CGNN_evaluator(trained_model, sample_size=test_sample_size, criteria_list=criteria_list, saving_path=eval_df_saving_path)
    df=pd.DataFrame()
    for dis_inx in range(len(df_evl.distance.tolist())):
        df.loc[0, 'quality_{}'.format(dis_inx)]=df_evl.quality.tolist()[dis_inx]
        df.loc[0, 'distance_{}'.format(dis_inx)]=df_evl.distance.tolist()[dis_inx]
        df.loc[0, 'calibrator_{}'.format(dis_inx)]=df_evl.calibrator.tolist()[dis_inx] 
        
    df['l_c']= l_c; df['l_t']= l_t; df['l_td']=l_td; df['sampling_rate']= sampling_rate; df['num_std_moments']= num_std_moments; df['wgCorr']= wgCorr;df['wgMarg']= wgMarg;
    df['L1_w']= L1_w;df['L2_w']= L2_w;df['guassian_w']= guassian_w;df['uniform_w']= uniform_w;df['adjacency_matrix']= str(adjacency_matrix);
    df['nh']= nh;df['lr']= lr;
    df['train_epochs']= train_epochs;df['batch_size']= batch_size;df['test_sample_size']= test_sample_size;df['criterion_index']= criterion_index;
    df['model_saving_path']= model_saving_path;df['generated_saving_path']= generated_saving_path;df['plot_saving_path']= plot_saving_path;
    df['eval_df_saving_path']= eval_df_saving_path; df['trained_model']= trained_model; df['meta_saving_path']=meta_saving_path;
    ##
    df['bw_m']=str(bw_m); df['bw_c']=str(bw_c); df['patience']=str(patience); 
    if meta_saving_path: object_saver(df, meta_saving_path)
    return df

##############################################################################
##############################################################################
# uni_generator
##############################################################################
##############################################################################

def uni_plotter(generated_data, original_data):
    #sns.set_style("dark")
    df=pd.DataFrame(generated_data.detach().numpy())
    df=df.rename(columns={0: "GenData"})
    df['OrigData']=original_data
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    sns.histplot(ax=axes[0], data=df, x='OrigData', stat='density', kde=True, bins=20, line_kws=dict(linewidth=1))
    sns.histplot(ax=axes[1], data=df, x='GenData', stat='density', kde=True, bins=20, line_kws=dict(linewidth=1))
    axes[0].set_ylabel(''); axes[1].set_ylabel(''); axes[0].set_xlabel('OrigData'); axes[1].set_xlabel('GenData');
    plt.show()
    plt.clf()
    plt.close()


def uni_forward(model, batch_size):
    noise=th.normal(0, 1, size=(batch_size, 1))
    #noise=th.zeros(batch_size, 1).normal_()
    return model(noise)

def uni_run(dataset, criterion, nh, lr, batch_size, train_epochs):
    model=CGNN_block([1, nh, 1]); 
    model.reset_parameters()
    optim = th.optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    e_verbose=trange(train_epochs, disable=False)
    for epoch in e_verbose:
        for inx, data in enumerate(dataloader):
            optim.zero_grad()
            gen_data=uni_forward(model, batch_size)
            loss = criterion(gen_data, data)
            loss.backward()
            optim.step()
            if not epoch%10 and inx == 0: e_verbose.set_postfix(loss=loss.item())
    return model


##############################################################################
##############################################################################
