def is_mediator(A,i,j,k):
    G = nx.DiGraph(A)
    paths_ij = [path for path in nx.all_simple_paths(G, i, j)]
    return any([k in path for path in paths_ij])
def is_commoncause(A,i,j,k):
    G = nx.DiGraph(A)
    paths_ki = [path for path in nx.all_simple_paths(G, k, i)]
    paths_kj = [path for path in nx.all_simple_paths(G, k, j)]
    return len(paths_ki)*len(paths_kj)>0

# version 1
def count_emergConns(A, k):
    A_prune = A.copy(); A_prune[:4,:4]=0
    nb_emergConns = 0
    for i in range(4):
        for j in range(1+i,4):
            s1 = not nx.d_separated(nx.DiGraph(A_prune), {i}, {j}, {})
            s2 = nx.d_separated(nx.DiGraph(A_prune), {i}, {j}, {k})
            if s1:
                if s2:
                    print((i,j), s1, s2)
                    nb_emergConns +=1 
    return nb_emergConns

# version 2
def count_emergConns(A, k):
    nb_emergConns = 0
    A_cc = np.zeros((4,4), int)
    A_md = np.zeros((4,4), int)
    for i in range(4):
        for j in range(1+i, 4):
            A_md[i,j] = is_mediator(A,i,j,k)
            A_md[j,i] = is_mediator(A,j,i,k)
            A_cc[i,j] = is_commoncause(A,i,j,k)
            A_cc[j,i] = is_commoncause(A,j,i,k)
            if A_md[i,j] or A_md[j,i] or A_cc[i,j] or A_cc[j,i]:
                nb_emergConns +=1
    return nb_emergConns

def count_hu(A, nb_base_hu, nb_layers, wg_conn, wg_latent, fix_hu_nb, latent_incommings=None):
    nb_vars = A.shape[0]
    hu_matrix = np.zeros((nb_vars, nb_layers))
    if nb_vars==4:
        if latent_incommings is None:
            latent_incommings = [0,0,0,0] 
        for n in range(nb_vars):
            nb_incomings = A.sum(0)[n]
            nb_incomings_latent = latent_incommings[n]
            if fix_hu_nb:
                nb_hu = np.round(nb_vars*nb_base_hu*(1+wg_conn*nb_incomings+wg_latent*nb_incomings_latent)/(wg_conn*A.sum()+wg_latent*latent_incommings.sum()+nb_vars)) 
            else:
                nb_hu = np.round(nb_base_hu*(1+wg_conn*nb_incomings+wg_latent*nb_incomings_latent))
            nb_layers_hu = np.ones([nb_layers],int) * np.floor(nb_hu / nb_layers).astype(int)
            index_layer = 0
            while np.sum(nb_layers_hu) < nb_hu: 
                nb_layers_hu[index_layer] += 1
                index_layer += 1
            hu_matrix[n] = nb_layers_hu
    elif nb_vars>4:
        A_bs = A[:4,:4]
        A_cc, A_me = extract_latent_connections(A)
        latent_incommings = (A_cc+A_me).sum(0)
        hu_matrix[0:4,:] = count_hu(A_bs, nb_base_hu, nb_layers, wg_conn, wg_latent, fix_hu_nb, latent_incommings)
        mean_hu_perEdge = hu_matrix.sum()/(4+A_bs.sum())

        for n in range(4, nb_vars):
            nb_incomings = A.sum(0)[n]
            print(mean_hu_perEdge, nb_incomings)
            nb_hu_perLayer = np.round(mean_hu_perEdge*(1+wg_conn*nb_incomings)/nb_layers)
            hu_matrix[n,:] = [nb_hu_perLayer for _ in range(nb_layers)]
    return hu_matrix.astype(int)

def count_hu(A, nb_base_hu, nb_layers, wg_conn, wg_latent, fix_hu_nb):
    nb_vars = A.shape[0]
    hu_matrix = np.zeros((nb_vars, nb_layers))
    if nb_vars==4:
        for n in range(nb_vars):
            nb_incomings = A[:,n].sum()
            if fix_hu_nb:
                nb_hu = np.round(nb_vars*nb_base_hu*(1+wg_conn*nb_incomings)/(wg_conn*A.sum()+nb_vars)) 
            else:
                nb_hu = np.round(nb_base_hu*(1+wg_conn*nb_incomings))
            nb_layers_hu = np.ones([nb_layers],int) * np.floor(nb_hu/nb_layers).astype(int)
            index_layer = 0
            while np.sum(nb_layers_hu)<nb_hu: 
                nb_layers_hu[index_layer]+=1
                index_layer+=1
            hu_matrix[n] = nb_layers_hu
    elif nb_vars>4:
        A_eq = get_equivGraph(A)
        hu_matrix[:4,:] = count_hu(A_eq, nb_base_hu, nb_layers, wg_conn, wg_latent, fix_hu_nb)
        for k in range(4, nb_vars):
            nb_emergConns = count_emergConns(A, k)
            nb_hu = np.round(nb_base_hu*(1+wg_latent*nb_emergConns))
            nb_layers_hu = np.ones([nb_layers],int) * np.floor(nb_hu/nb_layers).astype(int)
            index_layer = 0
            while np.sum(nb_layers_hu) < nb_hu: 
                nb_layers_hu[index_layer] += 1
                index_layer += 1
            hu_matrix[k] = nb_layers_hu
    return hu_matrix.astype(int)

def get_equivGraph(A):
    A_bs = A[:4,:4].copy()
    A_cc, A_me = extract_latent_connections(A)
    A_latent = A_cc+A_me
    A_eq = A_bs.copy() 
    for i in range(4):
        for j in range(i+1, 4):
            if not (A_bs[i,j]+A_bs[j,i]):
                if A_latent[i,j]:
                    A_eq[i,j]=1
                elif A_latent[j,i]:
                    A_eq[j,i]=1
    return A_eq

def edge_importance(A, edge):
    i,j = edge
    A0, A1 = A.copy(), A.copy()
    A0[i,j]=0; A1[i,j]=1
    c0, c1 = count_connections(A0), count_connections(A1)
    return c1, c0

def extract_latent_connections(A):
    n_vars = A.shape[0]
    hVar_ids = list(range(4,n_vars))
    A_cc = np.zeros((4,4), int); 
    A_me = np.zeros((4,4), int); 
    if n_vars>4:
        for i in range(4):
            for j in range(4):
                for index_h in hVar_ids:
                    A_cc[i,j]+=A[index_h,i]*A[index_h,j]
                    A_me[i,j]+=A[i,index_h]*A[index_h,j]
    return(A_cc, A_me)

def path_counter(A, source, target):
    G = nx.DiGraph(A)
    paths = [path for path in nx.all_simple_paths(G, source, target)]
    return len(paths)