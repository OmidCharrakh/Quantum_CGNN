import os
import cgnn_functions as cgnnFun
import distance_functions as dFun
#cw_dir = '/home/charrakho/causal/'
cw_dir = '/Users/omid/Documents/GitHub/Causality/empiricalCausality/CGNN/'
##############################################################################

# [MMDs]:
# MMD_cdt(bandwidth=[1])
# MMD_s_pro(kernels=['Cos'],bandwidths=[1,10],variances=[0.1,1,10]),
# MMD_Fourier(bandwidths = [0.1, 1, 10, 100], n_RandComps=100)
# [Custom]:
# CorrD_N(),
# CorrD([1,1],3)
# NpMom(num_moments = 3, weighting_exp = 2)
# CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1)
# [Marginals]:
# MargD(num_std_moments=4)

#[SINGLE DISTANCE TEST]
# eval_name = 'comb1_MargD'
# criteria_train = [dFun.MargD(num_std_moments=4)]
# criteria_train_wgs = [1]
#[COMBINATION TEST]
#eval_name = 'test'
eval_name = 'comb3_snr_0wg_1l_mixedN_40_newArc' 
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD([1,1],3)] #'comb1'
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD([1,1],3),dFun.NpMom(num_moments = 3, weighting_exp = 2)] #'comb1p'
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1)] #'comb2'
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1),dFun.NpMom(num_moments = 3, weighting_exp = 2)] #'comb2p'
criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD([1,1],3),dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1)] #'comb3'
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD([1,1],3),dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1),dFun.NpMom(num_moments = 3, weighting_exp = 2)] #'comb3p'
#criteria_train = [dFun.MargD(num_std_moments=4),dFun.CorrD([1,1],3),dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1)] #'comb4'
#criteria_train = [dFun.MargD(num_std_moments=4),dFun.CorrD([1,1],3),dFun.CndD(sample_weighting = True, sampling_rate = 3, num_std_moments = 4, weighting_exp = 1),dFun.NpMom(num_moments = 3, weighting_exp = 2)] #'comb4p'
#criteria_train_wgs = [1,1,1]
#criteria_train_wgs = [1,1,1,1]
criteria_train_wgs = []

#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD([1,1],3),dFun.CorrD_N(),dFun.MargD(num_std_moments=4)]
#criteria_train_wgs = [0.1,1,1,1]
#criteria_train_wgs = [1.5,40,35,30]
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD([1,1],3),dFun.MargD(num_std_moments=4)]
#criteria_train_wgs = [1.5,40,30]
#criteria_train = [dFun.MMD_cdt(bandwidth=[1]),dFun.CorrD_N(),dFun.MargD(num_std_moments=4)]
#criteria_train_wgs = [1.5,35,30]
#criteria_train = [dFun.CorrD([1,1],3),dFun.CorrD_N(),dFun.MargD(num_std_moments=4)]
#criteria_train_wgs = [40,35,30]
#criteria_train = [dFun.CorrD([1,1],3),dFun.MargD(num_std_moments=4)]
#criteria_train = [dFun.CorrD_N(),dFun.MargD(num_std_moments=4)]

CGNN_obj = cgnnFun.CGNN(
    arc_nb_base_hu = 40, #base number of hidden units
    arc_nb_layers = 2, #number of layers
    arc_wg_conn = 1, #how much weight do connections get, 0 means no additional hu for connections, 1 means same weight for connections as internal noise, etc.
    arc_wg_latent = 1, #how much weight do "latent" connections (i.e., mechanisms corresponding to hidden variables) get, 0 means no hu for hidden variables, 1 means mean_hu_perEdge x nb_equivEdges
    arc_fix_hu_nb = False, #if set to true, adding weight to the connections will not increase the total amount of hidden units in the model
    lr = 0.01, patience = 200,
    loss_sum_pow = 1,
    criteria_train = criteria_train,
    criteria_train_wgs = criteria_train_wgs,
    data_paths = [os.path.join(cw_dir, 'Data/dat_train.csv'), os.path.join(cw_dir, 'Data/dat_val.csv'), os.path.join(cw_dir, 'Data/dat_test.csv')], 
    data_paths_calib = [os.path.join(cw_dir, 'Data/dat_calib_high.csv'), os.path.join(cw_dir, 'Data/dat_calib_noCorr.csv')],
    data_path_saving = os.path.join(cw_dir, 'Results/'),
    mode_sgl_backprop = False, #NOT sgl backprop was the default way, and as far I can see multi-backprop also work clearly better (at least faster)
    show_cand_plot = True,
    evaluate_training = False, 
    nb_cpus=8) #number of cpus used for the computation

CGNN_obj.evaluate_all_candidates(
    eval_name = eval_name,
    nb_runs = 4,
    train_epochs = 200,
    test_epochs = 20,
    candidates_path = os.path.join(cw_dir, 'Data/candidates_debug_justX.txt'),
    batch_size = 8000,
    batch_size_MMD = 500,
    train_sample_size = 32000, #max 40k
    valid_sample_size = 16000, #max 20k
    test_sample_size = 16000, #max 20k
    calib_sample_size = 32000, #max 40k
    gen_sample_size = 20000, #for saved data
    cw_dir='') 

