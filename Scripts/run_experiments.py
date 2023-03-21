import cgnn_functions as cgnnFun
import cgnn_scc as cgnnFun_scc
import distance_functions as dFun
import utilities_functions as uFun
import os
###############################################################


def main(experiment: str):
    # ---------------------------------------------------------------
    # GLOBAL --------------------------------------------------------
    # ---------------------------------------------------------------
    #cw_dir = os.path.dirname(__file__)
    cw_dir = os.path.join(os.path.dirname(__file__), '..')
    print(cw_dir)

    cfg = uFun.load_json(os.path.join(cw_dir, 'Data/cfg.json'))

    data_paths = [
        os.path.join(cw_dir, 'Data/datasets/dat_train.csv'),
        os.path.join(cw_dir, 'Data/datasets/dat_val.csv'),
        os.path.join(cw_dir, 'Data/datasets/dat_test.csv'),
        ]

    data_paths_calib = [
        os.path.join(cw_dir, 'Data/datasets/dat_calib_high.csv'),
        os.path.join(cw_dir, 'Data/datasets/dat_calib_noCorr.csv')
    ]

    criteria_train = [
        dFun.MMD_cdt(bandwidth=[1]),
        dFun.CorrD([1, 1], 3),
        dFun.CndD(
            sample_weighting=True,
            sampling_rate=3,
            num_std_moments=4,
            weighting_exp=1,
            )
    ]
    criteria_train_wgs = []

    # ---------------------------------------------------------------
    # Experiments ---------------------------------------------------
    # ---------------------------------------------------------------
    if experiment == 'hu_optimization':
        for arc_nb_base_hu in list(range(10, 80)):
            model = cgnnFun.CGNN(
                arc_nb_base_hu=arc_nb_base_hu,
                arc_nb_layers=cfg['arc_nb_layers'],
                arc_wg_conn=cfg['arc_wg_conn'],
                lr=cfg['lr'],
                patience=cfg['patience'],
                loss_sum_pow=cfg['loss_sum_pow'],
                criteria_train=criteria_train,
                criteria_train_wgs=criteria_train_wgs,
                data_paths=data_paths,
                data_paths_calib=data_paths_calib,
                results_path=os.path.join(cw_dir, 'Results', experiment),
                mode_sgl_backprop=cfg['mode_sgl_backprop'],
                show_cand_plot=cfg['show_cand_plot'],
                evaluate_training=cfg['evaluate_training'],
                nb_cpus=cfg['nb_cpus'],
            )
            model.evaluate_candidates(
                eval_name='_{}'.format(arc_nb_base_hu),
                nb_runs=cfg['nb_runs'],
                train_epochs=cfg['train_epochs'],
                test_epochs=cfg['test_epochs'],
                candidates_path=os.path.join(cw_dir, 'Data/candidates/saturated.txt'),
                batch_size_REG=cfg['batch_size_REG'],
                batch_size_MMD=cfg['batch_size_MMD'],
                train_sample_size=cfg['train_sample_size'],
                valid_sample_size=cfg['valid_sample_size'],
                test_sample_size=cfg['test_sample_size'],
                calib_sample_size=cfg['calib_sample_size'],
                gen_sample_size=cfg['gen_sample_size'],
                candidates_ids=None,
                nodes_type=None,
                dim_q=cfg['dim_q'],
            )
    elif experiment == 'ccm':
        model = cgnnFun.CGNN(
            arc_nb_base_hu=cfg['arc_nb_base_hu'],
            arc_nb_layers=cfg['arc_nb_layers'],
            arc_wg_conn=cfg['arc_wg_conn'],
            lr=cfg['lr'],
            patience=cfg['patience'],
            loss_sum_pow=cfg['loss_sum_pow'],
            criteria_train=criteria_train,
            criteria_train_wgs=criteria_train_wgs,
            data_paths=data_paths,
            data_paths_calib=data_paths_calib,
            results_path=os.path.join(cw_dir, 'Results', experiment),
            mode_sgl_backprop=cfg['mode_sgl_backprop'],
            show_cand_plot=cfg['show_cand_plot'],
            evaluate_training=cfg['evaluate_training'],
            nb_cpus=cfg['nb_cpus'],
        )
        model.evaluate_candidates(
            eval_name='',
            nb_runs=cfg['nb_runs'],
            train_epochs=cfg['train_epochs'],
            test_epochs=cfg['test_epochs'],
            candidates_path=os.path.join(cw_dir, 'Data/candidates/ccm.txt'),
            batch_size_REG=cfg['batch_size_REG'],
            batch_size_MMD=cfg['batch_size_MMD'],
            train_sample_size=cfg['train_sample_size'],
            valid_sample_size=cfg['valid_sample_size'],
            test_sample_size=cfg['test_sample_size'],
            calib_sample_size=cfg['calib_sample_size'],
            gen_sample_size=cfg['gen_sample_size'],
            candidates_ids=list(range(0, 57)),
            nodes_type=None,
            dim_q=cfg['dim_q'],
            )
    elif experiment == 'qcm':
        model = cgnnFun.CGNN(
            arc_nb_base_hu=cfg['arc_nb_base_hu'],
            arc_nb_layers=cfg['arc_nb_layers'],
            arc_wg_conn=cfg['arc_wg_conn'],
            lr=cfg['lr'],
            patience=cfg['patience'],
            loss_sum_pow=cfg['loss_sum_pow'],
            criteria_train=criteria_train,
            criteria_train_wgs=criteria_train_wgs,
            data_paths=data_paths,
            data_paths_calib=data_paths_calib,
            results_path=os.path.join(cw_dir, 'Results', experiment),
            mode_sgl_backprop=cfg['mode_sgl_backprop'],
            show_cand_plot=cfg['show_cand_plot'],
            evaluate_training=cfg['evaluate_training'],
            nb_cpus=cfg['nb_cpus'],
        )
        model.evaluate_candidates(
            eval_name='',
            nb_runs=cfg['nb_runs'],
            train_epochs=cfg['train_epochs'],
            test_epochs=cfg['test_epochs'],
            candidates_path=os.path.join(cw_dir, 'Data/candidates/qcm.txt'),
            batch_size_REG=cfg['batch_size_REG'],
            batch_size_MMD=cfg['batch_size_MMD'],
            train_sample_size=cfg['train_sample_size'],
            valid_sample_size=cfg['valid_sample_size'],
            test_sample_size=cfg['test_sample_size'],
            calib_sample_size=cfg['calib_sample_size'],
            gen_sample_size=cfg['gen_sample_size'],
            candidates_ids=None,
            nodes_type=['c', 'c', 'c', 'c', 'q'],
            dim_q=cfg['dim_q'],
            )
    elif experiment == 'icm':
        model = cgnnFun.CGNN(
            arc_nb_base_hu=cfg['arc_nb_base_hu'],
            arc_nb_layers=cfg['arc_nb_layers'],
            arc_wg_conn=cfg['arc_wg_conn'],
            lr=cfg['lr'],
            patience=cfg['patience'],
            loss_sum_pow=cfg['loss_sum_pow'],
            criteria_train=criteria_train,
            criteria_train_wgs=criteria_train_wgs,
            data_paths=data_paths,
            data_paths_calib=data_paths_calib,
            results_path=os.path.join(cw_dir, 'Results', experiment),
            mode_sgl_backprop=cfg['mode_sgl_backprop'],
            show_cand_plot=cfg['show_cand_plot'],
            evaluate_training=cfg['evaluate_training'],
            nb_cpus=cfg['nb_cpus'],
            )
        model.evaluate_candidates(
            eval_name='',
            nb_runs=cfg['nb_runs'],
            train_epochs=cfg['train_epochs'],
            test_epochs=cfg['test_epochs'],
            candidates_path=os.path.join(cw_dir, 'Data/candidates/icm.txt'),
            batch_size_REG=cfg['batch_size_REG'],
            batch_size_MMD=cfg['batch_size_MMD'],
            train_sample_size=cfg['train_sample_size'],
            valid_sample_size=cfg['valid_sample_size'],
            test_sample_size=cfg['test_sample_size'],
            calib_sample_size=cfg['calib_sample_size'],
            gen_sample_size=cfg['gen_sample_size'],
            candidates_ids=None,
            nodes_type=['ns', 'ns', 'c', 'c'],
            dim_q=cfg['dim_q'],
            )
    elif experiment == 'qdim_optimization':
        dims = [2, 3, 4, 5, 6, 7]
        for i, dim_q in enumerate(dims):
            model = cgnnFun.CGNN(
                arc_nb_base_hu=cfg['arc_nb_base_hu'],
                arc_nb_layers=cfg['arc_nb_layers'],
                arc_wg_conn=cfg['arc_wg_conn'],
                lr=cfg['lr'],
                patience=cfg['patience'],
                loss_sum_pow=cfg['loss_sum_pow'],
                criteria_train=criteria_train,
                criteria_train_wgs=criteria_train_wgs,
                data_paths=data_paths,
                data_paths_calib=data_paths_calib,
                results_path=os.path.join(cw_dir, 'Results', experiment, f'dim_{dim_q}'),
                mode_sgl_backprop=cfg['mode_sgl_backprop'],
                show_cand_plot=cfg['show_cand_plot'],
                evaluate_training=cfg['evaluate_training'],
                nb_cpus=cfg['nb_cpus'],
            )
            model.evaluate_candidates(
                eval_name='',
                nb_runs=cfg['nb_runs'],
                train_epochs=cfg['train_epochs'],
                test_epochs=cfg['test_epochs'],
                candidates_path=os.path.join(cw_dir, 'Data/candidates/qcm.txt'),
                batch_size_REG=cfg['batch_size_REG'],
                batch_size_MMD=cfg['batch_size_MMD'],
                train_sample_size=cfg['train_sample_size'],
                valid_sample_size=cfg['valid_sample_size'],
                test_sample_size=cfg['test_sample_size'],
                calib_sample_size=cfg['calib_sample_size'],
                gen_sample_size=cfg['gen_sample_size'],
                candidates_ids=None,
                nodes_type=['c', 'c', 'c', 'c', 'q'],
                dim_q=dim_q,
                )
    elif experiment == 'scc':
        
        model = cgnnFun_scc.CGNN(
            arc_nb_base_hu=cfg['arc_nb_base_hu'],
            arc_nb_layers=cfg['arc_nb_layers'],
            arc_wg_conn=cfg['arc_wg_conn'],
            lr=cfg['lr'],
            patience=cfg['patience'],
            loss_sum_pow=cfg['loss_sum_pow'],
            criteria_train=criteria_train,
            criteria_train_wgs=criteria_train_wgs,
            data_paths=data_paths,
            data_paths_calib=data_paths_calib,
            results_path=os.path.join(cw_dir, 'Results', experiment),
            mode_sgl_backprop=cfg['mode_sgl_backprop'],
            show_cand_plot=cfg['show_cand_plot'],
            evaluate_training=cfg['evaluate_training'],
            nb_cpus=cfg['nb_cpus'],
            )
        model.evaluate_candidates(
            eval_name='',
            nb_runs=cfg['nb_runs'],
            train_epochs=cfg['train_epochs'],
            test_epochs=cfg['test_epochs'],
            candidates_path=os.path.join(cw_dir, 'Data/candidates/scc.txt'),
            batch_size_REG=cfg['batch_size_REG'],
            batch_size_MMD=cfg['batch_size_MMD'],
            train_sample_size=cfg['train_sample_size'],
            valid_sample_size=cfg['valid_sample_size'],
            test_sample_size=cfg['test_sample_size'],
            calib_sample_size=cfg['calib_sample_size'],
            gen_sample_size=cfg['gen_sample_size'],
            candidates_ids=None,
            nodes_type=None,
            dim_q=cfg['dim_q'],
            )
    else:
        raise IOError('Unknown experiment')


if __name__ == '__main__':
    experiment = input(
        'Please insert the experiment name from the list below:\n'
        '[ccm, qcm, icm, scc, hu_optimization, qdim_optimization: '
    )
    main(experiment)
