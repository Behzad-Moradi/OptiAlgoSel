
from pflacco.classical_ela_features import *
from pflacco.misc_features import *
from pflacco.local_optima_network_features import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from Utils.initial_sample_gen import initial_sample_gen
from sklearn.utils import resample
import Utils.pure_smd_problems as SMDProblems
import Utils.lower_level_smd_problems as LLSMDProblems

np.random.seed(1)

def compute_ela_features(i, norm_doe_bbob_total, num_fun, num_inst, num_seed, dim, lower_bound, upper_bound, data_set_name):

    fid = i // (num_inst * num_seed) + 1
    iid = (i % (num_inst * num_seed)) // num_seed + 1
    sid = (i % (num_inst * num_seed)) % num_seed + 1  

    if data_set_name == 'BBOB':
        feature_dict = {'function': f'bbob_{fid}_{iid}_{sid}'}
    
    if data_set_name == 'SMD':
        feature_dict = {'function': f'smd_{fid}_{iid}_{sid}'}

    try:
        ela_meta = calculate_ela_meta(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in ela_meta for function f_{fid}_{iid}_{sid}: {e}")
        ela_meta = {'ela_meta': np.nan}
    
    try:
        ela_distr = calculate_ela_distribution(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in ela_distr for function f_{fid}_{iid}_{sid}: {e}")
        ela_distr = {'ela_distr': np.nan}
    
    try:
        ela_level = calculate_ela_level(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in ela_level for function f_{fid}_{iid}_{sid}: {e}")
        ela_level = {'ela_level': np.nan}
    
    try:
        cm_angle = calculate_cm_angle(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in cm_angle for function f_{fid}_{iid}_{sid}: {e}")
        cm_angle = {'cm_angle': np.nan}
    
    try:
        cm_conv = calculate_cm_conv(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in cm_conv for function f_{fid}_{iid}_{sid}: {e}")
        cm_conv = {'cm_conv': np.nan}
    
    try:
        cm_grad = calculate_cm_grad(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in cm_grad for function f_{fid}_{iid}_{sid}: {e}")
        cm_grad = {'cm_grad': np.nan}
    
    try:
        disp = calculate_dispersion(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in disp for function f_{fid}_{iid}_{sid}: {e}")
        disp = {'disp': np.nan}
    
    try:
        ic = calculate_information_content(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1], seed=1)
    except Exception as e:
        print(f"Error in ic for function f_{fid}_{iid}_{sid}: {e}")
        ic = {'ic': np.nan}
    
    try:
        nbc = calculate_nbc(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in nbc for function f_{fid}_{iid}_{sid}: {e}")
        nbc = {'nbc': np.nan}
    
    try:
        pca = calculate_pca(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in pca for function f_{fid}_{iid}_{sid}: {e}")
        pca = {'pca': np.nan}
    
    try:
        limo = calculate_limo(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in limo for function f_{fid}_{iid}_{sid}: {e}")
        limo = {'limo': np.nan}
    
    try:
        fdc = calculate_fitness_distance_correlation(norm_doe_bbob_total[:, :-1], norm_doe_bbob_total[:, -1])
    except Exception as e:
        print(f"Error in fdc for function f_{fid}_{iid}_{sid}: {e}")
        fdc = {'fdc': np.nan}

    feature_dict.update(ela_meta)
    feature_dict.update(ela_distr)
    feature_dict.update(ela_level)
    feature_dict.update(cm_angle)
    feature_dict.update(cm_conv)
    feature_dict.update(cm_grad)
    feature_dict.update(disp)
    feature_dict.update(ic)
    feature_dict.update(nbc)
    feature_dict.update(pca)
    feature_dict.update(limo)
    feature_dict.update(fdc)

    return feature_dict


def ela_gen(doe, obj_val_bbob, obj_val_smd, num_sample_points, bootstrap_ratio, num_fun_bbob, num_fun_smd, num_inst, num_seed, ul_dim_smd, dim_bbob, ll_dim_smd, lb_bbob, ub_bbob, smd_problem_list):

    tot_bbob_num = num_fun_bbob*num_inst*num_seed
    tot_smd_num = num_fun_smd*num_inst*num_seed
    
    doe_bbob_total = np.zeros((tot_bbob_num, int(bootstrap_ratio * num_sample_points), dim_bbob+1))
    doe_smd_total = np.zeros((tot_smd_num, int(bootstrap_ratio * num_sample_points), dim_bbob+1))
    
    ll_lb_smd = np.zeros((num_fun_smd, ll_dim_smd)) 
    ll_ub_smd = np.zeros((num_fun_smd, ll_dim_smd))
    
    doe_scaled = np.zeros((num_fun_smd, num_sample_points, dim_bbob))
    
    for i, problem in enumerate(smd_problem_list):
        smd_problem = getattr(SMDProblems, problem.lower())
        _, _, ll_lb_smd[i], ll_ub_smd[i] = smd_problem(ul_dim_smd, ll_dim_smd)
        doe_scaled[i] = ((doe - lb_bbob) / (ub_bbob-lb_bbob))*(ll_ub_smd[i] - ll_lb_smd[i])+ll_lb_smd[i]
            
    for i in range(num_fun_bbob*num_inst):
        for j in range(num_seed):
            doe_bbob_total[i*num_seed+j, :, :dim_bbob], doe_bbob_total[i*num_seed+j, :, dim_bbob] = resample(doe, obj_val_bbob[i*num_sample_points:(i+1)*num_sample_points], replace=False, n_samples=int(bootstrap_ratio*num_sample_points), random_state=j+1, stratify=None)

    for i in range(num_fun_smd*num_inst):
        for j in range(num_seed):
            doe_smd_total[i*num_seed+j, :, :dim_bbob], doe_smd_total[i*num_seed+j, :, dim_bbob] = resample(doe_scaled[i//num_inst], obj_val_smd[i*num_sample_points:(i+1)*num_sample_points], replace=False, n_samples=int(bootstrap_ratio*num_sample_points), random_state=j+1, stratify=None)
    
    scaler = MinMaxScaler()
    for i in range(tot_bbob_num):
        doe_bbob_total[i, :, -1] = scaler.fit_transform(doe_bbob_total[i, :, -1].reshape(-1, 1)).flatten()
    
    for i in range(tot_smd_num):
        doe_smd_total[i, :, -1] = scaler.fit_transform(doe_smd_total[i, :, -1].reshape(-1, 1)).flatten()

    data_set_name = 'BBOB'
    ela_features_bbob = Parallel(n_jobs=-1)(delayed(compute_ela_features)(i, doe_bbob_total[i], num_fun_bbob, num_inst, num_seed, dim_bbob, lb_bbob, ub_bbob, data_set_name) for i in range(tot_bbob_num))

    data_set_name = 'SMD'
    ela_features_smd = Parallel(n_jobs=-1)(delayed(compute_ela_features)(i, doe_smd_total[i], num_fun_smd, num_inst, num_seed, ll_dim_smd, ll_lb_smd[i//(num_inst*num_seed)], ll_ub_smd[i//(num_inst*num_seed)], data_set_name) for i in range(tot_smd_num))

    ela_features_bbob = pd.DataFrame(ela_features_bbob)
    ela_features_smd = pd.DataFrame(ela_features_smd)
    
    ela_features_bbob = ela_features_bbob.drop(['function'], axis=1)
    nan_columns = ela_features_bbob.columns[ela_features_bbob.isna().any()].tolist()
    numeric_cols = ela_features_bbob.select_dtypes(include=[np.number])
    inf_columns = numeric_cols.columns[numeric_cols.applymap(np.isinf).any()].tolist() 
    constant_columns = ela_features_bbob.columns[ela_features_bbob.nunique() == 1].tolist()
    runtime_columns = [col for col in ela_features_bbob.columns if 'runtime' in col]
    fun_evals_columns = [col for col in ela_features_bbob.columns if 'fun_evals' in col]
    redundant_columns = list(set(nan_columns) | set(inf_columns) | set(constant_columns) | set(runtime_columns) | set(fun_evals_columns))
    
    sel_ela_features_bbob = ela_features_bbob.drop(redundant_columns, axis=1)
    coeff_vari_columns = sel_ela_features_bbob.columns[np.abs((sel_ela_features_bbob.std()/sel_ela_features_bbob.mean()))*100 < 5].tolist()
    sel_ela_features_bbob = sel_ela_features_bbob.drop(coeff_vari_columns, axis=1)
    reduced_sel_ela_features_bbob = sel_ela_features_bbob.groupby(np.arange(len(sel_ela_features_bbob))//num_seed).mean()    

    sel_ela_features_smd = ela_features_smd.drop(redundant_columns, axis=1)
    sel_ela_features_smd = sel_ela_features_smd.drop(coeff_vari_columns, axis=1)
    reduced_sel_ela_features_smd = sel_ela_features_smd.groupby(np.arange(len(sel_ela_features_smd))//num_seed).mean()
    
    return reduced_sel_ela_features_bbob, reduced_sel_ela_features_smd

    

