from joblib import Parallel, delayed
import cocoex
import numpy as np
import json
from Utils.initial_sample_gen import initial_sample_gen
from sklearn.utils import resample
import Utils.pure_smd_problems as SMDProblems
import Utils.lower_level_smd_problems as LLSMDProblems


def evaluate_bbob_problem(i, doe, num_bbob_fun, num_inst, dim_bbob):
    fun_instance = f"instances:1-{num_inst}"
    fun_name = f"function_indices:1-{num_bbob_fun}  dimensions:{dim_bbob}"
    suite = cocoex.Suite("bbob", fun_instance, fun_name)
    problem = suite[i]
    result = np.array([problem(x) for x in doe])
    return i, result

def evaluate_smd_problem(i, problem, doe, num_smd_fun, num_inst, num_sample_points, dim_bbob, ul_dim_smd, ll_dim_smd, lb_bbob, ub_bbob):
    smd_problem = getattr(SMDProblems, problem.lower())
    ul_lb, ul_ub, ll_lb, ll_ub = smd_problem(ul_dim_smd, ll_dim_smd)
    sample_type = 'sobol'
    ul_candidates = initial_sample_gen(dim=ul_dim_smd, n=num_inst, lower_bound=ul_lb, upper_bound=ul_ub, scramble=False, sample_type=sample_type, seed=1)
    ll_smd_problem = getattr(LLSMDProblems, problem.lower())
    doe_scaled = ((doe - lb_bbob) / (ub_bbob-lb_bbob))*(ll_ub - ll_lb)+ll_lb
    
    result = np.array([[ll_smd_problem(ul_x, doe_scaled[j]) for j in range(num_sample_points)] for ul_x in ul_candidates])
    
    return i, result, doe_scaled, ul_lb, ul_ub, ll_lb, ll_ub


def raw_data_gen(num_bbob_fun, num_smd_fun, num_inst, num_sample_points, dim_bbob, ul_dim_smd, ll_dim_smd, ub_bbob, lb_bbob, smd_problem_list):
    
    np.random.seed(1)

    obj_val_bbob = np.zeros(num_bbob_fun*num_inst*num_sample_points)
    obj_val_smd = np.zeros(num_smd_fun*num_inst*num_sample_points)

    sample_type = 'sobol'
    doe = initial_sample_gen(dim=dim_bbob, n=num_sample_points, lower_bound=lb_bbob, upper_bound=ub_bbob, scramble=False, sample_type=sample_type, seed=1)

    results = Parallel(n_jobs=-1)(delayed(evaluate_bbob_problem)(i, doe, num_bbob_fun, num_inst, dim_bbob) for i in range(num_bbob_fun*num_inst))

    for i, result in results:
        obj_val_bbob[i*num_sample_points:(i+1)*num_sample_points] = result
        

    results = Parallel(n_jobs=-1)(delayed(evaluate_smd_problem)(i, problem, doe, num_smd_fun, num_inst, num_sample_points, dim_bbob, ul_dim_smd, ll_dim_smd, lb_bbob, ub_bbob) for i, problem in enumerate(smd_problem_list))
    
    for i, result, doe_scaled, ul_lb, ul_ub, ll_lb, ll_ub in results:
        obj_val_smd[i*num_inst*num_sample_points:(i+1)*num_inst*num_sample_points] = result.reshape(-1)
        
        
    return doe, obj_val_bbob, obj_val_smd
    

