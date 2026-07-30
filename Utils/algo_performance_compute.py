import numpy as np
import cocoex
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import Problem
from joblib import Parallel, delayed
import Utils.pure_smd_problems as SMDProblems
import Utils.lower_level_smd_problems as LLSMDProblems
from Utils.initial_sample_gen import initial_sample_gen


np.random.seed(1)

class Problem(Problem):
    def __init__(self, problem):
        self.problem = problem
        super().__init__(
            n_var=problem.dimension,
            n_obj=1,
            n_constr=0,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [self.problem(x) for x in x]
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["problem"] = None
        return state

def run_optimiser_bbob(i, j, k, num_bbob_fun, num_inst, dim_bbob, opt_budget, optimiser):
    
    fun_instance = "instances:1-"+str(num_inst)
    fun_name = "function_indices:1-"+str(num_bbob_fun)+"  dimensions:"+str(dim_bbob)
    suite = cocoex.Suite("bbob", fun_instance, fun_name)

    problem = Problem(suite[i])

    results = minimize(problem, optimiser, ('n_eval', opt_budget), seed=j+1, save_history=True)

    optx = results.X
    opty = results.F.item()
    y_archive = [h.opt.get("F")[0].item() for h in results.history]
    
    auc = np.trapz(y_archive, [i*(opt_budget/len(y_archive)) for i in range(len(y_archive))])
    
    print (i,j,k)

    return i, j, k, (optx, opty, y_archive, auc)


def run_optimiser_smd(i, j, k, problem, num_smd_fun, num_inst, ul_dim_smd, ll_dim_smd, opt_budget, optimiser):

    problem = Problem(problem)

    results = minimize(problem, optimiser, ('n_eval', opt_budget), seed=j+1, save_history=True)

    optx = results.X
    opty = results.F.item()
    y_archive = [h.opt.get("F")[0].item() for h in results.history]
    
    auc = np.trapz(y_archive, [i*(opt_budget/len(y_archive)) for i in range(len(y_archive))])
    
    print (i,j,k)

    return i, j, k, (optx, opty, y_archive, auc)

    
class FixedXUProblem:
    def __init__(self, smd_problem, xu_fixed, dimension, lower_bounds, upper_bounds, fun_idx, inst_idx):
        self.smd_problem = smd_problem    
        self.xu_fixed = xu_fixed          
        self.dimension = dimension        
        self.lower_bounds = lower_bounds    
        self.upper_bounds = upper_bounds
        self.name = f'SMD_{fun_idx}_{inst_idx}_{dimension}D'

    def __call__(self, xl):
        return self.smd_problem(self.xu_fixed, xl)

    
def algo_performance_compute(opt_budget, num_bbob_fun, num_smd_fun, num_inst, dim_bbob, ul_dim_smd, ll_dim_smd, num_run, smd_problem_list):
      
    num_bbob_problem = num_bbob_fun*num_inst
    num_smd_problem = num_smd_fun*num_inst

    optimiser_dict = {
    'cmaes': CMAES(),
    'es': ES(),
    'ga': GA(),
    'de': DE(),
    'pso': PSO(),
    'nelder': NelderMead(),
    'pattern': PatternSearch()
    }

    optx_results_bbob = np.zeros((num_bbob_problem, num_run, len(optimiser_dict), dim_bbob))
    opty_results_bbob = np.zeros((num_bbob_problem, num_run, len(optimiser_dict)))
    auc_results_bbob = np.zeros((num_bbob_problem, num_run, len(optimiser_dict)))
    y_archive_bbob = np.empty((num_bbob_problem, num_run, len(optimiser_dict)), dtype=object)

    results = Parallel(n_jobs=-1)(delayed(run_optimiser_bbob)(i, j, k, num_bbob_fun, num_inst, dim_bbob, opt_budget, optimiser) for i in range(num_bbob_problem) for j in range(num_run) for k, optimiser in enumerate(optimiser_dict.values()))

    for i, j, k, result in results:
        optx_results_bbob[i,j,k], opty_results_bbob[i,j,k], y_archive_bbob[i, j, k], auc_results_bbob[i,j,k] = result
    
    optx_results_smd = np.zeros((num_smd_problem, num_run, len(optimiser_dict), ll_dim_smd))
    opty_results_smd = np.zeros((num_smd_problem, num_run, len(optimiser_dict)))
    auc_results_smd = np.zeros((num_smd_problem, num_run, len(optimiser_dict)))
    y_archive_smd = np.empty((num_smd_problem, num_run, len(optimiser_dict)), dtype=object)

    smd_suite = []
    for i, problem in enumerate(smd_problem_list):
        smd_problem = getattr(SMDProblems, problem.lower())
        ul_lb, ul_ub, ll_lb, ll_ub = smd_problem(ul_dim_smd, ll_dim_smd)
        sample_type = 'sobol'
        ul_candidates = initial_sample_gen(dim=ul_dim_smd, n=num_inst, lower_bound=ul_lb, upper_bound=ul_ub, scramble=False, sample_type=sample_type, seed=1)
        ll_smd_problem = getattr(LLSMDProblems, problem.lower())
        for j, ul_x in enumerate(ul_candidates):
            smd_suite.append(FixedXUProblem(ll_smd_problem, ul_x, ll_dim_smd, ll_lb, ll_ub, i, j))

    results = Parallel(n_jobs=-1)(delayed(run_optimiser_smd)(i, j, k, problem, num_smd_fun, num_inst, ul_dim_smd, ll_dim_smd, opt_budget, optimiser) for i, problem in enumerate(smd_suite) for j in range(num_run) for k, optimiser in enumerate(optimiser_dict.values()))

    for i, j, k, result in results:
        optx_results_smd[i,j,k], opty_results_smd[i,j,k], y_archive_smd[i, j, k], auc_results_smd[i,j,k] = result

    return opty_results_bbob, opty_results_smd

