from Utils.create_database import create_database_schema, populate_database
from Utils.train_test_models import train_test_models
from Utils.visualisation import visualisation
import sqlite3

ELA_FEATURE_NAMES = [
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.intercept",
    "ela_meta.lin_simple.coef.min",
    "ela_meta.lin_simple.coef.max",
    "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
    "ela_distr.skewness",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
    "ela_level.mmce_lda_10",
    "ela_level.mmce_qda_10",
    "ela_level.lda_qda_10",
    "ela_level.mmce_lda_25",
    "ela_level.mmce_qda_25",
    "ela_level.lda_qda_25",
    "ela_level.mmce_lda_50",
    "ela_level.mmce_qda_50",
    "ela_level.lda_qda_50",
    "disp.ratio_mean_02",
    "disp.ratio_mean_05",
    "disp.ratio_mean_10",
    "disp.ratio_median_02",
    "disp.ratio_median_05",
    "disp.ratio_median_10",
    "disp.diff_mean_02",
    "disp.diff_mean_05",
    "disp.diff_mean_10",
    "disp.diff_mean_25",
    "disp.diff_median_02",
    "disp.diff_median_05",
    "disp.diff_median_10",
    "disp.diff_median_25",
    "ic.eps_s",
    "ic.eps_max",
    "ic.m0",
    "nbc.nn_nb.sd_ratio",
    "nbc.nn_nb.cor",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
    "pca.expl_var_PC1.cor_init",
    "fitness_distance.fd_correlation",
    "fitness_distance.fd_cov",
    "fitness_distance.distance_mean",
    "fitness_distance.distance_std",
    "fitness_distance.fitness_mean",
    "fitness_distance.fitness_std",
]    


def main():
    
    NUM_FUN_BBOB = 24
    NUM_FUN_SMD = 8
    NUM_INST = 300
    NUM_SEED = 50
    NUM_SAMPLE_COEFF = 250
    PROB_DIM = 10
    UL_DIM_SMD = 2
    NUM_SAMPLE_POINTS = NUM_SAMPLE_COEFF*PROB_DIM
    OPT_BUDGET = 1000*PROB_DIM
    NUM_RUN = 51
    BOOTSTRAP_RATIO = 0.8
    BBOB_LOWER_BOUND = -5.0
    BBOB_UPPER_BOUND = 5.0
    
    SMD_PROBLEM_LIST = ['SMD1', 'SMD2', 'SMD3', 'SMD4', 'SMD5', 'SMD6', 'SMD7', 'SMD8']
    
    SUITE_LIST = [{'suite_name': 'BBOB', 'description': 'Black-Box Optimisation Benchmarking', 'num_fun': NUM_FUN_BBOB, 'num_inst': NUM_INST}, {'suite_name': 'SMD', 'description': 'Sinha-Malo-Deb', 'num_fun': NUM_FUN_SMD, 'num_inst': NUM_INST}]
    
    SAMPLING_SET_LIST = [{'sampling_method': 'Sobol', 'num_points': NUM_SAMPLE_POINTS, 'problem_dim': PROB_DIM, 'random_seed': 1}]
    
    DATABASE_DIR ='DataBase/optialgosel.db'
    SRC_DIR = 'Resources'
    RES_DIR = 'Results'
    MODEL_DIR = 'TrainedModels'
    
    ALGORITHM_LIST = [{'algorithm_name': 'CMAES', 'algorithm_description': 'Covariance Matrix Adaptation Evolution Strategy'}, {'algorithm_name': 'ES', 'algorithm_description': 'Evolution Strategy'}, {'algorithm_name': 'GA', 'algorithm_description': 'Genetic Algorithm'}, {'algorithm_name': 'DE', 'algorithm_description': 'Differential Evolution'}, {'algorithm_name': 'PSO', 'algorithm_description': 'Particle Swarm Optimization'}, {'algorithm_name': 'NM', 'algorithm_description': 'Nelder-Mead'}, {'algorithm_name': 'PS', 'algorithm_description': 'Particle Swarm'}]
     
    PERFORMANCE_METRIC_LIST = [{'metric_name': 'Optimal Solution', 'metric_description': 'Optimal Solution'}]

    #create_database_schema(DATABASE_DIR)
    #populate_database(SUITE_LIST, PROB_DIM, UL_DIM_SMD, BBOB_LOWER_BOUND, BBOB_UPPER_BOUND, SAMPLING_SET_LIST, ALGORITHM_LIST, PERFORMANCE_METRIC_LIST, OPT_BUDGET, NUM_RUN, ELA_FEATURE_NAMES, NUM_SEED, BOOTSTRAP_RATIO, DATABASE_DIR)
    train_test_models(NUM_FUN_BBOB, NUM_FUN_SMD, NUM_INST, PROB_DIM, DATABASE_DIR, SRC_DIR, RES_DIR, MODEL_DIR)
    #visualisation(NUM_FUN_BBOB, NUM_FUN_SMD, SRC_DIR, RES_DIR) 
    return

if __name__ == "__main__":
    main()