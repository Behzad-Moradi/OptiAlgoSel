import sqlite3
import Utils.pure_smd_problems as smd_problems
import numpy as np
import pandas as pd

SMD_UL_DIM = 2
BBOB_DIM = 10
LL_SMD_DIM = BBOB_DIM
RANDOM_SEED = 1
BBOB_LOWER_BOUND = -5.0
BBOB_UPPER_BOUND = 5.0
NUM_INSTANCES = 300

BBOB_SUITE_PROB_NAMES = ['bbob1', 'bbob2', 'bbob3', 'bbob4', 'bbob5', 'bbob6', 'bbob7', 'bbob8', 'bbob9', 'bbob10', 'bbob11', 'bbob12', 'bbob13', 'bbob14', 'bbob15', 'bbob16', 'bbob17', 'bbob18', 'bbob19', 'bbob20', 'bbob21', 'bbob22', 'bbob23', 'bbob24']
SMD_SUITE_PROB_NAMES = ['smd1', 'smd2', 'smd3', 'smd4', 'smd5', 'smd6', 'smd7', 'smd8']
ELA_FEATURES = [
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

################################################################################################
################################################################################################
def creat_suites_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS suites(
        suite_id INTEGER PRIMARY KEY AUTOINCREMENT,
        suite_name TEXT UNIQUE NOT NULL,
        suite_description TEXT
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_problems_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS problems(
        problem_id INTEGER PRIMARY KEY AUTOINCREMENT,
        suite_id INTEGER NOT NULL,
        problem_name TEXT NOT NULL,
        problem_number INTEGER NOT NULL,
        FOREIGN KEY(suite_id) REFERENCES suites(suite_id)
        )''')
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_problem_configs_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS problem_configs(
        config_id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL,
        problem_dim INTEGER NOT NULL,
        FOREIGN KEY(problem_id) REFERENCES problems(problem_id),
        UNIQUE(problem_id, problem_dim)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_variable_bounds_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS variable_bounds(
        config_id INTEGER NOT NULL,
        variable_idx INTEGER NOT NULL,
        lower_bound REAL NOT NULL,
        upper_bound REAL NOT NULL,
        PRIMARY KEY(config_id, variable_idx),
        FOREIGN KEY(config_id) REFERENCES problem_configs(config_id)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_problem_instances_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS problem_instances(
        instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id INTEGER NOT NULL,
        instance_num INTEGER NOT NULL,
        FOREIGN KEY(config_id) REFERENCES problem_configs(config_id)
        UNIQUE(config_id, instance_num)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_sampling_sets_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS sampling_sets(
        sampling_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_dim INTEGER NOT NULL,
        sampling_method TEXT NOT NULL,
        num_points INTEGER NOT NULL,
        random_seed INTEGER,
        UNIQUE(problem_dim, sampling_method, num_points)
        )''')
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_sampling_points_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS sampling_points(
        point_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sampling_set_id INTEGER NOT NULL,
        point_num INTEGER NOT NULL,
        FOREIGN KEY(sampling_set_id) REFERENCES sampling_sets(sampling_set_id)
        UNIQUE(sampling_set_id, point_num)
        )''')
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_sampling_coordinates_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS sampling_coordinates(
        point_id INTEGER NOT NULL,
        variable_idx INTEGER NOT NULL,
        coordinate REAL,
        PRIMARY KEY(point_id, variable_idx),
        FOREIGN KEY(point_id) REFERENCES sampling_points(point_id)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_objective_values_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS objective_values(
        instance_id INTEGER NOT NULL,
        point_id INTEGER NOT NULL,
        objective_value REAL,
        PRIMARY KEY(instance_id, point_id),
        FOREIGN KEY(instance_id) REFERENCES problem_instances(instance_id),
        FOREIGN KEY(point_id) REFERENCES sampling_points(point_id)
        )''')
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_algorithms_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS algorithms(
        algorithm_id INTEGER PRIMARY KEY AUTOINCREMENT,
        algorithm_name TEXT UNIQUE NOT NULL,
        algorithm_description TEXT
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return 
################################################################################################
def create_performance_metrics_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS performance_metrics(
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT UNIQUE NOT NULL,
        metric_description TEXT
        )''')
        
    conn.commit()
    cur.close()
    conn.close()   
    return
################################################################################################
def create_algorithm_performance_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS algorithm_performance(
        instance_id INTEGER NOT NULL,
        algorithm_id INTEGER NOT NULL,
        metric_id INTEGER NOT NULL,
        metric_value REAL,
        PRIMARY KEY(instance_id, algorithm_id, metric_id),
        FOREIGN KEY(instance_id) REFERENCES problem_instances(instance_id),
        FOREIGN KEY(algorithm_id) REFERENCES algorithms(algorithm_id),
        FOREIGN KEY(metric_id) REFERENCES performance_metrics(metric_id)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_features_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS features(
        feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT UNIQUE NOT NULL,
        feature_description TEXT
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def create_feature_values_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS feature_values(
        instance_id INTEGER NOT NULL,
        feature_id INTEGER NOT NULL,
        feature_type TEXT NOT NULL, -- raw, processed, normalised,
        feature_value REAL,
        PRIMARY KEY(instance_id, feature_id),
        FOREIGN KEY(instance_id) REFERENCES problem_instances(instance_id),
        FOREIGN KEY(feature_id) REFERENCES features(feature_id)
        )''')
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
################################################################################################

def create_tables():
    
    creat_suites_table()
    create_problems_table()
    create_problem_configs_table()
    create_variable_bounds_table()
    create_problem_instances_table()
    create_sampling_sets_table()
    create_sampling_points_table()
    create_sampling_coordinates_table()
    create_objective_values_table()
    create_algorithms_table()
    create_performance_metrics_table()
    create_algorithm_performance_table()
    create_features_table()
    create_feature_values_table()
    
    return

################################################################################################
################################################################################################
################################################################################################

def init_suites_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    suite_names = ['BBOB', 'SMD']
    for name in suite_names:
        cur.execute("INSERT INTO suites (suite_name, suite_description) VALUES (?, ?)", (name, name))
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_problems_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    for num, name in enumerate(BBOB_SUITE_PROB_NAMES):
        cur.execute("SELECT suite_id FROM suites WHERE suite_name = ?", ('BBOB',))
        suite_id = cur.fetchone()[0]
        cur.execute("INSERT INTO problems (suite_id, problem_name, problem_number) VALUES (?, ?, ?)", (suite_id, name, num+1))
        
    for num, name in enumerate(SMD_SUITE_PROB_NAMES):
        cur.execute("SELECT suite_id FROM suites WHERE suite_name = ?", ('SMD',))
        suite_id = cur.fetchone()[0]
        cur.execute("INSERT INTO problems (suite_id, problem_name, problem_number) VALUES (?, ?, ?)", (suite_id, name, num+1))
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_problem_configs_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute("SELECT problem_id FROM problems")
    problem_ids = cur.fetchall()
    
    for (problem_id, ) in problem_ids:
        cur.execute("INSERT INTO problem_configs (problem_id, problem_dim) VALUES (?, ?)", (problem_id, BBOB_DIM))
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_variable_bounds_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()

    cur.execute('''SELECT pc.config_id, pc.problem_dim, p.problem_name, s.suite_name FROM problem_configs pc
                   JOIN problems p ON pc.problem_id = p.problem_id
                   JOIN suites s ON p.suite_id = s.suite_id''')
    
    configs = cur.fetchall()
    
    for (config_id, problem_dim, problem_name, suite_name) in configs:
        if suite_name == 'BBOB':
            for variable_idx in range(1, problem_dim+1):
                cur.execute("INSERT INTO variable_bounds (config_id, variable_idx, lower_bound, upper_bound) VALUES (?, ?, ?, ?)", (config_id, variable_idx, BBOB_LOWER_BOUND, BBOB_UPPER_BOUND))
            
        if suite_name == 'SMD':
            smd_problem = getattr(smd_problems, problem_name.lower())
            _, _, ll_lb, ll_ub = smd_problem(SMD_UL_DIM, problem_dim)
            for lb, ub, variable_idx in zip(ll_lb, ll_ub, range(1, problem_dim+1)):
                cur.execute("INSERT INTO variable_bounds (config_id, variable_idx, lower_bound, upper_bound) VALUES (?, ?, ?, ?)", (config_id, variable_idx, lb, ub))
            
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_problem_instances_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute("SELECT config_id FROM problem_configs")
    config_ids = cur.fetchall()
    
    for (config_id, ) in config_ids:
        for instance_num in range(1, NUM_INSTANCES + 1):
            cur.execute("INSERT INTO problem_instances (config_id, instance_num) VALUES (?, ?)", (config_id, instance_num))
            
    conn.commit()
    cur.close()
    conn.close()
    return   
################################################################################################
def init_sampling_sets_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    num_sampling_points = 250*BBOB_DIM
    
    cur.execute("INSERT INTO sampling_sets (problem_dim, sampling_method, num_points, random_seed) VALUES (?, ?, ?, ?)", (BBOB_DIM, 'Sobol', num_sampling_points, RANDOM_SEED))
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_sampling_points_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute("SELECT sampling_set_id, num_points FROM sampling_sets")
    all_sampling_sets = cur.fetchall()
    
    for sampling_set_id, num_points in all_sampling_sets:
        for point_num in range(1, num_points+1):
            cur.execute("INSERT INTO sampling_points (sampling_set_id, point_num) VALUES (?, ?)", (sampling_set_id, point_num))
                
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_sampling_coordinates_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    doe = np.load(f'Resources/doe_{BBOB_DIM}d.npy')
    
    cur.execute("SELECT point_id FROM sampling_points")
    all_point_ids = cur.fetchall()
    
    for i, (point_id, ) in enumerate(all_point_ids):
        for variable_idx in range(1, BBOB_DIM+1):
            cur.execute("INSERT INTO sampling_coordinates (point_id, variable_idx, coordinate) VALUES (?, ?, ?)", (point_id, variable_idx, doe[i, variable_idx-1]))
   
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_objective_values_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    doe_bbob_total = np.load(f'Resources/doe_bbob_total_{BBOB_DIM}d.npy')
    doe_smd_total = np.load(f'Resources/doe_smd_total_{LL_SMD_DIM}d.npy')
    
    
    
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################   
def init_algorithms_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    algorithm_names = ['CMAES', 'ES', 'GA', 'DE', 'PSO', 'NM', 'PS']
    algorithm_descriptions = ['Covariance Matrix Adaptation Evolution Strategy', 'Evolution Strategy', 'Genetic Algorithm', 'Differential Evolution', 'Particle Swarm Optimization', 'Nelder-Mead', 'Particle Swarm']
    
    for algorithm_name, algorithm_description in zip(algorithm_names, algorithm_descriptions):
        cur.execute("INSERT INTO algorithms (algorithm_name, algorithm_description) VALUES (?, ?)", (algorithm_name, algorithm_description))

    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################    
def init_performance_metrics_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    perfromance_metrics = ['Optimal Solution']
    
    for metric_name in perfromance_metrics:
        cur.execute("INSERT INTO performance_metrics (metric_name, metric_description) VALUES (?, ?)", (metric_name, metric_name))
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_algorithm_performance_table():
    opt_sol_bbob = np.load(f'Resources/optimal_solution_bbob_{BBOB_DIM}d.npy')
    opt_sol_smd = np.load(f'Resources/optimal_solution_smd_{BBOB_DIM}d.npy')
    
    med_opt_sol_bbob = np.median(opt_sol_bbob, axis=1)
    med_opt_sol_smd = np.median(opt_sol_smd, axis=1)
    
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    cur.execute("SELECT algorithm_id FROM algorithms")
    algorithm_ids = cur.fetchall()
    
    cur.execute('''SELECT pi.instance_id, s.suite_name FROM problem_instances pi
                   JOIN problem_configs pc ON pi.config_id = pc.config_id
                   JOIN problems p ON pc.problem_id = p.problem_id
                   JOIN suites s ON p.suite_id = s.suite_id 
                ''')
    instances = cur.fetchall()
    
    cur.execute("SELECT algorithm_id FROM algorithms")
    algorithm_ids = cur.fetchall()
    
    cur.execute("SELECT metric_id FROM performance_metrics")
    metric_ids = cur.fetchall()
    
    for i, (instance_id, suite_name) in enumerate(instances):
        for j, (algorithm_id, ) in enumerate(algorithm_ids):
            for (metric_id, ) in metric_ids:
                if suite_name == 'BBOB' and i<len(BBOB_SUITE_PROB_NAMES)*NUM_INSTANCES:
                    cur.execute("INSERT INTO algorithm_performance (instance_id, algorithm_id, metric_id, metric_value) VALUES (?, ?, ?, ?)", (instance_id, algorithm_id, metric_id, med_opt_sol_bbob[i, j]))
                if suite_name == 'SMD' and i>=len(BBOB_SUITE_PROB_NAMES)*NUM_INSTANCES:
                    cur.execute("INSERT INTO algorithm_performance (instance_id, algorithm_id, metric_id, metric_value) VALUES (?, ?, ?, ?)", (instance_id, algorithm_id, metric_id, med_opt_sol_smd[i-len(BBOB_SUITE_PROB_NAMES)*NUM_INSTANCES, j]))

    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_features_table():
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
    
    for feature_name in ELA_FEATURES:
        cur.execute("INSERT INTO features (feature_name, feature_description) VALUES (?, ?)", (feature_name, feature_name))
        
    conn.commit()
    cur.close()
    conn.close()
    return
################################################################################################
def init_feature_values_table():   
    ela_features_bbob = pd.read_csv(f'Resources/ela_bbob_processed_{BBOB_DIM}d.csv')
    ela_features_smd = pd.read_csv(f'Resources/ela_smd_processed_{BBOB_DIM}d.csv')
  
    conn = sqlite3.connect('DataBase/optialgosel.db')
    cur = conn.cursor()
     
    cur.execute('''SELECT pi.instance_id, s.suite_name FROM problem_instances pi
                   JOIN problem_configs pc ON pi.config_id = pc.config_id
                   JOIN problems p ON pc.problem_id = p.problem_id
                   JOIN suites s ON p.suite_id = s.suite_id 
                ''')
    instances = cur.fetchall()
    
    cur.execute("SELECT feature_id FROM features")
    feature_ids = cur.fetchall()
    
    for i, (instance_id, suite_name) in enumerate(instances):
        for j, (feature_id, ) in enumerate(feature_ids):
            if suite_name == 'BBOB' and i<len(BBOB_SUITE_PROB_NAMES)*NUM_INSTANCES:
                cur.execute("INSERT INTO feature_values (instance_id, feature_id, feature_type, feature_value) VALUES (?, ?, ?, ?)", (instance_id, feature_id, 'processed', ela_features_bbob.iloc[i, j]))
            if suite_name == 'SMD' and i>=len(BBOB_SUITE_PROB_NAMES)*NUM_INSTANCES:
                cur.execute("INSERT INTO feature_values (instance_id, feature_id, feature_type, feature_value) VALUES (?, ?, ?, ?)", (instance_id, feature_id, 'processed', ela_features_smd.iloc[i-len(BBOB_SUITE_PROB_NAMES)*NUM_INSTANCES, j]))
                
    conn.commit()
    cur.close()
    conn.close()
    return
    
################################################################################################
################################################################################################
################################################################################################
def init_tables():
    
    init_suites_table()
    init_problems_table()
    init_problem_configs_table()
    init_variable_bounds_table()
    init_problem_instances_table()
    init_sampling_sets_table()
    init_sampling_points_table()
    init_sampling_coordinates_table()
    init_objective_values_table()
    init_algorithms_table()
    init_performance_metrics_table()
    init_algorithm_performance_table()
    init_features_table()
    init_feature_values_table()

    return


    