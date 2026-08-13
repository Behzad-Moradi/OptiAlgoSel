import Utils.pure_smd_problems as smd_problems
import numpy as np
import pandas as pd
from Utils.raw_data_gen import raw_data_gen
from Utils.algo_performance_compute import algo_performance_compute
from Utils.ela_gen import ela_gen
from Utils.get_con_db import get_connection

################################################################################################
################################################################################################
def create_suites_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS suites(
                suite_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                suite_name TEXT UNIQUE NOT NULL,
                suite_description TEXT,
                num_fun INTEGER,
                num_inst INTEGER
                )''')

    return
################################################################################################
def create_problems_table():
    
    with get_connection() as conn:    
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS problems(
                problem_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                problem_name TEXT NOT NULL,
                problem_number INTEGER NOT NULL,
                suite_id INTEGER NOT NULL,
                FOREIGN KEY(suite_id) REFERENCES suites(suite_id)
                )''')
  
    return
################################################################################################
def create_problem_configs_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS problem_configs(
                config_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                problem_id INTEGER NOT NULL,
                problem_dim INTEGER NOT NULL,
                FOREIGN KEY(problem_id) REFERENCES problems(problem_id),
                UNIQUE(problem_id, problem_dim)
                )''')
        
    return
################################################################################################
def create_variable_bounds_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS variable_bounds(
                config_id INTEGER NOT NULL,
                variable_idx INTEGER NOT NULL,
                lower_bound DOUBLE PRECISION NOT NULL,
                upper_bound DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(config_id, variable_idx),
                FOREIGN KEY(config_id) REFERENCES problem_configs(config_id)
                )''')
    
    return
################################################################################################
def create_problem_instances_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS problem_instances(
                instance_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                config_id INTEGER NOT NULL,
                instance_num INTEGER NOT NULL,
                FOREIGN KEY(config_id) REFERENCES problem_configs(config_id),
                UNIQUE(config_id, instance_num)
                )''')

    return
################################################################################################
def create_sampling_sets_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS sampling_sets(
                    sampling_set_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    sampling_method TEXT NOT NULL,
                    num_points INTEGER NOT NULL,
                    problem_dim INTEGER NOT NULL,
                    random_seed INTEGER,
                    UNIQUE(problem_dim, sampling_method, num_points)
                    )''')
   
    return
################################################################################################
def create_sampling_points_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS sampling_points(
                point_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                sampling_set_id INTEGER NOT NULL,
                point_num INTEGER NOT NULL,
                FOREIGN KEY(sampling_set_id) REFERENCES sampling_sets(sampling_set_id),
                UNIQUE(sampling_set_id, point_num)
                )''')
        
    return
################################################################################################
def create_sampling_coordinates_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur: 
            cur.execute('''CREATE TABLE IF NOT EXISTS sampling_coordinates(
                point_id INTEGER NOT NULL,
                variable_idx INTEGER NOT NULL,
                coordinate DOUBLE PRECISION,
                PRIMARY KEY(point_id, variable_idx),
                FOREIGN KEY(point_id) REFERENCES sampling_points(point_id)
                )''')
  
    return
################################################################################################
def create_objective_values_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur: 
            cur.execute('''CREATE TABLE IF NOT EXISTS objective_values(
                instance_id INTEGER NOT NULL,
                point_id INTEGER NOT NULL,
                objective_value DOUBLE PRECISION,
                PRIMARY KEY(instance_id, point_id),
                FOREIGN KEY(instance_id) REFERENCES problem_instances(instance_id),
                FOREIGN KEY(point_id) REFERENCES sampling_points(point_id)
                )''')
            
    return
################################################################################################
def create_algorithms_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:    
            cur.execute('''CREATE TABLE IF NOT EXISTS algorithms(
                algorithm_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                algorithm_name TEXT UNIQUE NOT NULL,
                algorithm_description TEXT
                )''')
        
    return 
################################################################################################
def create_performance_metrics_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:    
            cur.execute('''CREATE TABLE IF NOT EXISTS performance_metrics(
                metric_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                metric_name TEXT UNIQUE NOT NULL,
                metric_description TEXT
                )''')
         
    return
################################################################################################
def create_algorithm_performance_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:  
            cur.execute('''CREATE TABLE IF NOT EXISTS algorithm_performance(
                instance_id INTEGER NOT NULL,
                algorithm_id INTEGER NOT NULL,
                metric_id INTEGER NOT NULL,
                metric_value BYTEA,
                PRIMARY KEY(instance_id, algorithm_id, metric_id),
                FOREIGN KEY(instance_id) REFERENCES problem_instances(instance_id),
                FOREIGN KEY(algorithm_id) REFERENCES algorithms(algorithm_id),
                FOREIGN KEY(metric_id) REFERENCES performance_metrics(metric_id)
                )''')
  
    return
################################################################################################
def create_features_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:  
            cur.execute('''CREATE TABLE IF NOT EXISTS features(
                feature_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                feature_name TEXT UNIQUE NOT NULL,
                feature_description TEXT
                )''')
    
    return
################################################################################################
def create_feature_values_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur: 
            cur.execute('''CREATE TABLE IF NOT EXISTS feature_values(
                instance_id INTEGER NOT NULL,
                feature_id INTEGER NOT NULL,
                feature_type TEXT NOT NULL, -- raw, processed, normalised,
                feature_value DOUBLE PRECISION,
                PRIMARY KEY(instance_id, feature_id),
                FOREIGN KEY(instance_id) REFERENCES problem_instances(instance_id),
                FOREIGN KEY(feature_id) REFERENCES features(feature_id)
                )''')
        
    return

################################################################################################
################################################################################################

def create_database_schema_pg():
    
    create_suites_table()
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
   
    print("Database schema created successfully.")
    return

################################################################################################
################################################################################################
################################################################################################

def populate_suites_table(suite_list):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for suite in suite_list:
                cur.execute("INSERT INTO suites (suite_name, suite_description, num_fun, num_inst) VALUES (%s, %s, %s, %s)", (suite['suite_name'], suite['description'], suite['num_fun'], suite['num_inst']))
    
    return
################################################################################################
def populate_problems_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:  
            cur.execute("SELECT suite_id, suite_name, num_fun FROM suites")
            suites = cur.fetchall()
            
            for (suite_id, suite_name, num_fun) in suites:
                for problem_number in range(1, num_fun+1):
                    problem_name = suite_name + str(problem_number)
                    cur.execute("INSERT INTO problems (problem_name, problem_number, suite_id) VALUES (%s, %s, %s)", (problem_name, problem_number, suite_id))
                    
    return
################################################################################################
def populate_problem_configs_table(prob_dim):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT problem_id FROM problems")
            problem_ids = cur.fetchall()
            
            for (problem_id, ) in problem_ids:
                cur.execute("INSERT INTO problem_configs (problem_id, problem_dim) VALUES (%s, %s)", (problem_id, prob_dim))
        
    return
################################################################################################
def populate_variable_bounds_table(ul_dim_smd, lb_bbob, ub_bbob):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''SELECT pc.config_id, pc.problem_dim, p.problem_name, s.suite_name FROM problem_configs pc
                        JOIN problems p ON pc.problem_id = p.problem_id
                        JOIN suites s ON p.suite_id = s.suite_id''')
            
            configs = cur.fetchall()
            
            for (config_id, problem_dim, problem_name, suite_name) in configs:
                if suite_name == 'BBOB':
                    for variable_idx in range(1, problem_dim+1):
                        cur.execute("INSERT INTO variable_bounds (config_id, variable_idx, lower_bound, upper_bound) VALUES (%s, %s, %s, %s)", (config_id, variable_idx, lb_bbob, ub_bbob))
                    
                if suite_name == 'SMD':
                    smd_problem = getattr(smd_problems, problem_name.lower())
                    _, _, ll_lb, ll_ub = smd_problem(ul_dim_smd, problem_dim)
                    for smd_lb, smd_ub, variable_idx in zip(ll_lb, ll_ub, range(1, problem_dim+1)):
                        cur.execute("INSERT INTO variable_bounds (config_id, variable_idx, lower_bound, upper_bound) VALUES (%s, %s, %s, %s)", (config_id, variable_idx, smd_lb, smd_ub))
                
    return
################################################################################################
def populate_problem_instances_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''SELECT pc.config_id, s.num_inst FROM problem_configs pc
                        JOIN problems p ON pc.problem_id = p.problem_id
                        JOIN suites s ON p.suite_id = s.suite_id''')
            configs = cur.fetchall()
            
            for (config_id, num_inst) in configs:
                for instance_num in range(1, num_inst + 1):
                    cur.execute("INSERT INTO problem_instances (config_id, instance_num) VALUES (%s, %s)", (config_id, instance_num))
            
    return   
################################################################################################
def populate_sampling_sets_table(sampling_set_list):
    
    with get_connection() as conn:
        with conn.cursor() as cur:      
            for sampling_set in sampling_set_list:
                cur.execute("INSERT INTO sampling_sets (sampling_method, num_points, problem_dim, random_seed) VALUES (%s, %s, %s, %s)", (sampling_set['sampling_method'], sampling_set['num_points'], sampling_set['problem_dim'], sampling_set['random_seed']))
                
    return
################################################################################################
def populate_sampling_points_table():
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT sampling_set_id, num_points FROM sampling_sets")
            all_sampling_sets = cur.fetchall()
            
            for sampling_set_id, num_points in all_sampling_sets:
                for point_num in range(1, num_points+1):
                    cur.execute("INSERT INTO sampling_points (sampling_set_id, point_num) VALUES (%s, %s)", (sampling_set_id, point_num))
                    
    return
################################################################################################
def populate_doe_tables(ul_dim_smd):    
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT suite_name, num_fun, num_inst FROM suites")
            suites = cur.fetchall()
            
            for (suite_name, num_fun, num_inst) in suites:
                if suite_name == 'BBOB':
                    num_fun_bbob = num_fun
                if suite_name == 'SMD':
                    num_fun_smd = num_fun
                num_inst = num_inst
                
            cur.execute("SELECT num_points, problem_dim FROM sampling_sets")
            (num_sample_points, prob_dim) = cur.fetchone()
            
            cur.execute('''SELECT p.problem_name FROM problems p
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'SMD' ''')
            
            smd_problem_list = []
            
            for (problem_name, ) in cur.fetchall():
                smd_problem_list.append(problem_name)
                
                
            cur.execute('''SELECT vb.lower_bound, vb.upper_bound FROM variable_bounds vb
                        JOIN problem_configs pc ON pc.config_id = vb.config_id
                        JOIN problems p ON p.problem_id = pc.problem_id
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'BBOB' ''')
            
            (lb_bbob, ub_bbob) = cur.fetchone()
            
    
        
            #doe, obj_val_bbob, obj_val_smd = raw_data_gen(num_fun_bbob, num_fun_smd, num_inst, num_sample_points, prob_dim, ul_dim_smd, prob_dim, ub_bbob, lb_bbob, smd_problem_list)

            doe = np.load(f'Resources/doe_{prob_dim}d.npy')
            obj_val_bbob = np.load(f'Resources/obj_val_bbob_total_{prob_dim}d.npy')
            obj_val_smd = np.load(f'Resources/obj_val_smd_total_{prob_dim}d.npy')
            
            populate_sampling_coordinates_table(doe, cur)
            populate_objective_values_table(obj_val_bbob, obj_val_smd, cur)
        
    return

################################################################################################
def populate_sampling_coordinates_table(doe, cur):
   
    cur.execute("SELECT point_id FROM sampling_points")
    all_point_ids = cur.fetchall()
    
    cur.execute("SELECT problem_dim FROM sampling_sets")
    (prob_dim, ) = cur.fetchone()
    
    for i, (point_id, ) in enumerate(all_point_ids):
        for variable_idx in range(1, prob_dim+1):
            cur.execute("INSERT INTO sampling_coordinates (point_id, variable_idx, coordinate) VALUES (%s, %s, %s)", (point_id, variable_idx, float(doe[i, variable_idx-1])))

    return
################################################################################################
def populate_objective_values_table(obj_val_bbob, obj_val_smd, cur):

    cur.execute('''SELECT pi.instance_id FROM problem_instances pi
                JOIN problem_configs pc ON pi.config_id = pc.config_id
                JOIN problems p ON pc.problem_id = p.problem_id
                JOIN suites s ON p.suite_id = s.suite_id
                WHERE s.suite_name = 'BBOB' ''')
    all_instance_ids_bbob = cur.fetchall()

    cur.execute('''SELECT pi.instance_id FROM problem_instances pi
                JOIN problem_configs pc ON pi.config_id = pc.config_id
                JOIN problems p ON pc.problem_id = p.problem_id
                JOIN suites s ON p.suite_id = s.suite_id
                WHERE s.suite_name = 'SMD' ''')
    all_instance_ids_smd = cur.fetchall()
    
    cur.execute("SELECT point_id FROM sampling_points")
    all_point_ids = cur.fetchall()
    
    cur.execute("SELECT num_points FROM sampling_sets")
    (num_sample_points, ) = cur.fetchone()
    
    for i, (instance_id, ) in enumerate(all_instance_ids_bbob):
        for j, (point_id, ) in enumerate(all_point_ids):
            cur.execute("INSERT INTO objective_values (instance_id, point_id, objective_value) VALUES (%s, %s, %s)", (instance_id, point_id, float(obj_val_bbob[i*num_sample_points+j])))
    
    for i, (instance_id, ) in enumerate(all_instance_ids_smd):
        for j, (point_id, ) in enumerate(all_point_ids):
            cur.execute("INSERT INTO objective_values (instance_id, point_id, objective_value) VALUES (%s, %s, %s)", (instance_id, point_id, float(obj_val_smd[i*num_sample_points+j])))

    return
################################################################################################   
def populate_algorithms_table(algorithm_list):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for algorithm in algorithm_list:
                cur.execute("INSERT INTO algorithms (algorithm_name, algorithm_description) VALUES (%s, %s)", (algorithm['algorithm_name'], algorithm['algorithm_description']))
        
    return
################################################################################################    
def populate_performance_metrics_table(per_metric_list):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for metric_name in per_metric_list:
                cur.execute("INSERT INTO performance_metrics (metric_name, metric_description) VALUES (%s, %s)", (metric_name['metric_name'], metric_name['metric_description']))
                
    return
################################################################################################
def populate_algorithm_performance_table(opt_budget, num_run, ul_dim_smd):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT suite_name, num_fun, num_inst FROM suites")
            suites = cur.fetchall()
        
            for (suite_name, num_fun, num_inst) in suites:
                if suite_name == 'BBOB':
                    num_fun_bbob = num_fun
                if suite_name == 'SMD':
                    num_fun_smd = num_fun
                num_inst = num_inst
                
            cur.execute("SELECT num_points, problem_dim FROM sampling_sets")
            (num_sample_points, prob_dim) = cur.fetchone()
            
            cur.execute('''SELECT p.problem_name FROM problems p
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'SMD'
                        ''')
            
            smd_problem_list = []
            
            for (problem_name, ) in cur.fetchall():
                smd_problem_list.append(problem_name)
                
                
            cur.execute('''SELECT vb.lower_bound, vb.upper_bound FROM variable_bounds vb
                        JOIN problem_configs pc ON pc.config_id = vb.config_id
                        JOIN problems p ON p.problem_id = pc.problem_id
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'BBOB'
                        ''')
            
            (lb_bbob, ub_bbob) = cur.fetchone()
            
            
            #opt_sol_bbob, opt_sol_smd = algo_performance_compute(opt_budget, num_fun_bbob, num_fun_smd, num_inst, prob_dim, ul_dim_smd, prob_dim, num_run, smd_problem_list)
            
            opt_sol_bbob = np.load(f'Resources/optimal_solution_bbob_{prob_dim}d.npy')
            opt_sol_smd = np.load(f'Resources/optimal_solution_smd_{prob_dim}d.npy')
            
            
            cur.execute("SELECT algorithm_id FROM algorithms")
            algorithm_ids = cur.fetchall()
            
            cur.execute('''SELECT pi.instance_id FROM problem_instances pi
                        JOIN problem_configs pc ON pi.config_id = pc.config_id
                        JOIN problems p ON pc.problem_id = p.problem_id
                        JOIN suites s ON p.suite_id = s.suite_id
                        WHERE s.suite_name = 'BBOB'
                        ''')
            instance_ids_bbob = cur.fetchall()
            
            cur.execute('''SELECT pi.instance_id FROM problem_instances pi
                        JOIN problem_configs pc ON pi.config_id = pc.config_id
                        JOIN problems p ON pc.problem_id = p.problem_id
                        JOIN suites s ON p.suite_id = s.suite_id
                        WHERE s.suite_name = 'SMD'
                        ''')
            instance_ids_smd = cur.fetchall()
            
            cur.execute("SELECT algorithm_id FROM algorithms")
            algorithm_ids = cur.fetchall()
            
            cur.execute("SELECT metric_id FROM performance_metrics")
            metric_ids = cur.fetchall()

                
            for i, (instance_id,) in enumerate(instance_ids_bbob):
                for j, (algorithm_id, ) in enumerate(algorithm_ids):
                    for (metric_id, ) in metric_ids:
                        cur.execute("INSERT INTO algorithm_performance (instance_id, algorithm_id, metric_id, metric_value) VALUES (%s, %s, %s, %s)", (instance_id, algorithm_id, metric_id, opt_sol_bbob[i, :, j].astype(np.float64).tobytes()))
                                    
            for i, (instance_id,) in enumerate(instance_ids_smd):
                for j, (algorithm_id, ) in enumerate(algorithm_ids):
                    for (metric_id, ) in metric_ids:
                        cur.execute("INSERT INTO algorithm_performance (instance_id, algorithm_id, metric_id, metric_value) VALUES (%s, %s, %s, %s)", (instance_id, algorithm_id, metric_id, opt_sol_smd[i, :, j].astype(np.float64).tobytes()))

    return
################################################################################################
def populate_features_table(feature_list):
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            for feature_name in feature_list:
                cur.execute("INSERT INTO features (feature_name, feature_description) VALUES (%s, %s)", (feature_name, feature_name))
            
    return
################################################################################################
def populate_feature_values_table(ul_dim_smd, num_seed, bootstrap_ratio):   
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT suite_name, num_fun, num_inst FROM suites")
            suites = cur.fetchall()
            
            for (suite_name, num_fun, num_inst) in suites:
                if suite_name == 'BBOB':
                    num_fun_bbob = num_fun
                if suite_name == 'SMD':
                    num_fun_smd = num_fun
                num_inst = num_inst
                
            cur.execute("SELECT num_points, problem_dim FROM sampling_sets")
            (num_sample_points, prob_dim) = cur.fetchone()
            
            cur.execute('''SELECT p.problem_name FROM problems p
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'SMD'
                        ''')
            
            smd_problem_list = []
            
            for (problem_name, ) in cur.fetchall():
                smd_problem_list.append(problem_name)
                    
            cur.execute('''SELECT vb.lower_bound, vb.upper_bound FROM variable_bounds vb
                        JOIN problem_configs pc ON pc.config_id = vb.config_id
                        JOIN problems p ON p.problem_id = pc.problem_id
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'BBOB'
                        ''')
            
            (lb_bbob, ub_bbob) = cur.fetchone()
            
            
            cur.execute("SELECT point_id, variable_idx, coordinate FROM sampling_coordinates ORDER BY point_id, variable_idx")
            
            doe = np.zeros((num_sample_points, prob_dim))
            for (i, j, val) in cur.fetchall():
                doe[i-1, j-1] = val
                
            cur.execute('''SELECT ov.objective_value FROM objective_values ov
                        JOIN problem_instances pi ON pi.instance_id = ov.instance_id
                        JOIN problem_configs pc ON pc.config_id = pi.config_id
                        JOIN problems p ON p.problem_id = pc.problem_id
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'BBOB'
                        ORDER BY ov.instance_id, ov.point_id''')
            
            obj_val_bbob = []
            for (val, ) in cur.fetchall():
                obj_val_bbob.append(val)
            
            obj_val_bbob = np.array(obj_val_bbob)
            
            cur.execute('''SELECT ov.objective_value FROM objective_values ov
                        JOIN problem_instances pi ON pi.instance_id = ov.instance_id
                        JOIN problem_configs pc ON pc.config_id = pi.config_id
                        JOIN problems p ON p.problem_id = pc.problem_id
                        JOIN suites s ON s.suite_id = p.suite_id
                        WHERE s.suite_name = 'SMD'
                        ORDER BY ov.instance_id, ov.point_id''')
            
            obj_val_smd = []
            for (val, ) in cur.fetchall():
                obj_val_smd.append(val)
            
            obj_val_smd = np.array(obj_val_smd)
            
            
            #ela_gen(doe, obj_val_bbob, obj_val_smd, num_sample_points, bootstrap_ratio, num_fun_bbob, num_fun_smd, num_inst, num_seed, ul_dim_smd, prob_dim, prob_dim, lb_bbob, ub_bbob, smd_problem_list)
            
            ela_features_bbob = pd.read_csv(f'Resources/ela_bbob_processed_{prob_dim}d.csv')
            ela_features_smd = pd.read_csv(f'Resources/ela_smd_processed_{prob_dim}d.csv')

            cur.execute('''SELECT pi.instance_id FROM problem_instances pi
                        JOIN problem_configs pc ON pi.config_id = pc.config_id
                        JOIN problems p ON pc.problem_id = p.problem_id
                        JOIN suites s ON p.suite_id = s.suite_id
                        WHERE s.suite_name = 'BBOB'
                        ''')
            instance_ids_bbob = cur.fetchall()
            
            cur.execute('''SELECT pi.instance_id FROM problem_instances pi
                        JOIN problem_configs pc ON pi.config_id = pc.config_id
                        JOIN problems p ON pc.problem_id = p.problem_id
                        JOIN suites s ON p.suite_id = s.suite_id
                        WHERE s.suite_name = 'SMD'
                        ''')
            instance_ids_smd = cur.fetchall()
            
            cur.execute("SELECT feature_id FROM features")
            feature_ids = cur.fetchall()
            
            for i, (instance_id, ) in enumerate(instance_ids_bbob):
                for j, (feature_id, ) in enumerate(feature_ids):
                    cur.execute("INSERT INTO feature_values (instance_id, feature_id, feature_type, feature_value) VALUES (%s, %s, %s, %s)", (instance_id, feature_id, 'processed', ela_features_bbob.iloc[i, j]))
                    
            for i, (instance_id, ) in enumerate(instance_ids_smd):
                for j, (feature_id, ) in enumerate(feature_ids):
                    cur.execute("INSERT INTO feature_values (instance_id, feature_id, feature_type, feature_value) VALUES (%s, %s, %s, %s)", (instance_id, feature_id, 'processed', ela_features_smd.iloc[i, j]))
                    
    return
    
################################################################################################
################################################################################################
################################################################################################
def populate_database_pg(suite_list, prob_dim, ul_dim_smd, lb_bbob, ub_bbob, sampling_set_list, algortihm_list, per_metric_list, opt_budget, num_run, feature_names, num_seed, bootstrap_ratio):
    
    populate_suites_table(suite_list)
    populate_problems_table()
    populate_problem_configs_table(prob_dim)
    populate_variable_bounds_table(ul_dim_smd, lb_bbob, ub_bbob)
    populate_problem_instances_table()
    populate_sampling_sets_table(sampling_set_list)
    populate_sampling_points_table()
    populate_doe_tables(ul_dim_smd)
    populate_algorithms_table(algortihm_list)
    populate_performance_metrics_table(per_metric_list)
    populate_algorithm_performance_table(opt_budget, num_run, ul_dim_smd)
    populate_features_table(feature_names)
    populate_feature_values_table(ul_dim_smd, num_seed, bootstrap_ratio)

    return


    