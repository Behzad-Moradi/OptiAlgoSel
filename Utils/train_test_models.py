import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import defaultdict
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, classification_report, multilabel_confusion_matrix
from scipy.stats import wilcoxon
import sqlite3
from collections import defaultdict
from joblib import dump
import json

################################# Computing Multiclass Labels #################################    
def multi_class_labeling(num_fun_bbob, num_fun_smd, num_inst, opty_portfolio_results_bbob, opty_portfolio_results_smd, ll_solver_list):
    alpha = 0.05
    y_multi_bbob = np.zeros((num_fun_bbob*num_inst, len(ll_solver_list)), dtype=int)
    y_multi_smd = np.zeros((num_fun_smd*num_inst, len(ll_solver_list)), dtype=int)
    
    for p in range(num_fun_bbob*num_inst):
        medians = np.median(opty_portfolio_results_bbob[p], axis=0)
        best_idx = np.argmin(medians)
        best_runs = opty_portfolio_results_bbob[p, :, best_idx]
        y_multi_bbob[p, best_idx] = 1

        for s in range(len(ll_solver_list)):
            if s == best_idx:
                continue
            runs = opty_portfolio_results_bbob[p, :, s]
            if np.array_equal(best_runs, runs):
                pvalue = 1.0
            else:
                _, pvalue = wilcoxon(best_runs, runs)
                if pvalue > alpha:
                    y_multi_bbob[p, s] = 1
                    
    for p in range(num_fun_smd*num_inst):
        medians = np.median(opty_portfolio_results_smd[p], axis=0)
        best_idx = np.argmin(medians)
        best_runs = opty_portfolio_results_smd[p, :, best_idx]
        y_multi_smd[p, best_idx] = 1

        for s in range(len(ll_solver_list)):
            if s == best_idx:
                continue
            runs = opty_portfolio_results_smd[p, :, s]
            if np.array_equal(best_runs, runs):
                pvalue = 1.0
            else:
                _, pvalue = wilcoxon(best_runs, runs)
                if pvalue > alpha:
                    y_multi_smd[p, s] = 1
    
    return y_multi_bbob, y_multi_smd

###########################################################################################   
###########################################################################################   
###########################################################################################

def train_test_models(num_fun_bbob, num_fun_smd, num_inst, dim_bbob, db_dir, src_dir, res_dir, model_dir):
    conn = sqlite3.connect(db_dir)
    cur = conn.cursor()
    
    cur.execute('''SELECT fv.instance_id, fv.feature_value FROM feature_values fv
                   JOIN problem_instances pi ON pi.instance_id = fv.instance_id
                   JOIN problem_configs pc ON pc.config_id = pi.config_id
                   JOIN problems p ON p.problem_id = pc.problem_id
                   JOIN suites s ON s.suite_id = p.suite_id
                   WHERE s.suite_name == 'BBOB'
                   ORDER BY fv.instance_id, fv.feature_id
                   ''')
    
    feature_values_bbob = cur.fetchall()
    
    groups = defaultdict(list)
    for instance_id, value in feature_values_bbob:
        groups[instance_id].append(value)
    instance_ids = sorted(groups.keys())
    reduced_sel_ela_features_bbob = pd.DataFrame([groups[i] for i in instance_ids], dtype=float)

    cur.execute('''SELECT fv.instance_id, fv.feature_value FROM feature_values fv
                   JOIN problem_instances pi ON pi.instance_id = fv.instance_id
                   JOIN problem_configs pc ON pc.config_id = pi.config_id
                   JOIN problems p ON p.problem_id = pc.problem_id
                   JOIN suites s ON s.suite_id = p.suite_id
                   WHERE s.suite_name == 'SMD'
                   ORDER BY fv.instance_id, fv.feature_id
                   ''')
    
    feature_values_smd = cur.fetchall()
    
    groups = defaultdict(list)
    for instance_id, value in feature_values_smd:
        groups[instance_id].append(value)
    instance_ids = sorted(groups.keys())
    reduced_sel_ela_features_smd = pd.DataFrame([groups[i] for i in instance_ids], dtype=float)
    
    conn.commit()
    cur.close()
    conn.close()
    
    #reduced_sel_ela_features_bbob = pd.read_csv(f'{src_dir}/ela_bbob_processed_{dim_bbob}d.csv')
    #reduced_sel_ela_features_smd = pd.read_csv(f'{src_dir}/ela_smd_processed_{dim_bbob}d.csv')
    
    reduced_sel_ela_features_bbob_mean = reduced_sel_ela_features_bbob.mean()
    reduced_sel_ela_features_smd.fillna(reduced_sel_ela_features_bbob_mean, inplace=True)
    
    scaler = MinMaxScaler()
    scaled_reduced_sel_ela_features_bbob = scaler.fit_transform(reduced_sel_ela_features_bbob.values)
    scaled_reduced_sel_ela_features_smd = scaler.transform(reduced_sel_ela_features_smd.values)
    
    y_multi_bbob = np.load(f'{src_dir}/y_multi_bbob_{dim_bbob}d.npy')
    y_multi_smd = np.load(f'{src_dir}/y_multi_smd_{dim_bbob}d.npy')
    
    models = {
                "Dummy": OneVsRestClassifier(DummyClassifier()),
                "Random Uniform": OneVsRestClassifier(DummyClassifier(strategy="uniform", random_state=1)),
                "SVM": OneVsRestClassifier(SVC()),
                "XGBoost": OneVsRestClassifier(XGBClassifier(random_state=1)),
                "Random Forest": OneVsRestClassifier(RandomForestClassifier(random_state=1)),
                "KNN": OneVsRestClassifier(KNeighborsClassifier()),
                "MLP": OneVsRestClassifier(MLPClassifier(random_state=1))
            }

################## Leave One Instance Out ################
    mlc_loio_results = {}

    test_ratio = 0.1
    test_num_inst = int(test_ratio * num_inst)
    all_indices = np.arange(num_fun_bbob * num_inst)
    base = np.arange(num_fun_bbob)*num_inst  
    test_range = np.arange(test_num_inst)
    
    for i in range(int(1/test_ratio)): 
        
        test_indices = (base[:, None]+i*test_num_inst + test_range[None, :]).flatten()
        train_indices = np.setdiff1d(all_indices, test_indices)

        train_X = reduced_sel_ela_features_bbob.iloc[train_indices]
        train_y = y_multi_bbob[train_indices]
        test_X = reduced_sel_ela_features_bbob.iloc[test_indices]
        test_y = y_multi_bbob[test_indices]

        scaler = MinMaxScaler()
        train_X = scaler.fit_transform(train_X.values)
        test_X = scaler.transform(test_X.values)

        mlc_loio_results[i] = {}

        for name, model in models.items():
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)

            result = {
                        "Subset Accuracy": accuracy_score(test_y, y_pred),
                        "Hamming Score": 1 - hamming_loss(test_y, y_pred),
                        "Micro Precision": precision_score(test_y, y_pred, average='micro', zero_division=0),
                        "Micro Recall": recall_score(test_y, y_pred, average='micro', zero_division=0),
                        "Micro F1": f1_score(test_y, y_pred, average='micro', zero_division=0),
                        "Macro F1": f1_score(test_y, y_pred, average='macro', zero_division=0),
                        "Jaccard Score": jaccard_score(test_y, y_pred, average='samples', zero_division=0),
                        "Classification Report": classification_report(test_y, y_pred, output_dict=True, zero_division=0)
                    }
            
            mlc_loio_results[i][name] = result

    with open(f"{res_dir}/mlc_loio_results.pkl", "wb") as f:
        pickle.dump(mlc_loio_results, f)
    
################## Leave One Problem Out ################
    mlc_lopo_results = {}

    for i in range(num_fun_bbob):
        all_indices = np.arange(num_fun_bbob*num_inst)
        test_indices = np.arange(i*num_inst, (i+1)*num_inst)
        train_indices = np.setdiff1d(all_indices, test_indices)

        train_X = reduced_sel_ela_features_bbob.iloc[train_indices]
        train_y = y_multi_bbob[train_indices]
        test_X = reduced_sel_ela_features_bbob.iloc[test_indices]
        test_y = y_multi_bbob[test_indices]

        scaler = MinMaxScaler()
        train_X = scaler.fit_transform(train_X.values)
        test_X = scaler.transform(test_X.values)

        mlc_lopo_results[i] = {}
        
        for name, model in models.items():
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)

            result = {
                        "Subset Accuracy": accuracy_score(test_y, y_pred),
                        "Hamming Score": 1 - hamming_loss(test_y, y_pred),
                        "Micro Precision": precision_score(test_y, y_pred, average='micro', zero_division=0),
                        "Micro Recall": recall_score(test_y, y_pred, average='micro', zero_division=0),
                        "Micro F1": f1_score(test_y, y_pred, average='micro', zero_division=0),
                        "Macro F1": f1_score(test_y, y_pred, average='macro', zero_division=0),
                        "Jaccard Score": jaccard_score(test_y, y_pred, average='samples', zero_division=0),
                        "Classification Report": classification_report(test_y, y_pred, output_dict=True, zero_division=0)
                    }

            mlc_lopo_results[i][name] = result

    with open(f"{res_dir}/mlc_lopo_results.pkl", "wb") as f:
        pickle.dump(mlc_lopo_results, f)

################## Leave One Suite Out ################
    mlc_loso_results = {}

    for i in range(num_fun_smd):
        mlc_loso_results[i] = {}   
        for name, model in models.items():
                model.fit(scaled_reduced_sel_ela_features_bbob, y_multi_bbob)
                y_pred = model.predict(scaled_reduced_sel_ela_features_smd[i*num_inst:(i+1)*num_inst])

                result = {
                            "Subset Accuracy": accuracy_score(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred),
                            "Hamming Score": 1 - hamming_loss(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred),
                            "Micro Precision": precision_score(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred, average='micro', zero_division=0),
                            "Micro Recall": recall_score(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred, average='micro', zero_division=0),
                            "Micro F1": f1_score(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred, average='micro', zero_division=0),
                            "Macro F1": f1_score(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred, average='macro', zero_division=0),
                            "Jaccard Score": jaccard_score(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred, average='samples', zero_division=0),
                            "Classification Report": classification_report(y_multi_smd[i*num_inst:(i+1)*num_inst], y_pred, output_dict=True, zero_division=0)
                        }

                mlc_loso_results[i][name] = result

    with open(f"{res_dir}/mlc_loso_results.pkl", "wb") as f:
        pickle.dump(mlc_loso_results, f) 
        
################## Model Registry ################
    
    registry = []

    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        model.fit(scaled_reduced_sel_ela_features_bbob, y_multi_bbob)
        dump(model, f"{model_dir}/{filename}")
        registry.append({
                            "model_name": name,
                            "version": "1.0.0",
                            "status": "archived",
                            "file": filename
                        })
        
    for model in registry:
        if model["model_name"] == "SVM":
            model["status"] = "production"     
    
    with open(f"{model_dir}/model_registry.json", "w") as f:
        json.dump(registry, f, indent=4)

    
    return mlc_loio_results, mlc_lopo_results, mlc_loso_results