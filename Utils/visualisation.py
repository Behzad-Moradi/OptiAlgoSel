import numpy as np
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict


###########################################################################################   
def aggregate_metrics(results):
        aggregated = defaultdict(lambda: defaultdict(list))
        for i_fold, model_dict in results.items():
            for model_name, metrics in model_dict.items():
                for metric_name, value in metrics.items():
                    if metric_name != "Classification Report":
                        aggregated[model_name][metric_name].append(value)
        averaged_metrics = {
            model_name: {
                metric_name: np.mean(values) for metric_name, values in metrics.items()
            }
            for model_name, metrics in aggregated.items()
        }
        return averaged_metrics
###########################################################################################
    
def visualisation(num_fun_bbob, num_fun_smd, src_dir, res_dir):
    
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern"], "text.usetex": True})

    models = {
                "Dummy": OneVsRestClassifier(DummyClassifier()),
                "Random Uniform": OneVsRestClassifier(DummyClassifier(strategy="uniform", random_state=1)),
                "SVM": OneVsRestClassifier(SVC()),
                "XGBoost": OneVsRestClassifier(XGBClassifier(random_state=1)),
                "Random Forest": OneVsRestClassifier(RandomForestClassifier(random_state=1)),
                "KNN": OneVsRestClassifier(KNeighborsClassifier()),
                "MLP": OneVsRestClassifier(MLPClassifier(random_state=1))
            }

    metrics = ['Micro Precision', 'Micro Recall', 'Micro F1']
    metrics_names = ['Precision', 'Recall', 'F1 Score']

    model_names = list(models.keys())

    model_labels = {"Dummy": "Dummy", "Random Uniform": "Random", "SVM": "SVM", "XGBoost": "XGB", "Random Forest": "RF", "KNN": "KNN", "MLP": "MLP"}
 
    with open(f"{res_dir}/mlc_loio_results.pkl", "rb") as f:
        mlc_loio_results = pickle.load(f)
    with open(f"{res_dir}/mlc_lopo_results.pkl", "rb") as f:
        mlc_lopo_results = pickle.load(f)
    with open(f"{res_dir}/mlc_loso_results.pkl", "rb") as f:
        mlc_loso_results = pickle.load(f)
        
    loio_averaged = aggregate_metrics(mlc_loio_results)
    lopo_averaged = aggregate_metrics(mlc_lopo_results)
    loso_averaged = aggregate_metrics(mlc_loso_results)
    
    test_ratio = 0.1
    
    labels_loio = [f'Fold{i+1}' for i in range(int(1/test_ratio))] + ['Average']
    labels_lopo = [f'BBOB{i+1}' for i in range(num_fun_bbob)] + ['Average']
    labels_loso = [f'SMD{i+1}' for i in range(num_fun_smd)] + ['Average']

    scenarios = [
                    ("LOIO", mlc_loio_results, loio_averaged, labels_loio, 'Instance Fold'),
                    ("LOPO", mlc_lopo_results, lopo_averaged, labels_lopo, 'BBOB Problem Class'),
                    ("LOSO", mlc_loso_results, loso_averaged, labels_loso, 'SMD Problem Class')
                ]
    
    solver_colors = {'Dummy': 'tomato', 'Random': 'gray', 'SVM': 'steelblue', 'RF': 'forestgreen', 'XGB': 'gold', 'KNN': 'purple', 'MLP': 'orange'}

    fig, axs = plt.subplots(nrows=len(metrics), ncols=len(scenarios), figsize=(18, 12), sharey=False)
    for row_idx, metric in enumerate(metrics):
        for col_idx, (scenario_name, results, avg_metrics, labels, xlabel) in enumerate(scenarios):
            ax = axs[row_idx, col_idx]
            folds = list(results.keys())
            x = np.arange(len(labels))+1.5
            n_models = len(model_names)
            width = 0.15
            offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
          
            for model_idx, model_name in enumerate(model_names):
                scores = [results[i][model_name][metric] for i in folds]
                scores.append(avg_metrics[model_name][metric])
                ax.bar(x + offsets[model_idx], scores, width, label=model_labels[model_name], color=solver_colors[model_labels[model_name]])
        
            if row_idx == 0:
                ax.set_title(scenario_name, fontsize=16)

            if row_idx == 2:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=90, fontsize=14)
                ax.set_xlabel(xlabel, fontsize=16)
            else:
                ax.set_xticks([])

            if col_idx == 0:
                ax.set_ylabel(metrics_names[row_idx], fontsize=16)
         
            ax.set_ylim(0,1.05)

    handles, labels_legend = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', ncol=7, bbox_to_anchor=(0.5,1.005), title="Machine Learning Models", fontsize=12, title_fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(f'{res_dir}/performance_comparison_all_models.png', dpi=300, bbox_inches='tight')
    plt.show()
        
    return