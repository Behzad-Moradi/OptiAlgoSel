
from sklearn.preprocessing import MinMaxScaler
from pflacco.classical_ela_features import *
from pflacco.misc_features import *
from pflacco.local_optima_network_features import *
from collections import defaultdict




def compute_ela_features(norm_doe, lower_bound, upper_bound, fid):

    feature_dict = {'function': f'test_{fid}'}

    try:
        ela_meta = calculate_ela_meta(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in ela_meta for test function: {e}")
        ela_meta = {'ela_meta': np.nan}
    
    try:
        ela_distr = calculate_ela_distribution(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in ela_distr for function: {e}")
        ela_distr = {'ela_distr': np.nan}
    
    try:
        ela_level = calculate_ela_level(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in ela_level for function: {e}")
        ela_level = {'ela_level': np.nan}
    
    try:
        cm_angle = calculate_cm_angle(norm_doe[:, :-1], norm_doe[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in cm_angle for function: {e}")
        cm_angle = {'cm_angle': np.nan}
    
    try:
        cm_conv = calculate_cm_conv(norm_doe[:, :-1], norm_doe[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in cm_conv for function: {e}")
        cm_conv = {'cm_conv': np.nan}
    
    try:
        cm_grad = calculate_cm_grad(norm_doe[:, :-1], norm_doe[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in cm_grad for function: {e}")
        cm_grad = {'cm_grad': np.nan}
    
    try:
        disp = calculate_dispersion(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in disp for function: {e}")
        disp = {'disp': np.nan}
    
    try:
        ic = calculate_information_content(norm_doe[:, :-1], norm_doe[:, -1], seed=1)
    except Exception as e:
        print(f"Error in ic for function: {e}")
        ic = {'ic': np.nan}
    
    try:
        nbc = calculate_nbc(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in nbc for function: {e}")
        nbc = {'nbc': np.nan}
    
    try:
        pca = calculate_pca(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in pca for function: {e}")
        pca = {'pca': np.nan}
    
    try:
        limo = calculate_limo(norm_doe[:, :-1], norm_doe[:, -1], lower_bound=lower_bound, upper_bound=upper_bound)
    except Exception as e:
        print(f"Error in limo for function: {e}")
        limo = {'limo': np.nan}
    
    try:
        fdc = calculate_fitness_distance_correlation(norm_doe[:, :-1], norm_doe[:, -1])
    except Exception as e:
        print(f"Error in fdc for function: {e}")
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

################################################################################################
################################################################################################

def extract_features(doe, lb, up, fid, conn):
    
    scaler = MinMaxScaler()
    doe[:, -1] = scaler.fit_transform(doe[:, -1].reshape(-1, 1)).flatten()
    ela_features_test = compute_ela_features(doe, lb, up, fid)
    ela_features_test = pd.DataFrame(ela_features_test, index=[0])

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
    
    cur.execute("SELECT feature_name FROM features ORDER BY feature_id")
    feature_list = [row[0] for row in cur.fetchall()]
    
    cur.close()

    sel_ela_features_test = ela_features_test[feature_list]
    
    groups = defaultdict(list)
    for instance_id, value in feature_values_bbob:
        groups[instance_id].append(value)
    instance_ids = sorted(groups.keys())
    reduced_sel_ela_features_bbob = pd.DataFrame([groups[i] for i in instance_ids], dtype=float)
    
    reduced_sel_ela_features_bbob_mean = reduced_sel_ela_features_bbob.mean()
    sel_ela_features_test.fillna(reduced_sel_ela_features_bbob_mean, inplace=True)

    
    scaler = MinMaxScaler()
    scaled_reduced_sel_ela_features_bbob = scaler.fit_transform(reduced_sel_ela_features_bbob.values)
    scaled_sel_ela_features_test = scaler.transform(sel_ela_features_test.values)

    
    return scaled_sel_ela_features_test