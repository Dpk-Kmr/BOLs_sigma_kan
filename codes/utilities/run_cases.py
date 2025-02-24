from data_clean import *
from imp_funs import *
from optimizer import *
from model_utils import *




def full_data_case1(X, y, models = models, save_location = None):

    result = {}
    for name, model in models.items():
        result[name] = perform_model_cv(
            model, 
            X, 
            y, 
            scale = "min_max")
    return result

def uniform_cut_case1(X, y, models = None, max_cuts = 2, sigma_values = None, other_feats = (0, 1), save_location = None):
    uc_results = {}
    for n_cuts in range(1, max_cuts + 1):
        sigma_cuts = get_uniform_cuts(None, _start = -0.019, _end = 0.027, n_cuts = n_cuts)
        print(sigma_cuts)
        X_updated = final_data(X, sigma_values, sigma_cuts, other_feats = other_feats)
        uc_results[n_cuts] = {}
        for name, model in models.items():
            uc_results[n_cuts][name] = perform_model_cv(
                model, 
                X_updated, 
                y, 
                scale = "min_max")
    return uc_results



all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
X = all_data["vp_data"]["X_train"]
y = all_data["vp_data"]["y_train"]
print(uniform_cut_case1(
    X, 
    y, 
    models = models, 
    max_cuts = 2, 
    sigma_values = all_data["sigma_values"], 
    other_feats = (0, 1), 
    save_location = None))


