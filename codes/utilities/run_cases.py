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
        X_updated = final_data(X, sigma_values, sigma_cuts, other_feats = other_feats)
        uc_results[n_cuts] = {}
        for name, model in models.items():
            uc_results[n_cuts][name] = perform_model_cv(
                model, 
                X_updated, 
                y, 
                scale = "min_max")
    return uc_results

def opt_cut_case1(
        X, y, pop, objs, 
        models = None, 
        sigma_values = None, 
        other_feats = (0, 1), 
        save_location = None, 
        discrete_col_index = 0):
    """

    """
    modified_fronts = get_modified_fronts(objs, discrete_col_index, without_robin_round_ranking=True)
    selected_pop_inds = [modified_fronts[cut_counts][0] for cut_counts in modified_fronts.keys()]
    oc_results = {}
    for i in selected_pop_inds:
        sigma_cuts = pop[i]
        n_cuts = len(sigma_cuts)
        X_updated = final_data(X, sigma_values, sigma_cuts, other_feats = other_feats)
        oc_results[n_cuts] = {}
        for name, model in models.items():
            oc_results[n_cuts][name] = perform_model_cv(
                model, 
                X_updated, 
                y, 
                scale = "min_max")
    return oc_results



