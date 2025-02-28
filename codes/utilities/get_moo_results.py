
import numpy as np
from run_cases import *
from model_utils import *        
from optimizer import *
from imp_funs import *


def get_res(all_data, i_data, mod, max_x_len, indicator1 = "R2 Score", n_gen = 300, pop_size = 200, init_pop_size = 1000):
    all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
    X = all_data[f"{i_data}_data"]["X_train"]
    y = all_data[f"{i_data}_data"]["y_train"]
    sigma_values = all_data["sigma_values"]
    moo_model = MOO_model(
        base_model = models[mod], 
        X = X, 
        y = y, 
        X_train=None, 
        X_val=None, 
        y_train=None, 
        y_val=None, 
        sigma_values=sigma_values,
        other_feats=(0, 1),
        scale="min_max",
        cv = True, 
        n_splits = 5, 
        kf_shuffle = True,
        random_state = 42, 
        kpi = [indicator1, ],
        kpi_sign = [-1, ],
        kpi_data = ["val", ]
    )

    model = moo_model.get_objs
    optim = GAoptimizer(
        model, pop_size = pop_size, n_gen = n_gen, selection="hybrid", 
        min_x_len = 1, 
        max_x_len = max_x_len,
        mut_uniform_range=(-0.01, 0.01), 
        mut_normal_std = 0.005,
        init_pop_size=init_pop_size
    )
    save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
    optim.run(save_loc = save_loc, print_res = False)
"""
Best models for d: linear, ridge, NN, Huber, BayesR
Best models for v: gradboost, xgboost, NN
Best models for vp: gradboost, xgboost, extra tree, NN
"""



all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
case1 = get_res(all_data, "d", "XGBoost", 6)
case2 = get_res(all_data, "v", "XGBoost", 6)
case3 = get_res(all_data, "vp", "XGBoost", 6)
case4 = get_res(all_data, "d", "MLP Regressor", 6)
case5 = get_res(all_data, "v", "MLP Regressor", 6)
case6 = get_res(all_data, "vp", "MLP Regressor", 6)
case7 = get_res(all_data, "d", "Gradient Boosting", 6)
case8 = get_res(all_data, "v", "Gradient Boosting", 6)
case9 = get_res(all_data, "vp", "Gradient Boosting", 6)
case10 = get_res(all_data, "d", "Linear Regression", 6)
case11 = get_res(all_data, "v", "Linear Regression", 6)
case12 = get_res(all_data, "vp", "Linear Regression", 6)










# i_data = "vp"
# X = all_data[f"{i_data}_data"]["X_train"]
# y = all_data[f"{i_data}_data"]["y_train"]
# # result_dict = full_data_case1(X, y)
# # model_performance_bars(result_dict)




# sigma_values = all_data["sigma_values"]
# indicator1 = "R2 Score"
# mod = "XGBoost"

# moo_model = MOO_model(
#     base_model = models[mod], 
#     X = X, 
#     y = y, 
#     X_train=None, 
#     X_val=None, 
#     y_train=None, 
#     y_val=None, 
#     sigma_values=sigma_values,
#     other_feats=(0, 1),
#     scale="min_max",
#     cv = True, 
#     n_splits = 5, 
#     kf_shuffle = True,
#     random_state = 42, 
#     kpi = [indicator1, ],
#     kpi_sign = [-1, ],
#     kpi_data = ["val", ]
# )

# model = moo_model.get_objs
# n_gen = 200
# max_x_len = 6
# optim = GAoptimizer(
#     model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
#     min_x_len = 1, 
#     max_x_len = max_x_len,
#     mut_uniform_range=(-0.01, 0.01), 
#     mut_normal_std = 0.005,
#     init_pop_size=1000
# )
# save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
# optim.run(save_loc = save_loc, print_res = False)


























# all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
# i_data = "v"
# X = all_data[f"{i_data}_data"]["X_train"]
# y = all_data[f"{i_data}_data"]["y_train"]
# # result_dict = full_data_case1(X, y)
# # model_performance_bars(result_dict)

# # _ = uniform_cut_case1(
# #     X, 
# #     y, 
# #     models = {
# #         "Decision Tree": models["Decision Tree"], 
# #         "Random Forest": models["Random Forest"], 
# #         "Extra Trees": models["Extra Trees"], 
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #     max_cuts = 6, 
# #     sigma_values = all_data["sigma_values"], 
# #     other_feats = (0, 1), 
# #     save_location = None)
# # cut_bars(_)



# sigma_values = all_data["sigma_values"]
# indicator1 = "R2 Score"
# mod = "XGBoost"

# moo_model = MOO_model(
#     base_model = models[mod], 
#     X = X, 
#     y = y, 
#     X_train=None, 
#     X_val=None, 
#     y_train=None, 
#     y_val=None, 
#     sigma_values=sigma_values,
#     other_feats=(0, 1),
#     scale="min_max",
#     cv = True, 
#     n_splits = 5, 
#     kf_shuffle = True,
#     random_state = 42, 
#     kpi = [indicator1, ],
#     kpi_sign = [-1, ],
#     kpi_data = ["val", ]
# )

# model = moo_model.get_objs
# n_gen = 200
# max_x_len = 6
# optim = GAoptimizer(
#     model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
#     min_x_len = 1, 
#     max_x_len = max_x_len,
#     mut_uniform_range=(-0.01, 0.01), 
#     mut_normal_std = 0.005,
#     init_pop_size=1000
# )
# save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
# optim.run(save_loc = save_loc, print_res = False)
# # print(optim.history)
# # opt_pop_objs = load_history(save_loc)[n_gen]
# # _ = opt_cut_case1(
# #         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
# #         models = {
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #         sigma_values = all_data["sigma_values"], 
# #         other_feats = (0, 1), 
# #         save_location = None, 
# #         discrete_col_index = 0)
# # print(_)






























# all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
# i_data = "d"
# X = all_data[f"{i_data}_data"]["X_train"]
# y = all_data[f"{i_data}_data"]["y_train"]
# # result_dict = full_data_case1(X, y)
# # model_performance_bars(result_dict)

# # _ = uniform_cut_case1(
# #     X, 
# #     y, 
# #     models = {
# #         "Decision Tree": models["Decision Tree"], 
# #         "Random Forest": models["Random Forest"], 
# #         "Extra Trees": models["Extra Trees"], 
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #     max_cuts = 6, 
# #     sigma_values = all_data["sigma_values"], 
# #     other_feats = (0, 1), 
# #     save_location = None)
# # cut_bars(_)



# sigma_values = all_data["sigma_values"]
# indicator1 = "R2 Score"
# mod = "XGBoost"

# moo_model = MOO_model(
#     base_model = models[mod], 
#     X = X, 
#     y = y, 
#     X_train=None, 
#     X_val=None, 
#     y_train=None, 
#     y_val=None, 
#     sigma_values=sigma_values,
#     other_feats=(0, 1),
#     scale="min_max",
#     cv = True, 
#     n_splits = 5, 
#     kf_shuffle = True,
#     random_state = 42, 
#     kpi = [indicator1, ],
#     kpi_sign = [-1, ],
#     kpi_data = ["val", ]
# )

# model = moo_model.get_objs
# n_gen = 200
# max_x_len = 6
# optim = GAoptimizer(
#     model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
#     min_x_len = 1, 
#     max_x_len = max_x_len,
#     mut_uniform_range=(-0.01, 0.01), 
#     mut_normal_std = 0.005,
#     init_pop_size=1000
# )
# save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
# optim.run(save_loc = save_loc, print_res = False)
# # print(optim.history)
# # opt_pop_objs = load_history(save_loc)[n_gen]
# # _ = opt_cut_case1(
# #         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
# #         models = {
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #         sigma_values = all_data["sigma_values"], 
# #         other_feats = (0, 1), 
# #         save_location = None, 
# #         discrete_col_index = 0)
# # print(_)




























# all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
# i_data = "vp"
# X = all_data[f"{i_data}_data"]["X_train"]
# y = all_data[f"{i_data}_data"]["y_train"]
# # result_dict = full_data_case1(X, y)
# # model_performance_bars(result_dict)

# # _ = uniform_cut_case1(
# #     X, 
# #     y, 
# #     models = {
# #         "Decision Tree": models["Decision Tree"], 
# #         "Random Forest": models["Random Forest"], 
# #         "Extra Trees": models["Extra Trees"], 
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #     max_cuts = 6, 
# #     sigma_values = all_data["sigma_values"], 
# #     other_feats = (0, 1), 
# #     save_location = None)
# # cut_bars(_)



# sigma_values = all_data["sigma_values"]
# indicator1 = "R2 Score"
# mod = "MLP Regressor"

# moo_model = MOO_model(
#     base_model = models[mod], 
#     X = X, 
#     y = y, 
#     X_train=None, 
#     X_val=None, 
#     y_train=None, 
#     y_val=None, 
#     sigma_values=sigma_values,
#     other_feats=(0, 1),
#     scale="min_max",
#     cv = True, 
#     n_splits = 5, 
#     kf_shuffle = True,
#     random_state = 42, 
#     kpi = [indicator1, ],
#     kpi_sign = [-1, ],
#     kpi_data = ["val", ]
# )

# model = moo_model.get_objs
# n_gen = 200
# max_x_len = 6
# optim = GAoptimizer(
#     model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
#     min_x_len = 1, 
#     max_x_len = max_x_len,
#     mut_uniform_range=(-0.01, 0.01), 
#     mut_normal_std = 0.005,
#     init_pop_size=1000
# )
# save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
# optim.run(save_loc = save_loc, print_res = False)
# # print(optim.history)
# # opt_pop_objs = load_history(save_loc)[n_gen]
# # _ = opt_cut_case1(
# #         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
# #         models = {
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #         sigma_values = all_data["sigma_values"], 
# #         other_feats = (0, 1), 
# #         save_location = None, 
# #         discrete_col_index = 0)
# # print(_)

























# all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
# i_data = "v"
# X = all_data[f"{i_data}_data"]["X_train"]
# y = all_data[f"{i_data}_data"]["y_train"]
# # result_dict = full_data_case1(X, y)
# # model_performance_bars(result_dict)

# # _ = uniform_cut_case1(
# #     X, 
# #     y, 
# #     models = {
# #         "Decision Tree": models["Decision Tree"], 
# #         "Random Forest": models["Random Forest"], 
# #         "Extra Trees": models["Extra Trees"], 
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #     max_cuts = 6, 
# #     sigma_values = all_data["sigma_values"], 
# #     other_feats = (0, 1), 
# #     save_location = None)
# # cut_bars(_)



# sigma_values = all_data["sigma_values"]
# indicator1 = "R2 Score"
# mod = "MLP Regressor"

# moo_model = MOO_model(
#     base_model = models[mod], 
#     X = X, 
#     y = y, 
#     X_train=None, 
#     X_val=None, 
#     y_train=None, 
#     y_val=None, 
#     sigma_values=sigma_values,
#     other_feats=(0, 1),
#     scale="min_max",
#     cv = True, 
#     n_splits = 5, 
#     kf_shuffle = True,
#     random_state = 42, 
#     kpi = [indicator1, ],
#     kpi_sign = [-1, ],
#     kpi_data = ["val", ]
# )

# model = moo_model.get_objs
# n_gen = 200
# max_x_len = 6
# optim = GAoptimizer(
#     model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
#     min_x_len = 1, 
#     max_x_len = max_x_len,
#     mut_uniform_range=(-0.01, 0.01), 
#     mut_normal_std = 0.005,
#     init_pop_size=1000
# )
# save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
# optim.run(save_loc = save_loc, print_res = False)
# # print(optim.history)
# # opt_pop_objs = load_history(save_loc)[n_gen]
# # _ = opt_cut_case1(
# #         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
# #         models = {
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #         sigma_values = all_data["sigma_values"], 
# #         other_feats = (0, 1), 
# #         save_location = None, 
# #         discrete_col_index = 0)
# # print(_)






























# all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
# i_data = "d"
# X = all_data[f"{i_data}_data"]["X_train"]
# y = all_data[f"{i_data}_data"]["y_train"]
# # result_dict = full_data_case1(X, y)
# # model_performance_bars(result_dict)

# # _ = uniform_cut_case1(
# #     X, 
# #     y, 
# #     models = {
# #         "Decision Tree": models["Decision Tree"], 
# #         "Random Forest": models["Random Forest"], 
# #         "Extra Trees": models["Extra Trees"], 
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #     max_cuts = 6, 
# #     sigma_values = all_data["sigma_values"], 
# #     other_feats = (0, 1), 
# #     save_location = None)
# # cut_bars(_)



# sigma_values = all_data["sigma_values"]
# indicator1 = "R2 Score"
# mod = "MLP Regressor"

# moo_model = MOO_model(
#     base_model = models[mod], 
#     X = X, 
#     y = y, 
#     X_train=None, 
#     X_val=None, 
#     y_train=None, 
#     y_val=None, 
#     sigma_values=sigma_values,
#     other_feats=(0, 1),
#     scale="min_max",
#     cv = True, 
#     n_splits = 5, 
#     kf_shuffle = True,
#     random_state = 42, 
#     kpi = [indicator1, ],
#     kpi_sign = [-1, ],
#     kpi_data = ["val", ]
# )

# model = moo_model.get_objs
# n_gen = 200
# max_x_len = 6
# optim = GAoptimizer(
#     model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
#     min_x_len = 1, 
#     max_x_len = max_x_len,
#     mut_uniform_range=(-0.01, 0.01), 
#     mut_normal_std = 0.005,
#     init_pop_size=1000
# )
# save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
# optim.run(save_loc = save_loc, print_res = False)
# # print(optim.history)
# # opt_pop_objs = load_history(save_loc)[n_gen]
# # _ = opt_cut_case1(
# #         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
# #         models = {
# #         "XGBoost": models["XGBoost"], 
# #         "MLP Regressor": models["MLP Regressor"]
# #         }, 
# #         sigma_values = all_data["sigma_values"], 
# #         other_feats = (0, 1), 
# #         save_location = None, 
# #         discrete_col_index = 0)
# # print(_)