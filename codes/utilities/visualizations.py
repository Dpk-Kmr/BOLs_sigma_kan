import matplotlib.pyplot as plt
import numpy as np
from run_cases import *
from model_utils import *        
from optimizer import *
from imp_funs import *



def model_performance_bars(
        result_dict, 
        filter_models = True, 
        use_acronyms = True, 
        figsize=(14, 7), 
        width = 0.08, 
        gap_fraction = 0.5, save_location = None, dpi = 500, dtype = "val"):
    model_names = list(result_dict.keys())
    result_dict = [result_dict[i] for i in model_names]
    if use_acronyms:
        model_names = [model_acronyms[i] for i in model_names]
    if filter_models:
        
        # Remove models with any negative performance indicator value
        filtered_performance_data = []
        filtered_models = []
        for i, model in enumerate(result_dict):
            valid = True
            for dtype in [dtype, ]:
                for metric in ["MAE", "MSE", "RMSE", "MAPE", "R2 Score", "Adjusted R2 Score"]:
                    if model[dtype][metric]["mean"] < 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                filtered_performance_data.append(model)
                filtered_models.append(model_names[i])
        result_dict = filtered_performance_data
        model_names = filtered_models


    # Create a single figure with multiple y-axes sharing the same x-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    x = np.arange(len(result_dict))  # Model indices


    # model_names = 
    # Define colors for different metrics
    metrics = ['R2 Score', 'Adjusted R2 Score', 'MAPE', 'MAE', 'MSE', 'RMSE']
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    width = width  # Bar width
    gap = width*gap_fraction
    n_bars = len(colors)
    start_bar = -(n_bars*width + (n_bars-1)*gap)/2 + width/2
    axes = [ax1]

    ax1.set_xlim(min(x) - n_bars*(width+gap), max(x) + n_bars*(width+gap))

    # Add multiple y-axes for other metrics
    for i, metric in enumerate(metrics):
        mean_i = [ddi[dtype][metric]["mean"] for ddi in result_dict]
        std_i = [ddi[dtype][metric]["std"] for ddi in result_dict]
        min_y = min(np.array(mean_i) - (np.array(std_i)))
        max_y = max(np.array(mean_i) + (np.array(std_i)))
        range_y = max_y - min_y
        if i == 0:
            ax1.bar(x + start_bar, mean_i, width, yerr=std_i, label=f"{metric}", alpha=0.7, color=colors[i])
            ax1.set_ylabel(metric, color=colors[0])
            if metric in ['R2 Score', 'Adjusted R2 Score']:
                ax1.set_ylim(0, 1)
            else:
                ax1.set_ylim(max(0, min_y - 0.1*range_y), max_y + 0.1*range_y)
        else:
            start_bar = start_bar+width+gap
            ax_new = ax1.twinx()  # Create a new y-axis
            ax_new.spines["right"].set_position(("outward", 50 * (i-1)))  # Offset each axis
            
            ax_new.bar(x + start_bar, mean_i, width, yerr=std_i, label=f"{metric}", alpha=0.7, color=colors[i])
            ax_new.set_ylabel(metric, color=colors[i])
            axes.append(ax_new)
            if metric in ['R2 Score', 'Adjusted R2 Score']:
                ax_new.set_ylim(0, 1)
            else:
                ax_new.set_ylim(max(0, min_y - 0.1*range_y), max_y + 0.1*range_y)

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.set_xlabel("Models")

    # Title and legend
    # ax1.set_title("Comparison of Performance Indicators Across Models")
    fig.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), ncol = 2)

    plt.tight_layout()
    if save_location is not None:
        plt.savefig(save_location, dpi=dpi)

    plt.show()


def cut_bars(
        result_dict, 
        filter_models = True, 
        use_acronyms = True, 
        figsize=(14, 7), 
        width = 0.08, 
        gap_fraction = 0.5, save_location = None, dpi = 500, dtype = "val", performance = "R2 Score"):
    cut_nums = list(result_dict.keys())
    model_names = list(result_dict[cut_nums[0]].keys())
    result_dict_ = {}
    for i in cut_nums:
        result_dict_[i] = [result_dict[i][mod][dtype][performance] for mod in model_names]

    result_dict = result_dict_
    if use_acronyms:
        model_names = [model_acronyms[i] for i in model_names]
    if filter_models:
        
        if_use_model = {mod: 1 for mod in model_names}
        for i in result_dict.keys():
            for mod, perf in zip(model_names, result_dict[i]):
                if perf["mean"] - perf["std"] < 0:
                        if_use_model *= 0
                

    model_names = [mod for mod in model_names if if_use_model[mod] == 1]
    result_dict_ = {}
    for i in cut_nums:
        result_dict_[i] = [result_dict[i][j] for j, mod in enumerate(model_names) if if_use_model[mod] == 1]
    result_dict = result_dict_

    # Create a single figure with multiple y-axes sharing the same x-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    x = np.arange(len(cut_nums))  # cut indices


    # Define colors for different models
    colors = ["red", "blue", "green", "purple", "yellow", "brown", "silver", "orange", "cyan"]
    width = width  # Bar width
    gap = width*gap_fraction
    n_bars = len(model_names)
    start_bar = -(n_bars*width + (n_bars-1)*gap)/2 + width/2
    axes = [ax1]

    ax1.set_xlim(min(x) - n_bars*(width+gap), max(x) + n_bars*(width+gap))

    # Add multiple y-axes for other metrics
    for i, mod in enumerate(model_names):
        mean_i = [result_dict[cut_i][i]["mean"] for cut_i in cut_nums]
        std_i = [result_dict[cut_i][i]["std"] for cut_i in cut_nums]
        ax1.bar(x + start_bar, mean_i, width, yerr=std_i, label=f"{mod}", alpha=0.7, color=colors[i])
        start_bar = start_bar+width+gap

    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(cut_nums)
    ax1.set_xlabel("Number of cuts")
    ax1.set_ylabel(performance)

    # Title and legend
    # ax1.set_title("Comparison of Performance Indicators Across Models")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol = len(model_names))

    plt.tight_layout()
    if save_location is not None:
        plt.savefig(save_location, dpi=dpi)

    plt.show()


    
    






all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
i_data = "vp"
X = all_data[f"{i_data}_data"]["X_train"]
y = all_data[f"{i_data}_data"]["y_train"]
# result_dict = full_data_case1(X, y)
# model_performance_bars(result_dict)

# _ = uniform_cut_case1(
#     X, 
#     y, 
#     models = {
#         "Decision Tree": models["Decision Tree"], 
#         "Random Forest": models["Random Forest"], 
#         "Extra Trees": models["Extra Trees"], 
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#     max_cuts = 6, 
#     sigma_values = all_data["sigma_values"], 
#     other_feats = (0, 1), 
#     save_location = None)
# cut_bars(_)



sigma_values = all_data["sigma_values"]
indicator1 = "R2 Score"
mod = "XGBoost"

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
n_gen = 200
max_x_len = 6
optim = GAoptimizer(
    model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
    min_x_len = 1, 
    max_x_len = max_x_len,
    mut_uniform_range=(-0.01, 0.01), 
    mut_normal_std = 0.005,
    init_pop_size=1000
)
save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
optim.run(save_loc = save_loc, print_res = False)
# print(optim.history)
# opt_pop_objs = load_history(save_loc)[n_gen]
# _ = opt_cut_case1(
#         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
#         models = {
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#         sigma_values = all_data["sigma_values"], 
#         other_feats = (0, 1), 
#         save_location = None, 
#         discrete_col_index = 0)
# print(_)

























all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
i_data = "v"
X = all_data[f"{i_data}_data"]["X_train"]
y = all_data[f"{i_data}_data"]["y_train"]
# result_dict = full_data_case1(X, y)
# model_performance_bars(result_dict)

# _ = uniform_cut_case1(
#     X, 
#     y, 
#     models = {
#         "Decision Tree": models["Decision Tree"], 
#         "Random Forest": models["Random Forest"], 
#         "Extra Trees": models["Extra Trees"], 
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#     max_cuts = 6, 
#     sigma_values = all_data["sigma_values"], 
#     other_feats = (0, 1), 
#     save_location = None)
# cut_bars(_)



sigma_values = all_data["sigma_values"]
indicator1 = "R2 Score"
mod = "XGBoost"

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
n_gen = 200
max_x_len = 6
optim = GAoptimizer(
    model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
    min_x_len = 1, 
    max_x_len = max_x_len,
    mut_uniform_range=(-0.01, 0.01), 
    mut_normal_std = 0.005,
    init_pop_size=1000
)
save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
optim.run(save_loc = save_loc, print_res = False)
# print(optim.history)
# opt_pop_objs = load_history(save_loc)[n_gen]
# _ = opt_cut_case1(
#         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
#         models = {
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#         sigma_values = all_data["sigma_values"], 
#         other_feats = (0, 1), 
#         save_location = None, 
#         discrete_col_index = 0)
# print(_)






























all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
i_data = "d"
X = all_data[f"{i_data}_data"]["X_train"]
y = all_data[f"{i_data}_data"]["y_train"]
# result_dict = full_data_case1(X, y)
# model_performance_bars(result_dict)

# _ = uniform_cut_case1(
#     X, 
#     y, 
#     models = {
#         "Decision Tree": models["Decision Tree"], 
#         "Random Forest": models["Random Forest"], 
#         "Extra Trees": models["Extra Trees"], 
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#     max_cuts = 6, 
#     sigma_values = all_data["sigma_values"], 
#     other_feats = (0, 1), 
#     save_location = None)
# cut_bars(_)



sigma_values = all_data["sigma_values"]
indicator1 = "R2 Score"
mod = "XGBoost"

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
n_gen = 200
max_x_len = 6
optim = GAoptimizer(
    model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
    min_x_len = 1, 
    max_x_len = max_x_len,
    mut_uniform_range=(-0.01, 0.01), 
    mut_normal_std = 0.005,
    init_pop_size=1000
)
save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
optim.run(save_loc = save_loc, print_res = False)
# print(optim.history)
# opt_pop_objs = load_history(save_loc)[n_gen]
# _ = opt_cut_case1(
#         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
#         models = {
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#         sigma_values = all_data["sigma_values"], 
#         other_feats = (0, 1), 
#         save_location = None, 
#         discrete_col_index = 0)
# print(_)




























all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
i_data = "vp"
X = all_data[f"{i_data}_data"]["X_train"]
y = all_data[f"{i_data}_data"]["y_train"]
# result_dict = full_data_case1(X, y)
# model_performance_bars(result_dict)

# _ = uniform_cut_case1(
#     X, 
#     y, 
#     models = {
#         "Decision Tree": models["Decision Tree"], 
#         "Random Forest": models["Random Forest"], 
#         "Extra Trees": models["Extra Trees"], 
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#     max_cuts = 6, 
#     sigma_values = all_data["sigma_values"], 
#     other_feats = (0, 1), 
#     save_location = None)
# cut_bars(_)



sigma_values = all_data["sigma_values"]
indicator1 = "R2 Score"
mod = "MLP Regressor"

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
n_gen = 200
max_x_len = 6
optim = GAoptimizer(
    model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
    min_x_len = 1, 
    max_x_len = max_x_len,
    mut_uniform_range=(-0.01, 0.01), 
    mut_normal_std = 0.005,
    init_pop_size=1000
)
save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
optim.run(save_loc = save_loc, print_res = False)
# print(optim.history)
# opt_pop_objs = load_history(save_loc)[n_gen]
# _ = opt_cut_case1(
#         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
#         models = {
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#         sigma_values = all_data["sigma_values"], 
#         other_feats = (0, 1), 
#         save_location = None, 
#         discrete_col_index = 0)
# print(_)

























all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
i_data = "v"
X = all_data[f"{i_data}_data"]["X_train"]
y = all_data[f"{i_data}_data"]["y_train"]
# result_dict = full_data_case1(X, y)
# model_performance_bars(result_dict)

# _ = uniform_cut_case1(
#     X, 
#     y, 
#     models = {
#         "Decision Tree": models["Decision Tree"], 
#         "Random Forest": models["Random Forest"], 
#         "Extra Trees": models["Extra Trees"], 
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#     max_cuts = 6, 
#     sigma_values = all_data["sigma_values"], 
#     other_feats = (0, 1), 
#     save_location = None)
# cut_bars(_)



sigma_values = all_data["sigma_values"]
indicator1 = "R2 Score"
mod = "MLP Regressor"

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
n_gen = 200
max_x_len = 6
optim = GAoptimizer(
    model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
    min_x_len = 1, 
    max_x_len = max_x_len,
    mut_uniform_range=(-0.01, 0.01), 
    mut_normal_std = 0.005,
    init_pop_size=1000
)
save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
optim.run(save_loc = save_loc, print_res = False)
# print(optim.history)
# opt_pop_objs = load_history(save_loc)[n_gen]
# _ = opt_cut_case1(
#         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
#         models = {
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#         sigma_values = all_data["sigma_values"], 
#         other_feats = (0, 1), 
#         save_location = None, 
#         discrete_col_index = 0)
# print(_)






























all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
i_data = "d"
X = all_data[f"{i_data}_data"]["X_train"]
y = all_data[f"{i_data}_data"]["y_train"]
# result_dict = full_data_case1(X, y)
# model_performance_bars(result_dict)

# _ = uniform_cut_case1(
#     X, 
#     y, 
#     models = {
#         "Decision Tree": models["Decision Tree"], 
#         "Random Forest": models["Random Forest"], 
#         "Extra Trees": models["Extra Trees"], 
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#     max_cuts = 6, 
#     sigma_values = all_data["sigma_values"], 
#     other_feats = (0, 1), 
#     save_location = None)
# cut_bars(_)



sigma_values = all_data["sigma_values"]
indicator1 = "R2 Score"
mod = "MLP Regressor"

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
n_gen = 200
max_x_len = 6
optim = GAoptimizer(
    model, pop_size = 150, n_gen = n_gen, selection="hybrid", 
    min_x_len = 1, 
    max_x_len = max_x_len,
    mut_uniform_range=(-0.01, 0.01), 
    mut_normal_std = 0.005,
    init_pop_size=1000
)
save_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
optim.run(save_loc = save_loc, print_res = False)
# print(optim.history)
# opt_pop_objs = load_history(save_loc)[n_gen]
# _ = opt_cut_case1(
#         X, y, optim.history[2]["pop"], optim.history[2]["obj"], 
#         models = {
#         "XGBoost": models["XGBoost"], 
#         "MLP Regressor": models["MLP Regressor"]
#         }, 
#         sigma_values = all_data["sigma_values"], 
#         other_feats = (0, 1), 
#         save_location = None, 
#         discrete_col_index = 0)
# print(_)