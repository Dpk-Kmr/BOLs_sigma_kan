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
    if_use_model = {mod: 1 for mod in model_names}
    result_dict_ = {}
    for i in cut_nums:
        result_dict_[i] = [result_dict[i][mod][dtype][performance] for mod in model_names]

    result_dict = result_dict_
    
    if filter_models:
        for i in result_dict.keys():
            for mod, perf in zip(model_names, result_dict[i]):
                print(mod, perf["mean"], perf["std"])
                if perf["mean"] - perf["std"] < 0:
                        print("********************", perf["mean"], perf["std"])
                        if_use_model[mod] *= 0


    if use_acronyms:
        model_names = [model_acronyms[i] for i in model_names]
    
                
    result_dict_ = {}
    for i in cut_nums:
        result_dict_[i] = [result_dict[i][j] for j, mod in enumerate(model_names) if if_use_model[mod] == 1]
    result_dict = result_dict_

    model_names = [mod for mod in model_names if if_use_model[mod] == 1]

    

    # Create a single figure with multiple y-axes sharing the same x-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    x = np.arange(len(cut_nums))  # cut indices


    # Define colors for different models
    colors = ["red", "blue", "green", "purple", "yellow", "brown", "silver", "orange", "cyan",
              "pink", "gray", "gold", "lime", "magenta", "navy", "olive", "teal", "maroon"]
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




def two_cut_bars(
        result_dict_uniform,
        result_dict_optimum, 
        filter_models = True, 
        use_acronyms = True, 
        figsize=(14, 7), 
        width = 0.08, 
        gap_fraction = 0.5, save_location = None, dpi = 500, dtype = "val", performance = "R2 Score"):
    cut_nums = list(result_dict_optimum.keys())
    model_names = list(result_dict_optimum[cut_nums[0]].keys())
    if_use_model = {mod: 1 for mod in model_names}
    result_dict_uniform_ = {}
    result_dict_optimum_ = {}
    for i in cut_nums:
        result_dict_uniform_[i] = [result_dict_uniform[i][mod][dtype][performance] for mod in model_names]
        result_dict_optimum_[i] = [result_dict_optimum[i][mod][dtype][performance] for mod in model_names]

    result_dict_uniform = result_dict_uniform_
    result_dict_optimum = result_dict_optimum_

    if filter_models:
        for i in result_dict_optimum.keys():
            for j, mod in enumerate(model_names):
                perf1 = result_dict_uniform[i][j]
                perf2 = result_dict_optimum[i][j]
                if perf1["mean"] - perf1["std"] < 0 or perf2["mean"] - perf2["std"] < 0:
                        if_use_model[mod] *= 0            
    result_dict_uniform_ = {}
    result_dict_optimum_ = {}
    for i in cut_nums:
        result_dict_uniform_[i] = [result_dict_uniform[i][j] for j, mod in enumerate(model_names) if if_use_model[mod] == 1]
        result_dict_optimum_[i] = [result_dict_optimum[i][j] for j, mod in enumerate(model_names) if if_use_model[mod] == 1]
    result_dict_optimum = result_dict_optimum_
    result_dict_uniform = result_dict_uniform_

    model_names = [mod for mod in model_names if if_use_model[mod] == 1]

    if use_acronyms:
        model_names = [model_acronyms[i] for i in model_names]

    # Create a single figure with multiple y-axes sharing the same x-axis
    fig, ax1 = plt.subplots(figsize=figsize)

    x = np.arange(len(cut_nums))  # cut indices


    # Define colors for different models
    colors = ["red", "blue", "green", "purple", "yellow", "brown", "silver", "orange", "cyan",
              "pink", "gray", "gold", "lime", "magenta", "navy", "olive", "teal", "maroon"]
    width = width  # Bar width
    gap = width*gap_fraction
    n_bars = len(model_names)
    start_bar = -(n_bars*width + (n_bars-1)*gap)/2 + width/2
    axes = [ax1]

    ax1.set_xlim(min(x) - n_bars*(width+gap), max(x) + n_bars*(width+gap))

    # Add multiple y-axes for other metrics
    for i, mod in enumerate(model_names):
        mean1_i = [result_dict_uniform[cut_i][i]["mean"] for cut_i in cut_nums]
        std1_i = [result_dict_uniform[cut_i][i]["std"] for cut_i in cut_nums]
        ax1.bar(x + start_bar, mean1_i, width/2, yerr=std1_i, label=f"{mod}", alpha=0.7, facecolor="none", edgecolor=colors[i], hatch='////')
        
        mean2_i = [result_dict_optimum[cut_i][i]["mean"] for cut_i in cut_nums]
        std2_i = [result_dict_optimum[cut_i][i]["std"] for cut_i in cut_nums]
        ax1.bar(x + start_bar + width/2, mean2_i, width/2, yerr=std2_i, label=f"{mod}", alpha=0.7, color=colors[i])

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
i_datas = ["d", "v", "vp"]
mods = ["Linear Regression", "XGBoost", "Gradient Boosting", "MLP Regressor"]

for i_data in i_datas:
    X = all_data[f"{i_data}_data"]["X_train"]
    y = all_data[f"{i_data}_data"]["y_train"]
#     result_dict = full_data_case1(X, y)
#     model_performance_bars(result_dict, save_location = f"results/plots/{i_data}_full.png")
    uni_cut = uniform_cut_case1(
        X, 
        y, 
        models = {
            "Linear Regression": models["Linear Regression"], 
            "XGBoost": models["XGBoost"], 
            "Gradient Boosting": models["Gradient Boosting"],
            "MLP Regressor": models["MLP Regressor"]
            }, 
        max_cuts = 6, 
        sigma_values = all_data["sigma_values"], 
        other_feats = (0, 1), 
        save_location = None)
    

    for mod in mods:
        indicator1 = "R2 Score"
        """
        Change n_gen
        """
        n_gen = 300
        max_x_len = 6
        mod = "Gradient Boosting"
        data_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
        plot_loc = f"results/plots/uniopt_{i_data}_{mod}_{indicator1}.png"


        # data_loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_testing.pkl"
        # plot_loc = None


        opt_pop_objs = load_history(data_loc)[n_gen]

        opt_cut = opt_cut_case1(
                X, y, opt_pop_objs["pop"], opt_pop_objs["obj"], 
                models = {
                "Linear Regression": models["Linear Regression"], 
                "XGBoost": models["XGBoost"], 
                "Gradient Boosting": models["Gradient Boosting"],
                "MLP Regressor": models["MLP Regressor"]
                }, 
                sigma_values = all_data["sigma_values"], 
                other_feats = (0, 1), 
                save_location = None, 
                discrete_col_index = 0)

        two_cut_bars(
                uni_cut, opt_cut,  
                filter_models = False, 
                use_acronyms = True, 
                figsize=(14, 7), 
                width = 0.08, 
                gap_fraction = 0.5, save_location = plot_loc, dpi = 500, dtype = "val", performance = indicator1)



