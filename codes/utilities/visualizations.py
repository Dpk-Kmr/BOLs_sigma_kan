import matplotlib.pyplot as plt
import numpy as np
from run_cases import *
from model_utils import *



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

result_dict = full_data_case1()
model_performance_bars(result_dict)