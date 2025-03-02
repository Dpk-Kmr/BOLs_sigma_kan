import pickle
import numpy as np
from imp_funs import *


i_data = "d"
mod = "Linear Regression"
indicator1 = "R2 Score"
n_gen = 300
max_x_len = 6
loc = f"results/data/{i_data}_{mod}_{indicator1}_{max_x_len}_.pkl"
_ = load_history(loc)
objs = _[300]["obj"]
print()
modified_fronts = get_modified_fronts(objs, 0, without_robin_round_ranking=True)
selected_pop_inds = [modified_fronts[cut_counts][0] for cut_counts in modified_fronts.keys()]
print(modified_fronts)
print(selected_pop_inds)
print([[len(i), j[0]] for i, j in zip(_[1]["pop"], _[1]["obj"])])
