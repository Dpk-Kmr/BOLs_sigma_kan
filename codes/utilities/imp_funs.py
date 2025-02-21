import numpy as np
import random

def get_sigma_vals(pop):
    sigma_pop = []
    for gene in pop:
        min_float = (np.min(gene, axis = 1)*1000//1)/1000
        max_float = (np.max(gene, axis = 1)*1000//1 + 1)/1000
        sigma_pop.append(np.unique(np.vstack((min_float, max_float)).T, axis = 0))
    return sigma_pop

def shuffle_arr(arr):
    return arr[np.random.permutation(arr.shape[0])]#random.sample(arr, len(arr))

if __name__ == "__main__":
    pop = [
        [[0.010101, 0.01234], [0.25, 0.20167], [0.25, 0.20167]], 
        [[0.001, 0.1234], [0.2, 0.02], [0.25, 0.20167]], 
        [[0.001, 0.1234], [0.2, 0.02]]
    ]
    print(shuffle_arr(np.vstack(get_sigma_vals(pop))))
