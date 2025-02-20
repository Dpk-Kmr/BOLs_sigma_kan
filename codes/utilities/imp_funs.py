import numpy as np

def get_sigma_vals(pop):
    sigma_pop = []
    for gene in pop:
        print(gene)
        min_float = (np.min(gene, axis = 1)*1000//1)/1000
        max_float = (np.max(gene, axis = 1)*1000//1 + 1)/1000
        sigma_pop.append(np.vstack((min_float, max_float)).T)
    return sigma_pop

if __name__ == "__main__":
    pop = [
        [[0.010101, 0.01234], [0.25, 0.20167], [0.25, 0.20167]], 
        [[0.001, 0.1234], [0.2, 0.02]], 
        [[0.001, 0.1234], [0.2, 0.02]]
    ]
    print(get_sigma_vals(pop))
