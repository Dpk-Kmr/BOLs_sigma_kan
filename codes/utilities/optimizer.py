from imp_funs import *
import numpy as np

class GAoptimizer:
    def __init__(
            self, 
            obj_model, 
            n_objs = 2, 
            xl = -0.019, 
            xu = 0.027, 
            pop_size = 100, 
            init_pop_size = None, 
            min_x_len = 1, 
            max_x_len = 15,
            mut_uniform_range=(-0.01, 0.01), 
            mut_normal_std = 0.005,
            mutation_probability = None,
            addition_probability = None,
            deletion_probability = None,
            deletion_frac = 0.5, # additional parameters to control length of gene after deletion
            addition_frac = 0.5, # additional parameters to control length of gene after addition
            progress = 0
            ):
        self.obj_model = obj_model
        self.n_objs = n_objs
        self.pop = None
        self.xl = xl
        self.xu = xu
        self.pool = None
        self.pop_size = pop_size
        self.min_x_len = min_x_len
        self.max_x_len = max_x_len
        self.mut_uniform_range = mut_uniform_range
        self.mut_normal_std = mut_normal_std
        self.mutation_probability = mutation_probability
        self.addition_probability = addition_probability
        self.deletion_probability = deletion_probability
        self.deletion_frac = deletion_frac
        self.addition_frac = addition_frac
        self.progress = progress
        if init_pop_size is None:
            self.init_pop_size = pop_size
        else:
            self.init_pop_size = init_pop_size

        



    def init_pop(self):
        rand_gene_size = np.random.randint(self.min_x_len, self.max_x_len+1, self.init_pop_size)
        self.pop = []
        for igene_size in rand_gene_size:
            self.pop.append(np.random.uniform(self.xl, self.xu, igene_size*2).reshape(-1, 2))
        return get_sigma_vals(self.pop)


    def mutate_pop(self, population):
        """
        Applies a hybrid mutation approach: 
        - Uses uniform sampling in early stages (progress close to 0).
        - Uses normal (Gaussian) sampling in later stages (progress close to 1).
        
        Args:
        - population (list of np.array): List of candidate solutions (each is a NumPy array).
        - progress (float): Evolution progress (0.0 = start, 1.0 = end).
        - mutation_probability (float or None): Probability of mutating an element (adaptive if None).
        - uniform_range (tuple): Range (min, max) for uniform sampling.
        - normal_std (float): Standard deviation for normal sampling.
        
        Returns:
        - List of mutated candidate solutions.
        """

        mutated_population = []
        for candidate in population:
            # Adaptive mutation probability if not provided
            if self.mutation_probability is None:
                self.mutation_probability = 1 / candidate.shape[0]  # Inverse length of candidate

            # Copy candidate to avoid modifying original
            mutated_candidate = candidate.copy()

            for i in range(candidate.shape[0]):
                for j in range(candidate.shape[1]):
                    if np.random.rand() < self.mutation_probability:  # Apply mutation based on probability
                        
                        # Hybrid mutation: transition from uniform to normal
                        if np.random.rand() < (1 - self.progress):  
                            # Early stage → Uniform sampling
                            mutation = np.random.uniform(self.mut_uniform_range[0], self.mut_uniform_range[1])
                        else:
                            # Late stage → Normal (Gaussian) sampling
                            mutation = np.random.normal(0, self.mut_normal_std)
                        
                        # Apply mutation
                        mutated_candidate[i, j] += mutation

            mutated_population.append(np.unique(mutated_candidate, axis = 0))

        return get_sigma_vals(mutated_population)

    def replace_pop(self, population):
        self.pool = np.vstack(population)
        len_pool = self.pool.shape[0]
        replaced_population = []
        for candidate in population:
            if self.deletion_probability is None:
                self.deletion_probability = 1 / (self.max_x_len+1 -candidate.shape[0])  # inverse of len
            if self.addition_probability is None:
                self.addition_probability = 1 / candidate.shape[0]  # inverse of len
            replaced_candidate = []
            for gene in candidate:
                # non-deletion step
                if np.random.rand() >= self.deletion_probability:
                    replaced_candidate.append(gene)
                # addition step
                if np.random.rand() < self.addition_probability:
                    replaced_candidate.append(self.pool[np.random.randint(0, len_pool)])
            # handle empty replaced candidate
            if len(replaced_candidate) == 0:
                replaced_candidate.append(self.pool[np.random.randint(0, len_pool)])

            # shuffling and trimming step
            replaced_candidate = shuffle_arr(np.array(replaced_candidate))[:self.max_x_len]
            replaced_population.append(replaced_candidate)
        return get_sigma_vals(replaced_population)


    def select_pop(self):
        




    def get_pop(self):
        if self.pop is None:
            self.pop = self.init_pop()
            return self.pop
        else:
            self.pop = self.pop + self.replace_pop(self.mutate_pop(self.pop))
            return self.pop#self.select_pop(self.pop)




if __name__ == "__main__":
    optim = GAoptimizer(
        3, pop_size = 100
    )
    _ = optim.get_pop()

    # print(optim.mutate_pop(optim.init_pop()))
    __ = optim.get_pop()
    print(len(_), len(__))
    # for i, j in zip(_, __):
    #     print(len(i), len(j))