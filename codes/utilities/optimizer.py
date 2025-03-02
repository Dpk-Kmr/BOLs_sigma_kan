from imp_funs import *
import numpy as np
import pickle

class GAoptimizer:
    def __init__(
            self, 
            obj_model, 
            n_objs = 2, 
            xl = -0.019, 
            xu = 0.027, 
            pop_size = 100, 
            n_gen = 100,
            init_pop_size = None, 
            min_x_len = 1, 
            max_x_len = 6,
            mut_uniform_range=(-0.01, 0.01), 
            mut_normal_std = 0.005,
            mutation_probability = None,
            addition_probability = None,
            deletion_probability = None,
            deletion_frac = 0.5, # additional parameters to control length of gene after deletion
            addition_frac = 0.5, # additional parameters to control length of gene after addition
            progress = 0, 
            selection = "modified",
            random_state=42
            ):
        self.obj_model = obj_model
        self.n_objs = n_objs
        self.pop = None
        self.xl = xl
        self.xu = xu
        self.pool = None
        self.pop_size = pop_size
        self.n_gen = n_gen
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
        self.selection = selection
        self.random_state = random_state
        


        if init_pop_size is None:
            self.init_pop_size = pop_size
        else:
            self.init_pop_size = init_pop_size

        np.random.seed(self.random_state)  # Set global random seed
        self.pop = self.init_pop()
        self.opt_objs = [self.obj_model(cand) for cand in self.pop]
        self.current_gen = 0
        self.history = {self.current_gen: {"pop": self.pop, "obj": self.opt_objs}}



    def init_pop(self):
        np.random.seed(self.random_state)
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
        np.random.seed(self.random_state)  # Ensuring reproducibility
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
            
            mutated_candidate = np.clip(mutated_candidate, self.xl, self.xu)
            # clip mutated_population to keep in bounds
            mutated_population.append(np.unique(mutated_candidate, axis = 0))
        
        return get_sigma_vals(mutated_population)

    def replace_pop(self, population):
        np.random.seed(self.random_state)  # Ensuring reproducibility
        self.pool = np.vstack(self.pop)
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
            replaced_candidate = shuffle_arr(np.unique(np.array(replaced_candidate), axis = 0), random_state = self.random_state)[:self.max_x_len]
            replaced_population.append(replaced_candidate)
        return get_sigma_vals(replaced_population)


    def select_pop(self):
        
        # print("before selection______________________________________")
        # print([[len(i) == j[0]] for i, j in zip(self.pop, self.opt_objs)])
        if self.selection == "pareto":
            flat_fronts = flatten_recursive(get_pareto_fronts(self.opt_objs))[:self.pop_size]
        elif self.selection == "modified":
            flat_fronts = flatten_recursive(get_modified_fronts(self.opt_objs, 0))[:self.pop_size]
        elif self.selection == "hybrid":
            p_fronts = flatten_recursive(get_pareto_fronts(self.opt_objs))[:self.pop_size]
            m_fronts = flatten_recursive(get_modified_fronts(self.opt_objs, 0))[:self.pop_size]
            flat_fronts = front_merger(p_fronts, m_fronts)
        else:
            raise ValueError("check and change selection criteria")
        self.opt_objs = [self.opt_objs[i] for i in flat_fronts]
        self.pop = [self.pop[i] for i in flat_fronts]
        # print("after selection______________________________________")
        # print([[len(i) == j[0]] for i, j in zip(self.pop, self.opt_objs)])
        return self.pop


    def get_next_pop(self):
        new_pop = self.replace_pop(self.mutate_pop(self.pop))
        self.pop = self.pop + new_pop
        self.opt_objs = self.opt_objs + [self.obj_model(cand) for cand in new_pop]
        return self.select_pop()

    def run(self, save_loc = None, print_res = True):
        for i in range(self.n_gen):
            self.get_next_pop()
            self.current_gen += 1
            self.history[self.current_gen] = {"pop": self.pop, "obj": self.opt_objs}
            self.progress += 1/self.n_gen
            print(f"Progress: {self.progress} ------->------->------>------>------->------>------>")
            if print_res: 
                print(self.opt_objs)
        # Save history to file
        if save_loc:
            with open(save_loc, "wb") as f:
                pickle.dump(self.history, f)
            print(f"History saved to {save_loc}")


if __name__ == "__main__":
    def model(x):
        return [int(np.sum(x[:,0:1]).item()), np.sum(x[:,1:2]**2).item()]
    optim = GAoptimizer(
        model, pop_size = 8, xl = -2, xu = 2, n_gen = 10, selection="hybrid", 
        min_x_len = 1, 
        max_x_len = 4,
        mut_uniform_range=(-1.01, 1.01), 
        mut_normal_std = 0.505

    )
    optim.run()
    print(optim.history)
