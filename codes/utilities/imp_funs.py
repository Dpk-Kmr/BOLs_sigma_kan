import numpy as np

def get_sigma_vals(pop):
    sigma_pop = []
    for gene in pop:
        min_float = (np.min(gene, axis = 1)*1000//1)/1000
        max_float = (np.max(gene, axis = 1)*1000//1 + 1)/1000
        sigma_pop.append(np.unique(np.vstack((min_float, max_float)).T, axis = 0))
    return sigma_pop

def shuffle_arr(arr):
    return arr[np.random.permutation(arr.shape[0])]


def if_dominates(point1, point2):
    """
    Checks if point1 dominates point2 in a minimization problem.
    """
    return np.all(point1 <= point2) and np.any(point1 < point2)

def get_pareto_fronts(points):
    """
    Computes the Pareto fronts of a given set of points.
    
    A Pareto front is a set of non-dominated points. The function finds all
    Pareto fronts in a given dataset.

    Parameters:
        points (array-like): A 2D array where each row represents a solution and 
                             each column represents an objective.

    Returns:
        list: A list of Pareto fronts, where each front is a list of indices of points.
    """
    points = np.array(points)
    if points.shape[1] == 1: # for single objective case
        return [list(np.argsort(points[:,0])),]
    num_points = len(points)
    dominated_counts = np.zeros(num_points, dtype=int)
    dominates_list = [[] for _ in range(num_points)]
    
    # Compute dominance relationships
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                if if_dominates(points[i], points[j]):
                    dominates_list[i].append(j)
                elif if_dominates(points[j], points[i]):
                    dominated_counts[i] += 1

    # Identify Pareto fronts
    pareto_fronts = []
    remaining_points = set(range(num_points))

    while remaining_points:
        front = [i for i in remaining_points if dominated_counts[i] == 0]
        if not front:
            break  # No more non-dominated points

        pareto_fronts.append(front)

        # Remove points in the current front from further consideration
        for i in front:
            for j in dominates_list[i]:
                dominated_counts[j] -= 1 # decreases the dominated_counts by number of elements in front
            remaining_points.remove(i)

    return pareto_fronts

def round_robin_ranking(grouped_data, objs_arr):
    """
    Assigns ranks in a round-robin manner across different groups.
    Picks the best (lowest Pareto rank) element from each group iteratively.
    """
    remaining_groups = {k: list(v) for k, v in grouped_data.items()}  # Convert to lists for popping
    ranked_list = []
    while any(remaining_groups.values()):  # While there are elements in any group
        ranked_list.append([])
        for key in sorted(remaining_groups.keys()):  # Ensure consistent ordering
            if remaining_groups[key]:  # If the group still has elements
                ranked_list[-1].append(np.where((objs_arr == remaining_groups[key].pop(0)).all(axis = 1))[0][0].item())  # Assign rank and pick first element
    return ranked_list



def get_modified_fronts(arr, discrete_col_index):
    """
    
    """
    arr = np.array(arr)
    unique_values = np.unique(arr[:, discrete_col_index])
    grouped_data = {}

    for value in unique_values:
        subset = arr[arr[:, discrete_col_index] == value]
        objectives = np.delete(subset, discrete_col_index, axis=1)  # Remove discrete column for sorting
        pareto_fronts = get_pareto_fronts(objectives)
        sorted_indices = np.array(pareto_fronts, dtype=object).flatten().tolist()
        grouped_data[value] = subset[sorted_indices]

    return round_robin_ranking(grouped_data, arr)


def flatten_recursive(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_recursive(item))  # Recursively flatten
        else:
            flat_list.append(item)
    return flat_list


def front_merger(front1, front2):
    if len(front1) != len(front2):
        raise ValueError("both fronts should be equal")
    merged_front = []
    seen = set()

    for a, b in zip(front1, front2):
        if a not in seen:
            merged_front.append(a)
            seen.add(a)
        if b not in seen:
            merged_front.append(b)
            seen.add(b)
        if len(merged_front) == len(front1):
            break
    return merged_front

if __name__ == "__main__":
    points = np.array([[3, 1, 4], [2, 3, 2], [4, 2, 3], [5, 1, 1], [4, 2, 1], [6, 2, 3], [2, 4, 5]])
    points = np.array([[3, 1], [2, 3], [4, 2], [5, 1], [4, 1], [6, 3], [2, 5]])

    pareto_fronts = get_pareto_fronts(points)
    print(pareto_fronts)  # Expected output: [[3], [0, 1], [2]]
    modified_fronts = get_modified_fronts(points, 0)
    print(modified_fronts)
    print(flatten_recursive(modified_fronts))

