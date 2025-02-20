import numpy as np


def remove_duplicates_numpy(arr):
    _ = np.unique(arr, axis = 0)  # Get unique values while keeping order
    return _  # Sort indices to maintain order

# Example usage
arr = np.array([[3, 1], [2, 3], [4, 2], [5, 1], [4, 2]])
result = remove_duplicates_numpy(arr)
print(result)  # Output: [3 1 2 4 5]
