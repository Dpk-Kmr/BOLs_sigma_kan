import numpy as np
from imp_funs import *


 

# Example usage
# print(np.where(np.array([2, 3, 4])==4)[0][0])
arr = np.array([[3, 1, 4], [2, 3, 2], [4, 2, 3], [5, 1, 1], [4, 2, 1], [6, 2, 3], [2, 4, 5]])
result =  modified_fronts(arr, 0)
print(result)  # Output: [3 1 2 4 5]
# print(np.where((np.array([[2, 3], [3, 4]]) == [3, 4]).all(axis=1))[0][0])