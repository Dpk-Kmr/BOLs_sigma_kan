import numpy as np
import torch

"""
pip install git+https://github.com/KindXiaoming/pykan.git
pip install setuptools sympy tqdm seaborn pyyaml
2277 multkan
                # exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                # exec(f"x.append(x{ii})")
                x.append(sympy.Symbol(f'x_{ii}'))
"""

# Example function: y = sin(x1) + cos(x2)
def target_function(x):
    return torch.sin(x[:, 0]) + torch.cos(x[:, 1])

# # Generate dataset
# np.random.seed(42)
# X = np.random.uniform(-2, 2, (100, 2))  # 100 samples, 2 features
# y = target_function(X)

from kan import KAN
from kan.utils import create_dataset

# Create dataset in the required format
dataset = create_dataset(target_function, n_var=2)


# Initialize the KAN model
model = KAN(width=[2, 3, 1], grid=3, k=3)  # Adjust width and grid as needed

# Train the model
model.fit(dataset, steps=100)

# Prune the model to simplify the expression
model = model.prune()
model.auto_symbolic()

# Extract and print the symbolic expression
print(model.symbolic_formula()[0][0])
# print("Extracted Symbolic Expression:")
# print(symbolic_expression)



import matplotlib.pyplot as plt

# Generate test data
X_test = np.random.uniform(-2, 2, (100, 2))
y_true = target_function(torch.tensor(X_test, dtype=torch.float32))
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

# # Plot true vs. predicted values
# plt.scatter(y_true, y_pred, alpha=0.7)
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.title('True vs. Predicted Values')
# plt.plot([-2, 2], [-2, 2], 'r--')  # Diagonal line
# plt.show()





