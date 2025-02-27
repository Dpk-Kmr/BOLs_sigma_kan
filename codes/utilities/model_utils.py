from data_clean import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import ElasticNet, BayesianRidge, SGDRegressor, HuberRegressor, TheilSenRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
# from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
except:
    print("WARNING: catboost is not installed")
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
import numpy as np
# import tensorflow as tf
from sklearn.model_selection import KFold

import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)



# Set global random seed
random_state = 42
np.random.seed(random_state)




def regression_metrics(actual_property, predicted_property):
    """
    Computes various regression performance metrics for model evaluation.
    
    Parameters:
    - actual_property: numpy array of actual property values.
    - predicted_property: numpy array of predicted property values.
    
    Returns:
    - Dictionary containing all computed metrics.
    """

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_property, predicted_property)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(actual_property, predicted_property)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_property - predicted_property) / actual_property)) * 100

    # R2 Score
    r2 = r2_score(actual_property, predicted_property)

    # Adjusted R² Score
    n = len(actual_property)  # Number of samples
    p = 1  # Assuming only one feature (adjust if multiple features)
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    # Mean Bias Deviation (MBD) - Measures systematic bias
    # mbd = np.mean(predicted_property - actual_property)

    # Store results in a dictionary
    results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2 Score": r2,
        "Adjusted R2 Score": adjusted_r2
    }

    return results

def perform_model(
        model, 
        X_train=None, 
        X_val=None, 
        y_train=None, 
        y_val=None, 
        sigma_values=None,
        sigma_cuts=None,
        other_feats=(0, 1),
        scale=None, 
        random_state = 42):
    np.random.seed(random_state)
    # Apply sigma cut filtering if provided
    if sigma_cuts is not None:
        X_train = final_data(X_train, sigma_values, sigma_cuts, other_feats=other_feats)
        X_val = final_data(X_val, sigma_values, sigma_cuts, other_feats=other_feats)

    # Apply scaling if requested
    X_scaler, y_scaler = None, None
    if scale is not None:
        scaled_data = scale_data(X_train=X_train, X_val=X_val, X_test=None, 
                                 y_train=y_train, y_val=y_val, y_test=None, 
                                 method=scale, random_state = random_state)
        
        # Retrieve scaled values
        X_train = scaled_data["X_train"]
        y_train = scaled_data["y_train"]
        X_val = scaled_data["X_val"]
        y_val = scaled_data["y_val"]
        
        # Retrieve scalers
        X_scaler = scaled_data["X_scaler"]
        y_scaler = scaled_data["y_scaler"] 
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train).reshape(y_train.shape)
    y_val_pred = model.predict(X_val).reshape(y_val.shape)

    # Descale predictions if scaling was applied
    if scale is not None and y_scaler is not None:
        y_train = descale_data(X_data=None, y_data=y_train, x_scaler=None, y_scaler=y_scaler)["y_data"]
        y_val = descale_data(X_data=None, y_data=y_val, x_scaler=None, y_scaler=y_scaler)["y_data"]
        y_train_pred = descale_data(X_data=None, y_data=y_train_pred, x_scaler=None, y_scaler=y_scaler)["y_data"]
        y_val_pred = descale_data(X_data=None, y_data=y_val_pred, x_scaler=None, y_scaler=y_scaler)["y_data"]

    # Return performance metrics
    return {
        "train": regression_metrics(y_train, y_train_pred),
        "val": regression_metrics(y_val, y_val_pred)
    }


def perform_model_cv(model, X, y, 
                     sigma_values=None, sigma_cuts=None, other_feats=(0, 1), scale=None, 
                     n_splits = 5, kf_shuffle = True, random_state = 42):
    """
    Perform 5-Fold Cross-Validation on the given model.
    
    Parameters:
    - model: The regression model to evaluate.
    - X: Feature matrix.
    - y: Target variable.
    - sigma_values, sigma_cuts, other_feats, scale: Additional preprocessing options.
    
    Returns:
    - Dictionary containing average performance metrics across 5 folds.
    """
    np.random.seed(random_state)
    results = {
        "train": {"MAE": [], "MSE": [], "RMSE": [], "MAPE": [], "R2 Score": [], "Adjusted R2 Score": []},
        "val": {"MAE": [], "MSE": [], "RMSE": [], "MAPE": [], "R2 Score": [], "Adjusted R2 Score": []}
    }

    kf = KFold(n_splits=n_splits, shuffle=kf_shuffle, random_state=random_state)
    for train_index, val_index in kf.split(X):
        # Splitting data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Train and evaluate model
        fold_result = perform_model(
            model, 
            X_train=X_train, 
            X_val=X_val, 
            y_train=y_train, 
            y_val=y_val, 
            sigma_values=sigma_values, 
            sigma_cuts=sigma_cuts, 
            other_feats=other_feats, 
            scale=scale,
            random_state = random_state
        )
        
        # Store fold results
        for key in results["train"].keys():
            results["train"][key].append(fold_result["train"][key])
            results["val"][key].append(fold_result["val"][key])
    
    # Compute mean and standard deviation for each metric
    final_results = {
        "train": {metric: {"mean": np.mean(values), "std": np.std(values)} for metric, values in results["train"].items()},
        "val": {metric: {"mean": np.mean(values), "std": np.std(values)} for metric, values in results["val"].items()}
    }
    
    return final_results


# # Custom wrapper for Keras Neural Network

# class KerasRegressorWrapper:
#     def __init__(self, input_dim=None, epochs=100, batch_size=32, learning_rate=0.001):
#         self.input_dim = input_dim  # Will be set during fitting
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.model = None  # Will be initialized in fit()

#     def build_model(self):
#         """Dynamically builds the neural network model."""
#         model = Sequential([
#             input(shape=(self.input_dim,)),  # Corrected Input Layer
#             Dense(128, activation='relu'),
#             Dropout(0.2),
#             Dense(64, activation='relu'),
#             Dropout(0.2),
#             Dense(32, activation='relu'),
#             Dense(1, activation='linear')
#         ])
#         model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
#         return model

#     def fit(self, X, y, validation_data=None):
#         """Fits the model, detecting input dimension automatically."""
#         if self.input_dim is None:
#             self.input_dim = X.shape[1]  # Automatically detect input size
#         self.model = self.build_model()
#         self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, verbose=0)
    
#     def predict(self, X):
#         """Predicts using the trained model."""
#         return self.model.predict(X)




models = {
    
    # Linear Models
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Bayesian Ridge": BayesianRidge(),
    "SGD Regressor": SGDRegressor(max_iter=1000, tol=1e-3),
    "Huber Regressor": HuberRegressor(),
    "Theil-Sen Regressor": TheilSenRegressor(),

    # Tree-Based Models
    "Decision Tree": DecisionTreeRegressor(random_state=random_state),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=random_state),

    # Boosting Models
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=random_state),
    # "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=random_state),

    # Distance-Based Models
    "Support Vector Regression": SVR(kernel="rbf"),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),

    # Neural Networks
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=500, random_state=random_state, alpha = 0.001),
    # "Keras Neural Network": KerasRegressorWrapper()  # Automatically detects input size
}
try:
    models["CatBoost"] = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=0)
except:
    None

model_acronyms = {
    "Linear Regression": "Linear",
    "Ridge Regression": "Ridge",
    "Lasso Regression": "Lasso",
    "ElasticNet": "ElasticNet",
    "Bayesian Ridge": "BayesR",
    "SGD Regressor": "SGDR",
    "Huber Regressor": "Huber",
    "Theil-Sen Regressor": "Theil-Sen",
    "Decision Tree": "DT",
    "Extra Trees": "ET",
    "Random Forest": "RF",
    "Gradient Boosting": "GradBoost",
    "Hist Gradient Boosting": "HGBoost",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "AdaBoost": "AdaBoost",
    "Support Vector Regression": "SVR",
    "K-Nearest Neighbors": "KNN",
    "MLP Regressor": "NN",
    "CatBoost": "CatBoost"
} 

# all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
# X = all_data["vp_data"]["X_train"]
# y = all_data["vp_data"]["y_train"]
# for name, model in models.items():
#     print(f"**************************     {name}    ********************************")
#     print(perform_model_cv(
#         model, 
#         X, 
#         y, 
#         scale = "min_max"))
    
# all_data = get_complete_data_without_cut(output_cols = (-1, ))
# X_train = all_data["d_data"]["X_train"]
# X_val = all_data["d_data"]["X_val"]
# y_train = all_data["d_data"]["y_train"]
# y_val = all_data["d_data"]["y_val"]
# for name, model in models.items():
#     print(f"**************************     {name}    ********************************")
#     print(perform_model(
#         model, 
#         X_train = X_train, 
#         X_val = X_val, 
#         y_train = y_train, 
#         y_val = y_val, 
#         scale = "min_max"))
    

class MOO_model:
    def __init__(
            self, 
            base_model = None, 
            X = None, 
            y = None, 
            X_train=None, 
            X_val=None, 
            y_train=None, 
            y_val=None, 
            sigma_values=None,
            other_feats=(0, 1),
            scale=None,
            cv = True, 
            n_splits = 5, 
            kf_shuffle = True,
            random_state = 42, 
            kpi = ["R2 Score", ],
            kpi_data = ["val", ],
            kpi_sign = [-1,]
            ):
        self.base_model = base_model
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.sigma_values = sigma_values
        self.other_feats = other_feats
        self.scale = scale
        self.cv = cv
        self.n_splits = n_splits
        self.kf_shuffle = kf_shuffle
        self.random_state = random_state
        self.kpi = kpi
        self.kpi_data = kpi_data
        self.kpi_sign = kpi_sign

        np.random.seed(self.random_state)

    def get_objs(self, sigma_cuts):
        np.random.seed(self.random_state)
        if self.cv:
            all_kpis = perform_model_cv(
                self.base_model, 
                self.X, 
                self.y, 
                sigma_values=self.sigma_values, 
                sigma_cuts=sigma_cuts, 
                other_feats=self.other_feats, 
                scale=self.scale, 
                n_splits = self.n_splits, 
                kf_shuffle = self.kf_shuffle,
                random_state = self.random_state)
            return [len(sigma_cuts), ]+ [si*all_kpis[di][ki]["mean"] for si, ki, di in zip(self.kpi_sign, self.kpi, self.kpi_data)]
        else:
            all_kpis = perform_model(
                self.base_model, 
                X_train = self.X_train, 
                X_val = self.X_val, 
                y_train = self.y_train, 
                y_val = self.y_val, 
                sigma_values=self.sigma_values, 
                sigma_cuts=sigma_cuts, 
                other_feats=self.other_feats, 
                scale=self.scale, 
                random_state = self.random_state)
            return [len(sigma_cuts), all_kpis[self.kpi_data][self.kpi]]


if __name__ == "__main__":   
    from optimizer import *

    all_data = get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.0)
    X = all_data["vp_data"]["X_train"]
    y = all_data["vp_data"]["y_train"]
    sigma_values = all_data["sigma_values"]
    moo_model = MOO_model(
        base_model = models["XGBoost"], 
        X = X, 
        y = y, 
        X_train=None, 
        X_val=None, 
        y_train=None, 
        y_val=None, 
        sigma_values=sigma_values,
        other_feats=(0, 1),
        scale="min_max",
        cv = True, 
        n_splits = 2, 
        kf_shuffle = True,
        random_state = 42, 
        kpi = ["R2 Score", ],
        kpi_sign = [-1, ],
        kpi_data = ["val", ]
    )

    model = moo_model.get_objs
    optim = GAoptimizer(
        model, pop_size = 100, n_gen = 2, selection="hybrid", 
        min_x_len = 1, 
        max_x_len = 15,
        mut_uniform_range=(-0.01, 0.01), 
        mut_normal_std = 0.005,
        init_pop_size=500
    )
    optim.run()