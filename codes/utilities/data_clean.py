import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def mix_sigma(series_list, weights):
    final_sigma = pd.Series(np.zeros(len(series_list[0])), index=series_list[0].index)
    for i in range(len(series_list)):
        final_sigma += series_list[i] * weights[i]
    return final_sigma

def trapezoidal_area(arr, x_gap=0.001):
    area = np.zeros((arr.shape[0], 1))
    for i in range(arr.shape[1] - 1):
        area += (arr[:,(i,)] + arr[:,(i + 1,)]) / 2 * x_gap
    return area


def load_data():
    # Load the Excel file to read and extract relevant data
    folder_path = r"data"
    file_path = r"data/sigma profile_BOLs.xls"

    # Load all sheets into a dictionary
    data_sheets = pd.read_excel(file_path, sheet_name=None)
    experimental_data_file_path =r"data/Experimental data...BOLS.xlsx"
    experimental_data = pd.read_excel(experimental_data_file_path, sheet_name=None)
    density_data = experimental_data['density']
    viscosity_data = experimental_data['viscosity']
    vapor_pressure = experimental_data['vapor pressure']


    density_data.columns = density_data.iloc[3]
    density_data = density_data.iloc[4:]
    # drop first two columns
    density_data = density_data.drop(density_data.columns[0], axis=1)
    density_data = density_data.drop(density_data.columns[0], axis=1)



    viscosity_data.columns = viscosity_data.iloc[2]
    viscosity_data = viscosity_data.iloc[3:]
    # drop first column
    viscosity_data = viscosity_data.drop(viscosity_data.columns[0], axis=1)
    
    vapor_pressure = vapor_pressure.iloc[2:,2:]
    vapor_pressure = vapor_pressure.ffill()
    vapor_pressure_copy = {}
    vapor_pressure_copy_keys = ["CO2BOL",]+[str(i) + " K"  for i in vapor_pressure["Unnamed: 3"].unique()]
    for i in vapor_pressure_copy_keys:
        vapor_pressure_copy[i] = []
    vapor_pressure_copy[vapor_pressure_copy_keys[0]] = [i for i in vapor_pressure["Unnamed: 2"].unique()]
    for i in vapor_pressure_copy[vapor_pressure_copy_keys[0]]:
        _ = vapor_pressure[vapor_pressure["Unnamed: 2"] == i]
        for t in [i  for i in vapor_pressure["Unnamed: 3"].unique()]:
            vapor_pressure_copy[str(t)+" K"].append(np.nan)
            try:
                vapor_pressure_copy[str(t)+" K"][-1] = _[_["Unnamed: 3"] == t]["Unnamed: 4"].item()
            except:
                None
    vapor_pressure_data = pd.DataFrame(vapor_pressure_copy)


    sheets = list(data_sheets.keys())
    pure_sigmas_df = data_sheets['SIGMA PROFILES\n No.2 298.15 K']
    pure_sigmas_df.columns = pure_sigmas_df.iloc[0]
    pure_sigmas_df = pure_sigmas_df.iloc[1:]
    pure_sigmas_df = pure_sigmas_df.iloc[:61]
    pure_sigmas_df_column_names = ["sigma", "DBU", "BuOH", "HexOH", 'EGME', 'EGEE']
    pure_sigmas_df.columns = pure_sigmas_df_column_names


    cleaned_pure_sigma_df = pure_sigmas_df.copy()
    cleaned_pure_sigma_df['all'] = mix_sigma([cleaned_pure_sigma_df[i] for i in pure_sigmas_df_column_names[1:]], [1, 1, 1, 1, 1])
    cleaned_pure_sigma_df = pure_sigmas_df[cleaned_pure_sigma_df['all'] != 0]
    sigma_values = np.array(cleaned_pure_sigma_df['sigma'])
    return density_data, viscosity_data, vapor_pressure_data, cleaned_pure_sigma_df, sigma_values



def process_property_data(data, cleaned_pure_sigma_df):
    # Initialize variables
    components = []
    composition = []
    mole_fractions = []

    _ = data['CO2BOL']
    for i in _:
        composition.append([i[-4], i[-2]])
        components.append([i[:3], i[4:-6]])
        mole_fractions.append([float(i[-4])/(float(i[-4])+float(i[-2])), float(i[-2])/(float(i[-4])+float(i[-2]))])
    molar_mass = {
        "DBU": 152.24, 
        "BuOH": 74.12, 
        "HexOH": 102.17, 
        'EGME': 76.09, 
        'EGEE': 90.12
    }
    # Collect property data
    property_all_data = []
    for i in data['CO2BOL'].values:
        for j in list(data.columns)[1:]:
            property_all_data.append([])
            property_all_data[-1] += [float(j[:-2]), ] # added temperature as first column
            components = [i[:3], i[4:-6]]
            mole_fractions = [float(i[-4])/(float(i[-4])+float(i[-2])), float(i[-2])/(float(i[-4])+float(i[-2]))]
            # added combined molar mass as 2nd column
            property_all_data[-1] += [mole_fractions[0]*molar_mass[components[0]] + mole_fractions[1]*molar_mass[components[1]]]
            c1 = cleaned_pure_sigma_df[components[0]]
            c2 = cleaned_pure_sigma_df[components[1]]
            # added combined sigma values as additional columns
            property_all_data[-1] += list(mix_sigma([c1, c2], mole_fractions))
            # added property value as last column
            property_all_data[-1] += [float(data[data['CO2BOL'] == i][j]), ]
    property_all_data_np = np.array(property_all_data)
    property_all_data_np = property_all_data_np[~np.isnan(property_all_data_np).any(axis=1)]
    return property_all_data_np



def get_cut_areas(property_p_data, sigma_values, sigma_cuts, other_feats = (0, 1)):
    """
    sigma_cuts is a list with elements as [low, high] for each cut
    """
    areas = []
    for low, high in sigma_cuts:
        low, high = min([low, high]), max([low, high])
        valid_cols = (sigma_values>=low) & (sigma_values<=high)
        valid_p_data = np.delete(property_p_data, other_feats, axis=1)[:,valid_cols]
        areas.append(trapezoidal_area(valid_p_data, x_gap = 0.001))
    return np.concatenate(areas, axis = 1)

def get_uniform_cuts(gap, _start = -0.019, _end = 0.027, n_cuts = None):
    cuts = []
    if n_cuts is None:
        while _start < _end:
            cuts.append([round(_start, ndigits = 3), min(round(_start + gap, ndigits = 3), _end)])
            _start += gap
        return cuts
    else:
        if n_cuts < 1:
            raise ValueError("n_cuts must be greater than 0")
        if n_cuts == 1:
            return [[_start, _end],]
        else:
            gap = _end - _start
            while len(cuts) <= n_cuts:
                cuts = get_uniform_cuts(gap, _start = -0.019, _end = 0.027, n_cuts = None)
                gap -= 0.001
            return get_uniform_cuts(gap + 0.002, _start = -0.019, _end = 0.027, n_cuts = None)


def merge_cuts_and_other_feats(cut_areas_data, other_feats_data):
    # both must be 2D array
    return np.hstack((cut_areas_data, other_feats_data))


def final_data(property_p_xdata, sigma_values, sigma_cuts, other_feats = (0, 1)):
    cut_areas_data = get_cut_areas(property_p_xdata, sigma_values, sigma_cuts, other_feats = other_feats)
    other_feats_data = property_p_xdata[:,other_feats]
    return merge_cuts_and_other_feats(cut_areas_data, other_feats_data)

def get_train_val_test(X_data, y_data, test_size = 0.15, val_size = 0.15, random_split = True, random_state = 42):
    if random_split:
        np.random.seed(random_state)
        rand_ind = np.random.permutation(len(X_data))
        X_data = X_data[rand_ind]
        y_data = y_data[rand_ind]
    X_test = X_data[int(len(X_data)*(1-test_size)):]
    y_test = y_data[int(len(X_data)*(1-test_size)):]
    X_val = X_data[int(len(X_data)*(1-test_size-val_size)):int(len(X_data)*(1-test_size))]
    y_val = y_data[int(len(X_data)*(1-test_size-val_size)):int(len(X_data)*(1-test_size))]
    X_train = X_data[:int(len(X_data)*(1-test_size-val_size))]
    y_train = y_data[:int(len(X_data)*(1-test_size-val_size))]
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train=None, X_val=None, X_test=None, 
               y_train=None, y_val=None, y_test=None, 
               method='standard', random_state=42):
    """
    Scales the provided training, validation, and test datasets using the specified method.

    Parameters:
    - X_train, X_val, X_test: Feature datasets
    - y_train, y_val, y_test: Target datasets
    - method: 'standard' for StandardScaler, 'min_max' for MinMaxScaler

    Returns:
    - Dictionary containing scaled datasets and scalers
    """
    np.random.seed(random_state)
    
    # Choose the scaling method
    if method == 'standard':
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    elif method == 'min_max':
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'min_max'.")
    
    # Function to safely scale data
    def safe_transform(scaler, data, fit=False):
        if data is not None:
            if fit:
                return scaler.fit_transform(data)
            else:
                return scaler.transform(data)
        return None

    # Scale feature datasets
    X_train = safe_transform(x_scaler, X_train, fit=True)
    X_val = safe_transform(x_scaler, X_val)
    X_test = safe_transform(x_scaler, X_test)

    # Scale target datasets (reshape if necessary)
    if y_train is not None:
        y_train = safe_transform(y_scaler, y_train, fit=True)
    if y_val is not None:
        y_val = safe_transform(y_scaler, y_val)
    if y_test is not None:
        y_test = safe_transform(y_scaler, y_test)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test, 
        "y_train": y_train, "y_val": y_val, "y_test": y_test, 
        "X_scaler": x_scaler, "y_scaler": y_scaler
    }



def descale_data(X_data = None, y_data = None, x_scaler = None, y_scaler = None):
    if X_data is not None:
        X_data = x_scaler.inverse_transform(X_data)
    if y_data is not None:
        y_data = y_scaler.inverse_transform(y_data)
    return {"X_data": X_data, "y_data": y_data}


def get_complete_data_without_cut(output_cols = (-1, ), val_size = 0.15, test_size = 0.15, random_split = True, random_state = 42):
    density_data, viscosity_data, vapor_pressure_data, cleaned_pure_sigma_df, sigma_values = load_data()
    processed_d_data = process_property_data(density_data, cleaned_pure_sigma_df)
    processed_v_data = process_property_data(viscosity_data, cleaned_pure_sigma_df)
    processed_vp_data = process_property_data(vapor_pressure_data, cleaned_pure_sigma_df)

    X_d_train, X_d_val, X_d_test, y_d_train, y_d_val, y_d_test = \
        get_train_val_test(
            np.delete(processed_d_data, output_cols, axis=1), processed_d_data[:,output_cols], 
            val_size = val_size, test_size = test_size, random_split = random_split, random_state = random_state)

    X_v_train, X_v_val, X_v_test, y_v_train, y_v_val, y_v_test = \
        get_train_val_test(
            np.delete(processed_v_data, output_cols, axis=1), processed_v_data[:,output_cols], 
            val_size = val_size, test_size = test_size, random_split = random_split, random_state = random_state)
    
    X_vp_train, X_vp_val, X_vp_test, y_vp_train, y_vp_val, y_vp_test = \
        get_train_val_test(
            np.delete(processed_vp_data, output_cols, axis=1), processed_vp_data[:,output_cols], 
            val_size = val_size, test_size = test_size, random_split = random_split, random_state = random_state)

    return {"d_data": {"X_train": X_d_train, 
                       "X_val": X_d_val, 
                       "X_test": X_d_test, 
                       "y_train": y_d_train, 
                       "y_val": y_d_val, 
                       "y_test": y_d_test}, 
            "v_data": {"X_train": X_v_train, 
                       "X_val": X_v_val, 
                       "X_test": X_v_test, 
                       "y_train": y_v_train, 
                       "y_val": y_v_val, 
                       "y_test": y_v_test}, 
            "vp_data": {"X_train": X_vp_train, 
                       "X_val": X_vp_val, 
                       "X_test": X_vp_test, 
                       "y_train": y_vp_train, 
                       "y_val": y_vp_val, 
                       "y_test": y_vp_test},
            "sigma_values": sigma_values}


