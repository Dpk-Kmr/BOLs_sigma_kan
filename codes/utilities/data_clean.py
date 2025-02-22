import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

    pure_components = ["DBU", "BuOH", "hol", 'mol', 'eol']
    _ = data['CO2BOL']
    for i in _:
        composition.append([i[-4], i[-2]])
        components.append([i[:3], i[4:-6]])
        mole_fractions.append([float(i[-4])/(float(i[-4])+float(i[-2])), float(i[-2])/(float(i[-4])+float(i[-2]))])

    # Collect property data
    property_all_data = []
    for i in data['CO2BOL'].values:
        for j in list(data.columns)[1:]:
            property_all_data.append([])
            property_all_data[-1] += [float(j[:-2]), ]
            components = [i[:3], i[4:-6]]
            mole_fractions = [float(i[-4])/(float(i[-4])+float(i[-2])), float(i[-2])/(float(i[-4])+float(i[-2]))]
            c1 = cleaned_pure_sigma_df[components[0]]
            c2 = cleaned_pure_sigma_df[components[1]]
            property_all_data[-1] += list(mix_sigma([c1, c2], mole_fractions))
            property_all_data[-1] += [float(data[data['CO2BOL'] == i][j]), ]
    property_all_data_np = np.array(property_all_data)
    property_all_data_np = property_all_data_np[~np.isnan(property_all_data_np).any(axis=1)]
    return property_all_data_np



def get_cut_areas(property_p_data, sigma_values, sigma_cuts):
    """
    sigma_cuts is a list with elements as [low, high] for each cut
    """
    areas = []
    for low, high in sigma_cuts:
        valid_cols = (sigma_values>=low) & (sigma_values<=high)
        valid_p_data = property_p_data[:,valid_cols]
        areas.append(trapezoidal_area(valid_p_data, x_gap = 0.001))
    return np.concatenate(areas, axis = 1)

def get_uniform_cuts(gap, _start = -0.019, _end = 0.027):
    cuts = []
    while _start < _end:
        cuts.append([_start, _start + gap])
        _start += gap
    return cuts

def merge_cuts_and_other_feats(cut_areas, other_feats):
    # both must be 2D array
    return np.hstack((cut_areas, other_feats))

def get_train_val_test(X_data, y_data, test_size = 0.15, val_size = 0.15, random_split = True):
    rand_ind = np.random.permutation(len(X_data))
    X_data = X_data[rand_ind]
    y_data = y_data[rand_ind]
    X_test = X_data[int(len(X_data)*(1-test_size)):]
    y_test = y_data[int(len(X_data)*(1-test_size)):]
    X_val = X_data[int(len(X_data)*(1-test_size-val_size)):int(len(X_data)*(1-test_size))]
    y_val = y_data[int(len(X_data)*(1-test_size-val_size)):int(len(X_data)*(1-test_size))]
    X_train = X_data[:int(len(X_data)*(1-test_size-val_size))]
    y_train = y_data[:int(len(X_data)*(1-test_size-val_size))]
    return X_train, X_val, X_test, y_train, y_val, y_test, rand_ind

def scale_data(X_train, X_val, X_test, y_train, y_val, y_test, method = 'standard'):
    if method == 'standard':
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)

    elif method == 'min_max':
        x_scaler = MinMaxScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_val = y_scaler.transform(y_val)
        y_test = y_scaler.transform(y_test)

    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'min_max'.")
    return X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler

def descale_data(X_data = None, y_data = None, x_scaler = None, y_scaler = None):
    if X_data is not None:
        X_data = x_scaler.inverse_transform(X_data)
    if y_data is not None:
        y_data = y_scaler.inverse_transform(y_data)
    return X_data, y_data

