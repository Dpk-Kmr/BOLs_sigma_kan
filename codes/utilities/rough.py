from data_clean import *

density_data, viscosity_data, vapor_pressure_data, cleaned_pure_sigma_df, sigma_values = load_data()
processed_d_data = process_property_data(density_data, cleaned_pure_sigma_df)


print(final_data(processed_d_data, sigma_values, [[0.01, 0.02], [-0.01, -0.02], [0.001, 0.003]]))