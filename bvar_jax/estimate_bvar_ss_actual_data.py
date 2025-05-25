# --- estimate_bvar_ss.py ---
# This script ties everything together: load config, prepare data,
# run NumPyro MCMC, and use the single-draw smoother.

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
import numpyro
from numpyro.infer import MCMC, NUTS # , init_to_value # init_to_value can be useful for debugging
from numpyro.diagnostics import print_summary
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import time
import os
import yaml # Import yaml for loading config

# Assuming utils directory is in the Python path
# from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax # Not needed directly here
from utils.Kalman_filter_jax import simulate_state_space # Use the simulation from KF file

# Import the NumPyro model and the single-draw smoother routine
from var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax
from run_single_draw import run_simulation_smoother_single_params_jit

# Import BVARConfig class from your config.py
# This requires config.py to be accessible (e.g., in the same directory or installed)
# from config import BVARConfig # Assuming config.py is available


# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
# Configure JAX to use CPU (optional, but might be faster for small models/MCMC)
# jax.config.update("jax_platforms", "cpu")

# Helper to load YAML config (simplified version if BVARConfig is not directly available)
def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Loads YAML config and performs minimal parsing for dimensions/names."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract dimensions and names
    observable_vars = config_dict.get('variables', {}).get('observable', [])
    trend_vars = config_dict.get('variables', {}).get('trends', [])
    stationary_vars = config_dict.get('variables', {}).get('stationary', [])

    config_data = {
        'var_order': config_dict.get('var_order', 1),
        'variables': {
            'observable_names': [v['name'] for v in observable_vars],
            'trend_names': [v['name'] for v in trend_vars],
            'stationary_var_names': [v['name'] for v in stationary_vars],
        },
        'model_equations': config_dict.get('model_equations', {}),
        'initial_conditions': config_dict.get('initial_conditions', {}),
        'stationary_prior': config_dict.get('stationary_prior', {}),
        'trend_shocks': config_dict.get('trend_shocks', {}),
        'parameters': config_dict.get('parameters', {}), # Includes measurement parameters
    }

    # Pre-parse initial conditions using the function from var_ss_model
    config_data['initial_conditions_parsed'] = parse_initial_state_config(config_data['initial_conditions'])

    # Pre-parse model equations using the function from var_ss_model
    # Needs measurement parameter names to do this correctly
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config]

    parsed_model_eqs_list = []
    # Ensure model_eqs is a list of (lhs, rhs) tuples
    raw_model_eqs = config_data['model_equations']
    if isinstance(raw_model_eqs, dict):
         raw_model_eqs = list(raw_model_eqs.items())
    elif isinstance(raw_model_eqs, list):
         # If it's a list of dicts like [{'lhs1': 'rhs1'}, {'lhs2': 'rhs2'}]
         temp_list = []
         for eq_dict in raw_model_eqs:
              for lhs, rhs in eq_dict.items():
                   temp_list.append((lhs, rhs))
         raw_model_eqs = temp_list


    # Get observable indices mapping for parsing equations
    observable_indices = {name: i for i, name in enumerate(config_data['variables']['observable_names'])}
    trend_names = config_data['variables']['trend_names']
    stationary_names = config_data['variables']['stationary_var_names']


    for obs_name, eq_str in raw_model_eqs:
        parsed_terms = _parse_equation_jax(eq_str, trend_names, stationary_names, measurement_param_names)
        parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))

    config_data['model_equations_parsed'] = parsed_model_eqs_list


    return config_data


# --- Main Script ---
# Replace with your actual file paths
data_file_path = 'data_m5.csv'
yaml_file_path = 'bvar_stationary_model5_lager_variance.yml' # Assuming this is in the same directory

# Set working directory to script location (optional, helps with file paths)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print("Current Working Directory:", os.getcwd())


# --- 1. Load Configuration and Data ---
print("Loading configuration and data...")
try:
    # Use your BVARConfig if available, otherwise use the simple loader
    # config_handler = BVARConfigHandler(yaml_file_path) # If you have BVARConfigHandler class
    # config_data = config_handler.config # Get the BVARConfig object

    # Using the simplified loader:
    config_data = load_config_from_yaml(yaml_file_path)
    print("Configuration loaded.")

except Exception as e:
    print(f"Error loading or parsing configuration: {e}")
    # sys.exit(1) # Exit if config loading fails


# Read the CSV data using observable names from config
observable_names = config_data['variables']['observable_names']
try:
    df_raw = pd.read_csv(data_file_path)
    # Assuming 'Date' column is always present and needs conversion
    df_raw['Date'] = pd.PeriodIndex(df_raw['Date'], freq='Q').to_timestamp()
    df_raw = df_raw.set_index('Date')

    # Select only the observable columns specified in the config
    data_observed_pd = df_raw[observable_names]

    # Convert to JAX array, handling NaNs if any exist in the original data
    data_observed_jax = jnp.asarray(data_observed_pd.values, dtype=_DEFAULT_DTYPE)

    print(f"Data loaded. Shape: {data_observed_jax.shape}")
    print(f"Data preview:\n{data_observed_pd.head()}")

except FileNotFoundError:
    print(f"Error: Data file not found at {data_file_path}")
    # sys.exit(1)
except KeyError as e:
    print(f"Error: Observable variable '{e}' not found in data file columns.")
    # sys.exit(1)
except Exception as e:
    print(f"Error loading or processing data: {e}")
    # sys.exit(1)


# Identify static NaN handling parameters based on the loaded data
valid_obs_mask_cols = jnp.any(jnp.isfinite(data_observed_jax), axis=0)
static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
static_n_obs_actual = static_valid_obs_idx.shape[0]

if static_n_obs_actual == 0:
    print("Warning: No observation series are observed after removing all-NaN columns.")
    print("MCMC cannot be run. Exiting.")
    # sys.exit(1)


# --- 2. Setup and Run MCMC ---
print("\nSetting up MCMC...")

# Extract necessary config pieces for the model
trend_var_names = config_data['variables']['trend_names']
stationary_var_names = config_data['variables']['stationary_var_names']
model_eqs = config_data['model_equations'] # Pass raw equations, model will use parsed
initial_conds = config_data['initial_conditions'] # Pass raw, model will use parsed
stationary_prior_config = config_data['stationary_prior']
trend_shocks_config = config_data['trend_shocks']
measurement_params_config = config_data.get('parameters', {}).get('measurement', []) # Default to empty list if no params


# Prepare initial parameter values for MCMC (optional, can use default init_to_sample)
# This helps guide the sampler, but requires careful selection.
# Based on PyMC code: a_flat ~ Normal(es, fs), stationary variances ~ IG, stationary_chol ~ LKJ, trend variances ~ IG, measurement_params ~ priors
# Initial values could be means of priors or slightly perturbed.
# For simplicity, let's use NumPyro's default init_to_sample which is often robust.
init_strategy = numpyro.infer.init_to_sample()

key_mcmc = random.PRNGKey(1) # Use a different key for MCMC

# Define kernel and MCMC
kernel = NUTS(
    model=numpyro_bvar_stationary_model,
    init_strategy=init_strategy,
    # adapt_state_size=True, # Can help with large state spaces
    # dense_mass=False, # Use diagonal mass matrix for potentially faster adaptation
    allow_partial_steps=False
)

num_warmup = 500
num_samples = 1000
num_chains = 2 # Run multiple chains for convergence check

print(f"Running MCMC with {num_warmup} warmup steps and {num_samples} sampling steps per chain ({num_chains} chains)...")
start_time_mcmc = time.time()

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

# Run MCMC, passing data and config pieces to the model
# The model function expects parsed initial_conditions and model_equations
# Pass config_data which contains these pre-parsed structures.
mcmc.run(
    key_mcmc,
    y=data_observed_jax,
    config_data=config_data, # Pass the full config_data dict
    static_valid_obs_idx=static_valid_obs_idx,
    static_n_obs_actual=static_n_obs_actual,
    # Explicitly pass these for clarity, though they are in config_data
    trend_var_names=trend_var_names,
    stationary_var_names=stationary_var_names,
    observable_names=observable_names,
    model_eqs=model_eqs, # Pass raw equations, model uses parsed from config_data
    initial_conds=initial_conds, # Pass raw, model uses parsed from config_data
    stationary_prior_config=stationary_prior_config,
    trend_shocks_config=trend_shocks_config,
    measurement_params_config=measurement_params_config
)

end_time_mcmc = time.time()
print(f"MCMC completed in {end_time_mcmc - start_time_mcmc:.2f} seconds.")

mcmc.print_summary()

# Get posterior samples
posterior_samples = mcmc.get_samples()

# Convert to ArviZ InferenceData object for easier analysis (optional)
# idata = az.from_numpyro(mcmc)
# az.summary(idata)


# --- 3. Use the Single-Draw Simulation Smoother Routine ---
print("\nRunning single-draw simulation smoother with posterior mean parameters...")

# Get posterior mean for sampled parameters
# Need to collect all sampled parameter names from the model
# Use the keys from posterior_samples directly, except for 'log_likelihood' and deterministic ones
sampled_param_names = [
    name for name in posterior_samples.keys()
    if name not in ['log_likelihood', 'Phi_1', 'Sigma_cycles', 'Sigma_trends',
                    'T_comp', 'R_comp', 'C_comp', 'H_comp', 'init_x_comp',
                    'init_P_comp', 'is_stationary', 'k_states'] # Exclude deterministics and likelihood
]

posterior_mean_params = {
    name: jnp.mean(posterior_samples[name], axis=0)
    for name in sampled_param_names
}

# Run the single-draw smoother routine with the posterior mean parameters
key_smooth_draws = random.PRNGKey(2) # Use a new key for the draws
num_sim_draws = 10 # Number of simulation draws for this example
print(f"Running simulation smoother for {num_sim_draws} draws...")

start_time_smooth_draws = time.time()

# Pass the config_data directly, which contains pre-parsed structures
smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
    posterior_mean_params,
    data_observed_jax, # Pass original data (potentially with NaNs)
    config_data, # Pass the full config_data dict
    key_smooth_draws,
    num_draws=num_sim_draws # Specify number of draws
)

end_time_smooth_draws = time.time()
print(f"Simulation smoother draws completed in {end_time_smooth_draws - start_time_smooth_draws:.2f} seconds.")

# Process simulation results (mean, median, all_draws)
if num_sim_draws > 1:
    mean_sim_states, median_sim_states, all_sim_draws = simulation_results
    print(f"Smoothed states (original data, posterior mean params) shape: {smoothed_states_original.shape}")
    print(f"Mean simulated states shape: {mean_sim_states.shape}")
    print(f"Median simulated states shape: {median_sim_states.shape}")
    print(f"All simulation draws shape: {all_sim_draws.shape}")

    # Check for NaNs in results
    print(f"NaNs in smoothed_states_original: {jnp.any(jnp.isnan(smoothed_states_original))}")
    print(f"NaNs in mean_sim_states: {jnp.any(jnp.isnan(mean_sim_states))}")
    print(f"NaNs in median_sim_states: {jnp.any(jnp.isnan(median_sim_states))}")
    print(f"NaNs in all_sim_draws: {jnp.any(jnp.isnan(all_sim_draws))}")


    # --- 4. (Optional) Plotting ---
    print("\nGenerating plots (optional)...")
    # Get state names for plotting
    k_states = posterior_samples['k_states'].mean().astype(int) # Get k_states from deterministic
    # Need state names list, consistent with state vector ordering [trends, stationary_t, ...]
    trend_var_names = config_data['variables']['trend_names']
    stationary_var_names = config_data['variables']['stationary_var_names']
    p = config_data['var_order']
    state_names = trend_var_names + [
         f"{name}_t_minus_{lag}" if lag > 0 else name
         for lag in range(p) for name in stationary_var_names
    ]
    # Adjust names for p=1: [trend_vars, stationary_t]
    if p == 1:
        state_names = trend_var_names + stationary_var_names


    try:
        # Plotting the first few states (trends and cycles)
        num_states_to_plot = min(k_states, 6) # Plot up to 6 states

        fig, axes = plt.subplots(num_states_to_plot, 1, figsize=(12, 3 * num_states_to_plot), sharex=True)
        if num_states_to_plot == 1:
            axes = [axes]

        dates = data_observed_pd.index # Use pandas index for dates

        # Set up date formatting
        formatter = mdates.DateFormatter('%Y')
        # locator = mdates.YearLocator(2) # Optional: major ticks every 2 years

        for i in range(num_states_to_plot):
            ax = axes[i]
            state_name = state_names[i]

            # Plot smoothed state mean
            ax.plot(dates, smoothed_states_original[:, i], label='Smoothed State Mean', color='blue')

            # Plot mean and median simulated states
            ax.plot(dates, mean_sim_states[:, i], label='Mean Simulated State', color='red', linestyle='--')
            ax.plot(dates, median_sim_states[:, i], label='Median Simulated State', color='green', linestyle=':')

            # Plot a few individual draws (optional)
            # for j in range(min(num_sim_draws, 3)): # Plot up to 3 draws
            #      ax.plot(dates, all_sim_draws[j, :, i], color='purple', alpha=0.1)

            ax.set_title(f'State: {state_name}')
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_major_locator(locator)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


        plt.tight_layout()
        plt.show()

        # Optional: Plot observed data with fitted/smoothed values (from C @ x_smooth)
        # This requires computing C @ x_smooth using the posterior mean C matrix
        print("\nGenerating observed data vs fitted values plot (optional)...")
        try:
            # Get the mean C matrix from posterior samples
            mean_C_comp = jnp.mean(posterior_samples['C_comp'], axis=(0, 1)) # Average over chain and draw

            # Compute fitted values: y_fitted = C @ x_smoothed_mean
            fitted_values = smoothed_states_original @ mean_C_comp.T # (T, k_states) @ (k_states, k_endog)

            # Plot the first few observed variables
            num_obs_to_plot = min(k_endog, 4)
            fig, axes = plt.subplots(num_obs_to_plot, 1, figsize=(12, 3 * num_obs_to_plot), sharex=True)
            if num_obs_to_plot == 1:
                 axes = [axes]

            for i in range(num_obs_to_plot):
                 ax = axes[i]
                 obs_name = observable_names[i]

                 # Plot original observed data (handle NaNs)
                 ax.plot(dates, data_observed_pd[obs_name], label='Observed', color='gray', alpha=0.7)

                 # Plot fitted/smoothed values
                 ax.plot(dates, fitted_values[:, i], label='Fitted (C @ Smoothed Mean)', color='red')

                 ax.set_title(f'Observed vs Fitted: {obs_name}')
                 ax.legend(fontsize=8)
                 ax.grid(True)
                 ax.xaxis.set_major_formatter(formatter)
                 # ax.xaxis.set_major_locator(locator)
                 plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            plt.tight_layout()
            plt.show()


        except Exception as e:
             print(f"Error during plotting observed vs fitted: {e}")


    except Exception as e:
        print(f"Error during plotting states: {e}")

elif num_sim_draws == 1:
     single_sim_draw = simulation_results
     print(f"Single simulation draw shape: {single_sim_draw.shape}")
     print(f"NaNs in single_sim_draw: {jnp.any(jnp.isnan(single_sim_draw))}")

     # Plotting for single draw (first few states)
     try:
        # Get state names for plotting
        k_states = posterior_samples['k_states'].mean().astype(int)
        trend_var_names = config_data['variables']['trend_names']
        stationary_var_names = config_data['variables']['stationary_var_names']
        p = config_data['var_order']
        state_names = trend_var_names + [
             f"{name}_t_minus_{lag}" if lag > 0 else name
             for lag in range(p) for name in stationary_var_names
        ]
        if p == 1:
             state_names = trend_var_names + stationary_var_names


        num_states_to_plot = min(k_states, 6) # Plot up to 6 states
        fig, axes = plt.subplots(num_states_to_plot, 1, figsize=(12, 3 * num_states_to_plot), sharex=True)
        if num_states_to_plot == 1:
            axes = [axes]

        dates = data_observed_pd.index

        formatter = mdates.DateFormatter('%Y')

        for i in range(num_states_to_plot):
            ax = axes[i]
            state_name = state_names[i]

            ax.plot(dates, smoothed_states_original[:, i], label='Smoothed State Mean', color='blue')
            ax.plot(dates, single_sim_draw[:, i], label='Single Simulated State Draw', color='red', linestyle='--')

            ax.set_title(f'State: {state_name}')
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

     except Exception as e:
        print(f"Error during plotting single draw: {e}")


print("\nScript finished.")