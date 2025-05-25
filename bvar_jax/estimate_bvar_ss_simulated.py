# --- estimate_bvar_ss_simulated.py (Final Corrected Version - Call JITted Function Correctly) ---
# This script runs the NumPyro BVAR with trends estimation on SIMULATED data,
# performs simulation smoothing, and compares to the true simulated states.

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value # init_to_value can be useful for debugging
from numpyro.diagnostics import print_summary
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import time
import os
import yaml # Import yaml for loading config
from typing import Dict, Any, List, Tuple

# Configure JAX as requested
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") # Use CPU as requested
_DEFAULT_DTYPE = jnp.float64 # Define _DEFAULT_DTYPE here

# Assuming utils directory is in the Python path
# Need create_companion_matrix_jax for defining true Phi list
from utils.stationary_prior_jax_simplified import create_companion_matrix_jax

# Import the simulation function
from simulate_bvar_jax import simulate_bvar_with_trends_jax

# Import the NumPyro model and the single-draw smoother routine, and parsing helpers
from var_ss_model import numpyro_bvar_stationary_model, parse_initial_state_config, _parse_equation_jax, _get_off_diagonal_indices
from run_single_draw import run_simulation_smoother_single_params_jit

# Helper to load YAML config and parse for model/sim functions
def load_config_and_parse(yaml_path: str) -> Dict[str, Any]:
    """Loads YAML config and performs necessary parsing for use in JAX functions."""
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
        'model_equations': config_dict.get('model_equations', {}), # Keep raw for reference
        'initial_conditions': config_dict.get('initial_conditions', {}), # Keep raw for reference
        'stationary_prior': config_dict.get('stationary_prior', {}),
        'trend_shocks': config_dict.get('trend_shocks', {}),
        'parameters': config_dict.get('parameters', {}), # Includes measurement parameters
    }

    # --- Perform Parsing for JAX/NumPyro Consumption ---

    # 1. Parse initial conditions using the function from var_ss_model
    config_data['initial_conditions_parsed'] = parse_initial_state_config(config_data['initial_conditions'])

    # 2. Parse model equations using the function from var_ss_model
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config]

    parsed_model_eqs_list = []
    raw_model_eqs = config_data['model_equations']
    if isinstance(raw_model_eqs, dict):
         raw_model_eqs = list(raw_model_eqs.items())
    elif isinstance(raw_model_eqs, list):
         temp_list = []
         for eq_dict in raw_model_eqs:
              for lhs, rhs in eq_dict.items():
                   temp_list.append((lhs, rhs))
         raw_model_eqs = temp_list
    # Ensure raw_model_eqs is a list of (obs_name, eq_str) tuples

    observable_indices = {name: i for i, name in enumerate(config_data['variables']['observable_names'])}
    trend_names = config_data['variables']['trend_names']
    stationary_names = config_data['variables']['stationary_var_names']


    for obs_name, eq_str in raw_model_eqs:
        # Use the JAX parsing function
        parsed_terms = _parse_equation_jax(eq_str, trend_names, stationary_names, measurement_param_names)
        parsed_model_eqs_list.append((observable_indices[obs_name], parsed_terms))

    config_data['model_equations_parsed'] = parsed_model_eqs_list

    # 3. Identify trend names with shocks
    trend_shocks_spec = config_data['trend_shocks'].get('trend_shocks', {})
    config_data['trend_names_with_shocks'] = [name for name in trend_names if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and 'distribution' in trend_shocks_spec[name]]

    # 4. Pre-calculate static indices for off-diagonal elements of A (k_stationary x k_stationary)
    k_stationary = len(config_data['variables']['stationary_var_names'])
    # Use the helper function now defined in var_ss_model
    off_diag_rows, off_diag_cols = _get_off_diagonal_indices(k_stationary)
    config_data['static_off_diag_indices'] = (off_diag_rows, off_diag_cols)
    config_data['num_off_diag'] = k_stationary * (k_stationary - 1)


    return config_data


# --- Define True Parameters for Simulation ---
# This function now lives in estimate_bvar_ss_simulated.py
def define_true_params(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Defines a set of true parameters for simulation, matching the model structure."""
    k_trends = len(config_data['variables']['trend_names'])
    k_stationary = len(config_data['variables']['stationary_var_names']) # Corrected access

    p = config_data['var_order']
    trend_names_with_shocks = config_data['trend_names_with_shocks']
    n_trend_shocks = len(trend_names_with_shocks)

    # True VAR(p) coefficient matrices (list of p matrices, k_stationary x k_stationary)
    # Ensure stationarity. For a simple example, diagonal matrices with values < 1.
    # Simulate_bvar_data had Phi = [[0.7, 0.2], [0.1, 0.5]] for k_stat=2, p=1
    Phi_list_true = [jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.7 for _ in range(p)] # Start with diagonal 0.7
    if k_stationary >= 2 and p >= 1:
         Phi_list_true[0] = Phi_list_true[0].at[0, 1].set(0.2)
         Phi_list_true[0] = Phi_list_true[0].at[1, 0].set(0.1) # Use 0.1 instead of -0.05

    # Check stationarity of the chosen true Phi
    # Use create_companion_matrix_jax from stationary_prior
    try:
        T_check = create_companion_matrix_jax(Phi_list_true, p, k_stationary)
        eigenvalues = jnp.linalg.eigvals(T_check)
        max_abs_eig = jnp.max(jnp.abs(eigenvalues))
        if max_abs_eig >= 1.0 - 1e-6:
            print(f"Warning: True Phi_list results in non-stationary companion matrix (max abs eig: {max_abs_eig}).")
            # Adjust values if needed or accept that simulated data is non-stationary
            # For a simple example, let's ensure stationarity by picking smaller values
            Phi_list_true = [jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.5 for _ in range(p)]
            if k_stationary >= 2 and p >= 1:
                 Phi_list_true[0] = Phi_list_true[0].at[0, 1].set(0.1)
                 Phi_list_true[0] = Phi_list_true[0].at[1, 0].set(0.05)
            T_check_adj = create_companion_matrix_jax(Phi_list_true, p, k_stationary)
            print(f"Adjusted true Phi_list. Max abs eig: {jnp.max(jnp.abs(jnp.linalg.eigvals(T_check_adj)))}")

    except ImportError:
        print("Warning: Could not import create_companion_matrix_jax to check true Phi stationarity.")
    except Exception as e:
         print(f"Warning: Could not check true Phi stationarity: {e}")


    # True stationary cycle shock covariance (k_stationary x k_stationary)
    # Simulate_bvar_data had Sigma_stationary = [[0.5, 0.1], [0.1, 0.3]] for k_stat=2
    Sigma_cycles_true = jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 0.5 # Start diagonal 0.5
    if k_stationary >= 2:
        Sigma_cycles_true = Sigma_cycles_true.at[0, 1].set(0.1)
        Sigma_cycles_true = Sigma_cycles_true.at[1, 0].set(0.1)
    Sigma_cycles_true = (Sigma_cycles_true + Sigma_cycles_true.T) / 2.0 # Ensure symmetry


    # True trend shock variances (n_trend_shocks,) - for diagonal Sigma_trends_sim
    # Corresponds to trend_names_with_shocks
    # simulate_bvar_data used sigma_gdp=0.05 (var 0.0025), sigma_inf=0.0707 (var 0.005)
    true_trend_vars_dict_sim = {
        'trend_gdp': 0.05**2, # variance 0.0025
        'trend_inf': 0.0707**2, # variance 0.005
        # Add other defaults if more trends exist but not specified in YAML
        # For simplicity, assume trends not listed have zero variance shock in sim
    }
    # Collect true trend variances for trends *with shocks defined in YAML*
    true_trend_vars_with_shocks_sim = jnp.array([
         true_trend_vars_dict_sim.get(name, 0.0) # Use variance if name in dict, 0 otherwise
         for name in trend_names_with_shocks # Use the filtered list
    ], dtype=_DEFAULT_DTYPE)
    Sigma_trends_sim_true = jnp.diag(true_trend_vars_with_shocks_sim) # Shape (n_trend_shocks, n_trend_shocks)


    # True measurement parameters (dict of scalars)
    measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
    measurement_param_names = [p['name'] for p in measurement_params_config]
    true_measurement_params = {
        name: 1.0 # Default to 1.0 if no specific true value provided and it's a parameter name
        for name in measurement_param_names
    }
    # Add specific true values if needed (e.g., from a previous estimation)
    # true_measurement_params['my_param'] = 0.5


    true_params_dict = {
        'Phi_list': Phi_list_true,
        'Sigma_cycles': Sigma_cycles_true,
        'Sigma_trends_sim': Sigma_trends_sim_true,
        'measurement_params': true_measurement_params,
    }

    return true_params_dict


# --- Main Script ---
yaml_file_path = 'bvar_stationary_sim.yml' # Use the new simulation YAML

# Set working directory to script location (optional, helps with file paths)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print("Current Working Directory:", os.getcwd())


# --- 1. Load Configuration and Parse ---
print("Loading configuration and parsing...")
try:
    config_data = load_config_and_parse(yaml_file_path)
    print("Configuration loaded and parsed.")

except Exception as e:
    print(f"Error loading or parsing configuration: {e}")
    # sys.exit(1) # Exit if config loading fails

# Extract dimensions and names from parsed config
observable_names = config_data['variables']['observable_names']
trend_var_names = config_data['variables']['trend_names']
stationary_var_names = config_data['variables']['stationary_var_names']
k_endog = len(observable_names)
k_trends = len(trend_var_names)
k_stationary = len(stationary_var_names)
p = config_data['var_order']
k_states = k_trends + k_stationary * p

# Identify trends with shocks defined in YAML config
trend_shocks_spec = config_data['trend_shocks'].get('trend_shocks', {})
trend_names_with_shocks_in_config = [name for name in trend_var_names if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and 'distribution' in trend_shocks_spec[name]]
n_trend_shocks_config = len(trend_names_with_shocks_in_config)
n_shocks_model = n_trend_shocks_config + k_stationary # Number of shocks in the model

# --- 2. Define True Parameters and Simulate Data ---
print("\nDefining true parameters and simulating data...")
key = random.PRNGKey(0)
key_true_params, key_sim, key_mcmc, key_smooth_draws = random.split(key, 4)

T_sim = 100 # Number of time steps for simulation

# Define true parameters based on the config structure (using the function defined above)
true_params = define_true_params(config_data) # This now includes Phi_list, Sigma_cycles, Sigma_trends_sim, measurement_params


# Simulate data using the true parameters and the state-space structure derived from config
print(f"Simulating {T_sim} steps with true parameters...")
# Simulate function now takes individual components of true_params
y_simulated_jax, true_states_sim, true_cycles_sim, true_trends_sim = simulate_bvar_with_trends_jax(
    key_sim,
    T_sim,
    config_data, # Pass config data for dimensions/structure
    true_params['Phi_list'], # True Phi_list
    true_params['Sigma_cycles'], # True Sigma_cycles
    true_params['Sigma_trends_sim'], # True Sigma_trends_sim
    true_params['measurement_params'], # True measurement_params dict
)
print("Data simulation complete.")
print(f"Simulated data shape: {y_simulated_jax.shape}")
print(f"True states shape: {true_states_sim.shape}")
print(f"True trends shape: {true_trends_sim.shape}")
print(f"True cycles shape: {true_cycles_sim.shape}")

# Create a dummy pandas DataFrame for plotting with dates
dummy_dates = pd.period_range(start='1800Q1', periods=T_sim, freq='Q').to_timestamp()
y_simulated_pd = pd.DataFrame(y_simulated_jax, index=dummy_dates, columns=observable_names)


# --- Identify static NaN handling parameters (for simulated data, should be none) ---
valid_obs_mask_cols = jnp.any(jnp.isfinite(y_simulated_jax), axis=0)
static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
static_n_obs_actual = static_valid_obs_idx.shape[0]

if static_n_obs_actual == 0:
    print("Warning: Simulated data resulted in no observed series.")
    print("MCMC cannot be run. Exiting.")
    # sys.exit(1)


# --- 3. Setup and Run MCMC ---
print("\nSetting up MCMC...")

# Extract necessary config pieces for the model (already loaded and parsed)
# The model function now takes the full config_data dict and uses parsed structures within it.
# It also takes the raw equations and initial conditions for consistency with signature,
# but primarily uses the parsed versions from config_data.
model_args = {
    'y': y_simulated_jax,
    'config_data': config_data, # Pass the full config_data dict
    'static_valid_obs_idx': static_valid_obs_idx,
    'static_n_obs_actual': static_n_obs_actual,
    # Explicitly pass these for clarity, though they are in config_data
    'trend_var_names': trend_var_names,
    'stationary_var_names': stationary_var_names,
    'observable_names': observable_names,
    'model_eqs': config_data['model_equations'], # Raw equations
    'initial_conds': config_data['initial_conditions'], # Raw initial conditions
    'stationary_prior_config': config_data['stationary_prior'],
    'trend_shocks_config': config_data['trend_shocks'],
    'measurement_params_config': config_data.get('parameters', {}).get('measurement', [])
}


# Use NumPyro's default init_to_sample
init_strategy = numpyro.infer.init_to_sample()
# Or initialize near true values for debugging (requires mapping true_params to sampled_param_names)
# This mapping is complex due to A vs Phi, variances vs Sigma, etc.
# For now, stick to init_to_sample.


# Define kernel and MCMC
kernel = NUTS(
    model=numpyro_bvar_stationary_model,
    init_strategy=init_strategy,
    # Removed: allow_partial_steps=False # Removed in previous fix
)

num_warmup = 50
num_samples = 100
num_chains = 2 # Run multiple chains

print(f"Running MCMC with {num_warmup} warmup steps and {num_samples} sampling steps per chain ({num_chains} chains)...")
start_time_mcmc = time.time()

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

# Run MCMC with the prepared arguments
mcmc.run(key_mcmc, **model_args)

end_time_mcmc = time.time()
print(f"MCMC completed in {end_time_mcmc - start_time_mcmc:.2f} seconds.")

mcmc.print_summary()

# Get posterior samples
posterior_samples = mcmc.get_samples()


# --- 4. Use the Single-Draw Simulation Smoother Routine ---
print("\nRunning single-draw simulation smoother with posterior mean parameters...")

# Get posterior mean for sampled parameters
# Collect names of parameters sampled in the model (`numpyro.sample` names)
# This requires knowing the sample names from var_ss_model.py
sampled_param_names = [
    'A_diag', # VAR A matrix diagonal
]
# Add A_offdiag only if num_off_diag > 0
num_off_diag_config = config_data['num_off_diag']
if num_off_diag_config > 0:
    sampled_param_names.append('A_offdiag')

# Add stationary correlation Cholesky only if k_stationary > 1
k_stationary = len(config_data['variables']['stationary_var_names'])
if k_stationary > 1:
    sampled_param_names.append('stationary_chol')


# Add stationary variances by name (all stationary variables in config)
stationary_shocks_spec = config_data['stationary_prior'].get('stationary_shocks', {})
stationary_var_names_config = config_data['variables']['stationary_var_names']
# Only add if they have a shock specification
sampled_param_names.extend([f'stationary_var_{name}' for name in stationary_var_names_config if name in stationary_shocks_spec])


# Add trend variances by name (only trends with shocks specified in config)
trend_shocks_spec = config_data['trend_shocks'].get('trend_shocks', {})
trend_var_names_config = config_data['variables']['trend_names']
trend_names_with_shocks_in_config = [name for name in trend_var_names_config if name in trend_shocks_spec and isinstance(trend_shocks_spec[name], dict) and 'distribution' in trend_shocks_spec[name]]
sampled_param_names.extend([f'trend_var_{name}' for name in trend_names_with_shocks_in_config])

# Add measurement parameters by name
measurement_params_config = config_data.get('parameters', {}).get('measurement', [])
sampled_param_names.extend([p['name'] for p in measurement_params_config])


posterior_mean_params = {
    name: jnp.mean(posterior_samples[name], axis=0)
    for name in sampled_param_names
    if name in posterior_samples # Handle case where a param might not be sampled (e.g., k_stat=0)
}


# Run the single-draw smoother routine with the posterior mean parameters
key_smooth_draws = random.PRNGKey(2) # Use a new key for the draws
num_sim_draws = 100 # Number of simulation draws for smoother band
print(f"Running simulation smoother for {num_sim_draws} draws...")

start_time_smooth_draws = time.time()

# Get static arguments needed by the JITted smoother function from config_data
# These are the arguments that must be marked in run_simulation_smoother_single_params_jit's static_argnames
static_smoother_args = {
    'static_k_endog': k_endog,
    'static_k_trends': k_trends,
    'static_k_stationary': k_stationary,
    'static_p': p,
    'static_k_states': k_states,
    'static_n_trend_shocks': n_trend_shocks_config, # Use config-derived count
    'static_n_shocks_state': n_shocks_model,       # Use model-derived shock count
    'static_num_off_diag': config_data['num_off_diag'],
    'static_off_diag_rows': config_data['static_off_diag_indices'][0],
    'static_off_diag_cols': config_data['static_off_diag_indices'][1],
    'static_valid_obs_idx': static_valid_obs_idx,
    'static_n_obs_actual': static_n_obs_actual,
    'model_eqs_parsed': config_data['model_equations_parsed'],
    'initial_conds_parsed': config_data['initial_conditions_parsed'],
    'trend_names_with_shocks': trend_names_with_shocks_in_config,
    'stationary_var_names': stationary_var_names,
    'trend_var_names': trend_var_names,
    'measurement_params_config': measurement_params_config,
    'num_draws': num_sim_draws,
}


# Call the JITted function, passing dynamic arguments first, then static arguments using **
smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
    posterior_mean_params, # Dynamic
    y_simulated_jax, # Dynamic
    key_smooth_draws, # Dynamic
    **static_smoother_args # All Static arguments
)

end_time_smooth_draws = time.time()
print(f"Simulation smoother draws completed in {end_time_smooth_draws - start_time_smooth_draws:.2f} seconds.")


# Process simulation results (mean, median, all_draws)
if num_sim_draws > 1:
    mean_sim_states, median_sim_states, all_sim_draws = simulation_results
    print(f"Smoothed states (simulated data, posterior mean params) shape: {smoothed_states_original.shape}")
    print(f"Mean simulated states shape: {mean_sim_states.shape}")
    print(f"Median simulated states shape: {median_sim_states.shape}")
    print(f"All simulation draws shape: {all_sim_draws.shape}")

    # Check for NaNs in results
    print(f"NaNs in smoothed_states_original: {jnp.any(jnp.isnan(smoothed_states_original))}")
    print(f"NaNs in mean_sim_states: {jnp.any(jnp.isnan(mean_sim_states))}")
    print(f"NaNs in median_sim_states: {jnp.any(jnp.isnan(median_sim_states))}")
    print(f"NaNs in all_sim_draws: {jnp.any(jnp.isnan(all_sim_draws))}")


    # --- 5. Plotting Comparison with True States ---
    print("\nGenerating comparison plots (smoothed/simulated vs. true states)...")

    # Get state names for plotting, consistent with state vector [trends, stationary_t, ...]
    state_names = trend_var_names + [
         f"{name}_t_minus_{lag}" if lag > 0 else name
         for lag in range(p) for name in stationary_var_names
    ]
    # Adjust names for p=1: [trend_vars, stationary_t]
    if p == 1:
        state_names = trend_var_names + stationary_var_names


    try:
        # Plotting comparison for trends and stationary cycles (first k_trends + k_stationary states)
        # For p=1, these are the first k_trends+k_stationary states.
        num_states_to_plot = k_trends + k_stationary # Plot all trend and current cycle states

        fig, axes = plt.subplots(num_states_to_plot, 1, figsize=(12, 3 * num_states_to_plot), sharex=True)
        if num_states_to_plot == 1:
            axes = [axes]

        dates = y_simulated_pd.index # Use pandas index for dates

        # Set up date formatting
        formatter = mdates.DateFormatter('%Y')
        # locator = mdates.YearLocator(2) # Optional: major ticks every 2 years

        for i in range(num_states_to_plot):
            ax = axes[i]
            state_name = state_names[i]

            # Determine if it's a trend or cycle state and get the corresponding true path
            if i < k_trends:
                 true_path = true_trends_sim[:, i]
                 component_type = "True Trend"
            elif i < k_trends + k_stationary:
                 true_path = true_cycles_sim[:, i - k_trends] # Index into true_cycles_sim
                 component_type = "True Cycle"
            else:
                 # Handle lagged states if p > 1, which are just shifted cycles
                 # This example is for p=1, so this block is not needed
                 pass # Placeholder for p > 1 logic

            # Plot true state path
            ax.plot(dates, true_path, label=component_type, color='black', linestyle='-', linewidth=2, alpha=0.7)


            # Plot smoothed state mean
            ax.plot(dates, smoothed_states_original[:, i], label='Smoothed State Mean (Est)', color='blue', linestyle='--')

            # Plot mean and median simulated states
            ax.plot(dates, mean_sim_states[:, i], label='Mean Sim State (Est)', color='red', linestyle=':')
            # ax.plot(dates, median_sim_states[:, i], label='Median Sim State (Est)', color='green', linestyle='-.') # Often close to mean

            # Plot simulation smoother band (e.g., 90% HDI)
            # Calculate HDI for each state path from all_sim_draws
            try:
                 # Need to use arviz.hdi, requires dataarray or similar shape
                 # Convert the draws for this state to a DataArray
                 # az.wrap expects shape (chain, draw, *data_dims)
                 # all_sim_draws[:, :, i] is (num_draws, T_sim).
                 # Need to split num_draws back into (num_chains, num_samples_per_chain)
                 num_chains_plot = mcmc.num_chains # Get number of chains from MCMC
                 num_samples_plot = num_sim_draws // num_chains_plot # Assumes num_draws is divisible by num_chains
                 if num_sim_draws % num_chains_plot != 0:
                     print("Warning: Total smoother draws not divisible by MCMC chains. HDI plotting may be inaccurate.")
                     num_samples_plot = num_sim_draws
                     num_chains_plot = 1 # Treat as single chain for wrapping

                 state_draws_reshaped = all_sim_draws[:, :, i].reshape(num_chains_plot, num_samples_plot, T_sim)

                 state_draws_da = az.wrap(state_draws_reshaped, dims=['chain', 'draw', 'time'])
                 hdi_sim = az.hdi(state_draws_da, hdi_prob=0.90, skipna=True) # skipna=True if NaNs possible

                 # hdi_sim will have dimensions (time, hdi). Plot lower and higher bands.
                 ax.fill_between(dates,
                                hdi_sim.sel(hdi='lower').values,
                                hdi_sim.sel(hdi='higher').values,
                                color='red', alpha=0.2, label='90% Sim Smoother Band')

            except Exception as hdi_e:
                 print(f"Warning: Could not compute/plot HDI for state {state_name}: {hdi_e}")
                 # Fallback: just plot min/max range (less informative)
                 min_draw = jnp.min(all_sim_draws[:, :, i], axis=0)
                 max_draw = jnp.max(all_sim_draws[:, :, i], axis=0)
                 ax.fill_between(dates, min_draw, max_draw, color='red', alpha=0.1, label='Sim Smoother Min/Max')


            ax.set_title(f'True vs Estimated State: {state_name}')
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.xaxis.set_major_formatter(formatter)
            # ax.xaxis.set_major_locator(locator)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


        plt.tight_layout()
        plt.show()

        # Optional: Plot observed data vs. true observed data (should be identical)
        # and vs. fitted/smoothed values (from C @ x_smooth)
        print("\nGenerating simulated observed data vs fitted values plot (optional)...")
        try:
            # Get the mean C matrix from posterior samples
            # C_comp is deterministic in the model, so average over chain and draw
            mean_C_comp = jnp.mean(posterior_samples['C_comp'], axis=(0, 1)) # Shape (k_endog, k_states)

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

                 # Plot original simulated observed data
                 ax.plot(dates, y_simulated_pd[obs_name], label='Simulated Observed', color='gray', alpha=0.7)

                 # Plot fitted/smoothed values
                 ax.plot(dates, fitted_values[:, i], label='Fitted (C @ Smoothed Mean)', color='red', linestyle='--')

                 ax.set_title(f'Simulated Observed vs Fitted: {obs_name}')
                 ax.legend(fontsize=8)
                 ax.grid(True)
                 ax.xaxis.set_major_formatter(formatter)
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

     # Plotting comparison for single draw (first few states)
     try:
        # Get state names for plotting, consistent with state vector [trends, stationary_t, ...]
        state_names = trend_var_names + [
             f"{name}_t_minus_{lag}" if lag > 0 else name
             for lag in range(p) for name in stationary_var_names
        ]
        if p == 1:
             state_names = trend_var_names + stationary_var_names

        num_states_to_plot = k_trends + k_stationary # Plot all trend and current cycle states
        fig, axes = plt.subplots(num_states_to_plot, 1, figsize=(12, 3 * num_states_to_plot), sharex=True)
        if num_states_to_plot == 1:
            axes = [axes]

        dates = y_simulated_pd.index

        formatter = mdates.DateFormatter('%Y')

        for i in range(num_states_to_plot):
            ax = axes[i]
            state_name = state_names[i]

            # Determine if it's a trend or cycle state and get the corresponding true path
            if i < k_trends:
                 true_path = true_trends_sim[:, i]
                 component_type = "True Trend"
            elif i < k_trends + k_stationary:
                 true_path = true_cycles_sim[:, i - k_trends] # Index into true_cycles_sim
                 component_type = "True Cycle"
            else:
                 pass # Lagged states for p > 1

            ax.plot(dates, true_path, label=component_type, color='black', linestyle='-', linewidth=2, alpha=0.7)
            ax.plot(dates, smoothed_states_original[:, i], label='Smoothed State Mean (Est)', color='blue', linestyle='--')
            ax.plot(dates, single_sim_draw[:, i], label='Single Simulated State Draw (Est)', color='red', linestyle=':')

            ax.set_title(f'True vs Estimated State: {state_name}')
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

     except Exception as e:
        print(f"Error during plotting single draw: {e}")


print("\nScript finished.")