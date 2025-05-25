# --- estimate_var_ss.py ---
# This script ties everything together: generate data, run MCMC, and use the single-draw smoother.

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import print_summary
import arviz as az
import matplotlib.pyplot as plt
import time

# Assuming utils directory is in the Python path
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax
from utils.Kalman_filter_jax import simulate_state_space # Use the simulation from KF file
from var_ss_model import numpyro_var_ss_model # Import the model
from run_single_draw import run_simulation_smoother_single_params_jit # Import the single-draw smoother routine

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
# Configure JAX to use CPU (optional, but might be faster for small models/MCMC)
# jax.config.update("jax_platforms", "cpu")


# --- 1. Generate Synthetic Data (Example) ---
print("Generating synthetic data...")
key = random.PRNGKey(0)
key_sim, key_mcmc, key_smooth_draws = random.split(key, 3)

m = 3 # Number of variables
p = 2 # VAR order
T_sim = 200 # Number of time steps for simulation
full_obs_dim = m # Assume we observe exactly the m VAR variables

# True VAR parameters (ensure stationarity for simulation)
# Hand-pick simple stable parameters
Phi_1_true = jnp.eye(m) * 0.5
Phi_2_true = jnp.eye(m) * 0.1
# Ensure companion matrix eigenvalues are < 1
Phi_list_true = [Phi_1_true, Phi_2_true]
T_comp_true = create_companion_matrix_jax(Phi_list_true, p, m)
true_eigenvalues = jnp.linalg.eigvals(T_comp_true)
print(f"True companion matrix eigenvalues: {jnp.abs(true_eigenvalues)}")
# If max abs eigenvalue is close to 1, pick smaller coefficients.
# e.g. Phi_1_true = jnp.eye(m) * 0.3, Phi_2_true = jnp.eye(m) * 0.05

# True Shock Covariance (Sigma) for VAR noise
L_sigma_true = jnp.eye(m) * jnp.array([0.1, 0.2, 0.3]) # Example diagonal Sigma
Sigma_true = L_sigma_true @ L_sigma_true.T
R_comp_true = jnp.vstack([L_sigma_true, jnp.zeros((m * (p - 1), m), dtype=Sigma_true.dtype)])

# True Observation Noise Covariance (H) - for measurement error
# For simplicity, assume diagonal H
L_H_true = jnp.eye(full_obs_dim) * jnp.array([0.01, 0.01, 0.01]) # Small measurement error
H_full_true = L_H_true @ L_H_true.T

# Observation matrix C_true: maps state [y_t', ...] to observations [y_t']
C_for_kf_true = jnp.vstack([
    jnp.hstack([jnp.eye(m, dtype=_DEFAULT_DTYPE), jnp.zeros((m, m * (p - 1)), dtype=_DEFAULT_DTYPE)]),
    jnp.zeros((full_obs_dim - m, m * p), dtype=_DEFAULT_DTYPE) # Assume other observed series (if any) map to zero state
])

# Calculate true stationary initial state distribution
# This requires the Q matrix for the companion state space
Q_comp_true = R_comp_true @ R_comp_true.T
Q_comp_true = (Q_comp_true + Q_comp_true.T) / 2.0
Q_comp_true_reg = Q_comp_true + 1e-8 * jnp.eye(m*p)

# Ensure T_comp_true is stable before solving Lyapunov
true_T_is_stable = jnp.max(jnp.abs(true_eigenvalues)) < (1.0 - 1e-6)

init_P_comp_true = jnp.eye(m*p) * 1e6
if true_T_is_stable:
    try:
         init_P_comp_true = jsl.solve_discrete_lyapunov(T_comp_true.T, Q_comp_true_reg)
         init_P_comp_true = (init_P_comp_true + init_P_comp_true.T) / 2.0 # Ensure symmetry
    except Exception:
         print("Warning: Could not solve Lyapunov for true params, using large default init_P.")
         init_P_comp_true = jnp.eye(m*p) * 1e6
else:
    print("Warning: True companion matrix is unstable, using large default init_P.")
    init_P_comp_true = jnp.eye(m*p) * 1e6

init_x_comp_true = jnp.zeros(m*p) # Assume zero-mean VAR state


# Simulate data using the companion state space form
print(f"Simulating {T_sim} steps...")
# simulate_state_space takes P_aug (T), R_aug, Omega (C), H_obs, init_x, init_P
# R_aug is shock impact matrix, not shock cov. R_aug is R_comp_true (mp, m)
# Omega is observation matrix C_true (full_obs_dim, mp)
# H_obs is observation noise cov H_full_true (full_obs_dim, full_obs_dim)
# init_x, init_P are init_x_comp_true, init_P_comp_true (mp, mp)
# num_steps is T_sim
true_states, y_observed = simulate_state_space(
    T_comp_true,
    R_comp_true, # R_aug
    C_for_kf_true, # Omega
    H_full_true, # H_obs
    init_x_comp_true,
    init_P_comp_true,
    key_sim,
    T_sim
)
print("Data simulation complete.")
print(f"Simulated data shape: {y_observed.shape}")

# Optional: Introduce NaNs in observed data for testing the filter's static NaN handling
# Let's make one full observation series NaN for this example
if full_obs_dim > m:
    # If we simulated more obs dims than m VAR variables
    y_observed = y_observed.at[:, m].set(jnp.nan) # Make the (m+1)-th series all NaN
    print(f"Introduced NaNs in column {m} of observed data.")
elif m > 1:
     # If full_obs_dim == m, let's make one VAR series all NaN (if m > 1)
     y_observed = y_observed.at[:, 0].set(jnp.nan) # Make first VAR series all NaN
     print("Introduced NaNs in first column of observed data.")


# --- 2. Setup and Run MCMC ---
print("Setting up MCMC...")

# Identify static args for the model based on the potentially NaN data
# These define which observation series are *always* potentially observed
valid_obs_mask_cols = jnp.any(jnp.isfinite(y_observed), axis=0)
static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
static_n_obs_actual = static_valid_obs_idx.shape[0]

# C and H sliced to the actually observed subset dimensions (used *inside* KF for update)
# Need to compute these *outside* the model function and pass them as non-sampled parameters
# C_for_kf is computed INSIDE the model based on sampled params, shape (full_obs_dim, mp)
# H_full is computed INSIDE the model based on sampled params, shape (full_obs_dim, full_obs_dim)
# The KF needs these full matrices AND the static sliced ones derived from them.
# We need to pass static_valid_obs_idx and static_n_obs_actual as arguments to the model.
# The sliced C_obs, H_obs, I_obs can be computed inside the model using these static indices.

# The model function takes y, m, p. It also needs full_obs_dim
# and the static observation indices/count.
# It should also potentially take initial values if desired.

# Add static arguments to the model function signature
# `numpyro_var_ss_model(y, m, p, full_obs_dim, static_valid_obs_idx, static_n_obs_actual)`
# We need to pass these when calling the model.

# Check if there are any observed series left after filtering out all-NaN columns
if static_n_obs_actual == 0:
    print("Warning: No observation series are observed after removing all-NaN columns.")
    print("Cannot run MCMC. Exiting.")
    # Handle this case: maybe raise an error or return early
    exit() # Or sys.exit()


# Initializing MCMC - finding a good initial state can be tricky.
# Using init_to_value with plausible values (e.g., identity/zeros) is often helpful.
# A simple init for 'a_flat' could be zeros (identity Phi corresponds to A=0).
# A simple init for Choleskys could be small positive values on diagonal, zeros elsewhere.
initial_a_flat = jnp.zeros(p * m * m, dtype=_DEFAULT_DTYPE)
initial_sigma_chol_packed = jnp.ones(m * (m + 1) // 2, dtype=_DEFAULT_DTYPE) * 0.1 # Small diagonal
initial_h_chol_packed = jnp.ones(full_obs_dim * (full_obs_dim + 1) // 2, dtype=_DEFAULT_DTYPE) * 0.01 # Small diagonal

initial_params = {
    'a_flat': initial_a_flat,
    'sigma_chol_packed': initial_sigma_chol_packed,
    'h_chol_packed': initial_h_chol_packed,
}

# It's generally better to use default initialization strategies (init_to_uniform, init_to_median, init_to_sample)
# unless you have known good starting points. init_to_sample is often robust.
# init_strategy = init_to_sample()

# Or explicitly provide initial values if you know good ones (e.g., from optimization)
# init_strategy = init_to_value(values=initial_params)


# Define kernel and MCMC
# Using default init_to_sample for robustness
kernel = NUTS(
    model=numpyro_var_ss_model,
    # init_strategy=init_strategy,
    allow_partial_steps=False # Keep False for correctness with discrete parameters if any (not here)
)

num_warmup = 500
num_samples = 1000
num_chains = 2 # Run multiple chains

print(f"Running MCMC with {num_warmup} warmup steps and {num_samples} sampling steps per chain ({num_chains} chains)...")
start_time_mcmc = time.time()

mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

# Run MCMC, passing the data and static parameters to the model
mcmc.run(
    key_mcmc,
    y=y_observed,
    m=m,
    p=p,
    full_obs_dim=full_obs_dim, # Pass full observation dimension
    static_valid_obs_idx=static_valid_obs_idx, # Pass static valid indices
    static_n_obs_actual=static_n_obs_actual # Pass static number of actual observations
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

# Get posterior mean for parameters
posterior_mean_params = {
    name: jnp.mean(samples, axis=0)
    for name, samples in posterior_samples.items()
    if name in ['a_flat', 'sigma_chol_packed', 'h_chol_packed'] # Only get the sampled parameters
}

# Check if the mean parameters are valid for the stationary transformation and Lyapunov solve
# This is important because the mean of parameters might not correspond to a stationary model.
# The single-draw function includes checks, but a heads-up is good.
try:
    L_sigma_mean = jsl.build_triangle(posterior_mean_params['sigma_chol_packed'], m, m, False)
    Sigma_mean = L_sigma_mean @ L_sigma_mean.T
    A_list_mean = [jnp.reshape(posterior_mean_params['a_flat'][i * m * m : (i + 1) * m * m], (m, m)) for i in range(p)]
    phi_list_mean, _ = make_stationary_var_transformation_jax(Sigma_mean, A_list_mean, m, p)
    is_mean_params_stationary = check_stationarity_jax(phi_list_mean, m, p)

    if not is_mean_params_stationary:
        print("\nWarning: Posterior mean parameters result in a non-stationary VAR process.")
        print("The simulation smoother results using these parameters might be invalid.")

    T_comp_mean = create_companion_matrix_jax(phi_list_mean, p, m)
    Q_comp_mean = jnp.vstack([Sigma_mean, jnp.zeros((m*(p-1), m))]) @ jnp.vstack([Sigma_mean, jnp.zeros((m*(p-1), m))]).T
    Q_comp_mean_reg = Q_comp_mean + 1e-8 * jnp.eye(m*p)
    try:
        init_P_mean = jsl.solve_discrete_lyapunov(T_comp_mean.T, Q_comp_mean_reg)
        init_P_mean_valid = jnp.all(jnp.isfinite(init_P_mean)) and jnp.max(jnp.abs(init_P_mean)) < 1e5 # Check for non-default value
        if not init_P_mean_valid:
             print("Warning: Stationary initial covariance computation failed for posterior mean parameters.")
    except Exception:
         print("Warning: Stationary initial covariance computation failed (exception) for posterior mean parameters.")
         init_P_mean_valid = False

except Exception as e:
    print(f"\nWarning: Could not perform stationary check or Lyapunov solve for posterior mean parameters: {e}")
    is_mean_params_stationary = False
    init_P_mean_valid = False


# Run the single-draw smoother routine with the posterior mean parameters
# We need a JAX random key for the simulation draws
num_sim_draws = 10 # Number of simulation draws for this example
print(f"Running simulation smoother for {num_sim_draws} draws...")

start_time_smooth_draws = time.time()

smoothed_states_original, simulation_results = run_simulation_smoother_single_params_jit(
    posterior_mean_params,
    y_observed,
    m,
    p,
    key_smooth_draws, # Use a new key for the draws
    num_draws=num_sim_draws # Specify number of draws
)

end_time_smooth_draws = time.time()
print(f"Simulation smoother draws completed in {end_time_smooth_draws - start_time_smooth_draws:.2f} seconds.")

# simulation_results will be a tuple (mean_draws, median_draws, all_draws) if num_draws > 1
# or just the single draw if num_draws == 1
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
    try:
        # Plot original data and smoothed states for the first VAR variable (state index 0)
        plt.figure(figsize=(12, 6))
        # Original data for the first VAR variable (first m columns assumed to be VAR vars)
        # Check if the first VAR variable series was made NaN
        y_plot_idx = 0 # Index of the VAR variable to plot (0 to m-1)
        if full_obs_dim <= y_plot_idx:
             print(f"Cannot plot VAR variable {y_plot_idx}: full_obs_dim is {full_obs_dim}")
        else:
             plt.plot(y_observed[:, y_plot_idx], label=f'Observed VAR {y_plot_idx+1}', color='gray', alpha=0.6)
             # If data was made NaN, indicate it
             if jnp.all(jnp.isnan(y_observed[:, y_plot_idx])):
                  print(f"Note: Observed data for VAR {y_plot_idx+1} is all NaN in the generated data.")


        # Smoothed states: state vector is [y_t', y_{t-1}', ...]. First m elements are y_t
        plt.plot(smoothed_states_original[:, y_plot_idx], label=f'Smoothed State Mean (VAR {y_plot_idx+1})', color='blue')

        # Plot mean and median simulated states
        plt.plot(mean_sim_states[:, y_plot_idx], label=f'Mean Simulated State (VAR {y_plot_idx+1})', color='red', linestyle='--')
        plt.plot(median_sim_states[:, y_plot_idx], label=f'Median Simulated State (VAR {y_plot_idx+1})', color='green', linestyle=':')

        # Plot a few individual draws (optional, can be noisy)
        # for i in range(min(num_sim_draws, 5)): # Plot up to 5 draws
        #      plt.plot(all_sim_draws[i, :, y_plot_idx], color='purple', alpha=0.1)

        plt.title(f'Smoothed and Simulated States for VAR Variable {y_plot_idx+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error during plotting: {e}")

elif num_sim_draws == 1:
     single_sim_draw = simulation_results
     print(f"Single simulation draw shape: {single_sim_draw.shape}")
     print(f"NaNs in single_sim_draw: {jnp.any(jnp.isnan(single_sim_draw))}")

     # Plotting for single draw
     try:
        plt.figure(figsize=(12, 6))
        y_plot_idx = 0 # Index of the VAR variable to plot (0 to m-1)
        if full_obs_dim > y_plot_idx:
             plt.plot(y_observed[:, y_plot_idx], label=f'Observed VAR {y_plot_idx+1}', color='gray', alpha=0.6)
        plt.plot(smoothed_states_original[:, y_plot_idx], label=f'Smoothed State Mean (VAR {y_plot_idx+1})', color='blue')
        plt.plot(single_sim_draw[:, y_plot_idx], label=f'Single Simulated State Draw (VAR {y_plot_idx+1})', color='red', linestyle='--')

        plt.title(f'Smoothed State and Single Simulated Draw for VAR Variable {y_plot_idx+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
     except Exception as e:
        print(f"Error during plotting single draw: {e}")

print("\nScript finished.")