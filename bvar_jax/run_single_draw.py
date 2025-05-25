# --- run_single_draw.py (Final Corrected Version - Pass All Static Args Explicitly) ---
# Routine to run the simulation smoother for a single parameter vector
# based on the BVAR with trends state space model.

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
# REMOVE: import numpyro # Remove numpyro import
from typing import Dict, Tuple, Union, List, Any, Optional, Sequence

# Configure JAX as requested
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") # Use CPU as requested
_DEFAULT_DTYPE = jnp.float64 # Define default dtype

# Add a small jitter consistent with model/filter
_DRAW_JITTER = 1e-8

# Assuming utils directory is in the Python path
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax # Removed check_stationarity_jax
from utils.Kalman_filter_jax import KalmanFilter # Use the standard KF for initial smooth
from utils.hybrid_dk_smoother import HybridDKSimulationSmoother # Use the DK smoother class

# Import the parsing logic used in the model setup
# Assuming these are in var_ss_model.py or accessible via import
from var_ss_model import _parse_equation_jax, parse_initial_state_config, _get_off_diagonal_indices # Import helpers


# run_simulation_smoother_single_params will be JITted, so its static_argnames must be declared.
# All arguments that are dimensions, indices, counts, or config structures are static.
# Only 'params', 'y', 'key' are dynamic.

def run_simulation_smoother_single_params(
    params: Dict[str, jax.Array], # Dictionary of parameter values (DYNAMIC)
    y: jax.Array, # Observed data (DYNAMIC)
    key: jax.random.PRNGKey, # JAX random key (DYNAMIC)
    # Static arguments derived from config and data (MARKED AS STATIC):
    static_k_endog: int, # k_endog = len(observable_names)
    static_k_trends: int, # k_trends = len(trend_var_names)
    static_k_stationary: int, # k_stationary = len(stationary_var_names)
    static_p: int, # p = var_order
    static_k_states: int, # k_states = static_k_trends + static_k_stationary * static_p
    static_n_trend_shocks: int, # n_trend_shocks = len(trend_names_with_shocks)
    static_n_shocks_state: int, # n_shocks_state = static_n_trend_shocks + static_k_stationary
    static_num_off_diag: int, # num_off_diag = k_stationary * (k_stationary - 1)
    static_off_diag_rows: jax.Array, # indices for off-diagonal A
    static_off_diag_cols: jax.Array, # indices for off-diagonal A
    static_valid_obs_idx: jax.Array, # Indices of observed series
    static_n_obs_actual: int,      # Number of actually observed series
    # Pre-parsed config structures (STATIC, used for string/list lookups):
    model_eqs_parsed: List[Tuple[int, List[Tuple[Optional[str], str, float]]]],
    initial_conds_parsed: Dict[str, Dict[str, float]],
    trend_names_with_shocks: List[str], # List of trend names with shocks
    stationary_var_names: List[str], # List of stationary variable names (for param names)
    trend_var_names: List[str], # List of trend variable names (for state indices)
    measurement_params_config: List[Dict[str, Any]], # List of measurement param specs (for param names)
    num_draws: int = 1, # Number of simulation draws (STATIC)

) -> Tuple[jax.Array, Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]]:
    """
    Runs the simulation smoother using a fixed parameter vector from the BVAR model.

    Args:
        params: Dictionary of parameter values (e.g., posterior mean/median).
        y: Observed data (time_steps, static_k_endog). Can contain NaNs.
        key: JAX random key for simulation draws.
        static_k_endog, etc.: Static dimensions derived from config.
        static_off_diag_rows, static_off_diag_cols: Static indices for A matrix construction.
        static_valid_obs_idx, static_n_obs_actual: Static observation indices and count from y.
        model_eqs_parsed, initial_conds_parsed, etc.: Pre-parsed config structures.
        trend_names_with_shocks, stationary_var_names, etc.: Lists of names.
        measurement_params_config: List of measurement param specs.
        num_draws: Number of simulation draws to perform.

    Returns:
        Tuple: (smoothed_states_original, simulation_results).
        Shapes involve static_k_states and num_draws.
    """
    # --- Reconstruct Parameters from Input `params` ---
    # Reconstruct A_draws (static_p x static_k_stationary x static_k_stationary) using static indices
    if static_k_stationary > 0:
        diag_A = params['A_diag'] # shape (static_p, static_k_stationary)
        if static_num_off_diag > 0:
            offdiag_A_flat = params['A_offdiag'] # shape (static_p, static_num_off_diag)
        else:
             offdiag_A_flat = jnp.empty((static_p, 0), dtype=_DEFAULT_DTYPE)


        A_draws = jnp.zeros((static_p, static_k_stationary, static_k_stationary), dtype=_DEFAULT_DTYPE)
        A_draws = A_draws.at[:, jnp.arange(static_k_stationary), jnp.arange(static_k_stationary)].set(diag_A)
        if static_num_off_diag > 0:
            A_draws = A_draws.at[:, static_off_diag_rows, static_off_diag_cols].set(offdiag_A_flat)
    else: # static_k_stationary == 0
        A_draws = jnp.empty((static_p, 0, 0), dtype=_DEFAULT_DTYPE)


    # Reconstruct stationary variances and stationary_D_sds
    if static_k_stationary > 0:
        stationary_variances_list = []
        for name in stationary_var_names: # Iterate through the list of names
             stationary_variances_list.append(params[f'stationary_var_{name}'])

        stationary_variances = jnp.stack(stationary_variances_list, axis=-1) # shape (static_k_stationary,)
        stationary_D_sds = jnp.diag(jnp.sqrt(jnp.maximum(stationary_variances, 1e-12))) # Ensure non-negative variance before sqrt! (static_k_stationary, static_k_stationary)

        # Reconstruct stationary_chol (Cholesky of correlation)
        if static_k_stationary > 1:
             stationary_chol = params['stationary_chol'] # (static_k_stationary, static_k_stationary)
        elif static_k_stationary == 1:
             stationary_chol = jnp.eye(1, dtype=_DEFAULT_DTYPE) # For 1x1, Cholesky of correlation is 1
        else: # static_k_stationary == 0 - handl
             stationary_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)


        # Reconstruct Sigma_cycles using the specific formula: Sigma_cycles = L_corr @ D_sds @ L_corr.T
        Sigma_cycles = stationary_chol @ stationary_D_sds @ stationary_chol.T
        Sigma_cycles = (Sigma_cycles + Sigma_cycles.T)/2.0 # Ensure symmetry
    else:
        stationary_variances_list = []
        stationary_variances = jnp.empty((0,), dtype=_DEFAULT_DTYPE)
        stationary_D_sds = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        stationary_chol = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)
        Sigma_cycles = jnp.empty((0,0), dtype=_DEFAULT_DTYPE)


    # Reconstruct trend variances and Sigma_trends
    trend_variances_list = []
    for trend_name in trend_names_with_shocks:
        # Get the variance from params dict by name
        trend_variances_list.append(params[f'trend_var_{trend_name}'])

    if static_n_trend_shocks == 0:
        Sigma_trends = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
    else:
        trend_variances = jnp.stack(trend_variances_list, axis=-1) # shape (static_n_trend_shocks,)
        Sigma_trends = jnp.diag(jnp.maximum(trend_variances, 1e-12)) # Ensure non-negative variance! (static_n_trend_shocks, static_n_trend_shocks)
        Sigma_trends = (Sigma_trends + Sigma_trends.T)/2.0 # Ensure symmetry


    # Reconstruct measurement parameters dictionary
    measurement_params_sampled = {
        param_spec['name']: params[param_spec['name']] for param_spec in measurement_params_config
    }

    # --- Reconstruct State-Space Matrices ---

    # 1. Phi_list
    # Only perform transform if static_k_stationary > 0
    if static_k_stationary > 0:
        # Pass A_draws list and Sigma_cycles
        phi_list, _ = make_stationary_var_transformation_jax(Sigma_cycles, [A_draws[i] for i in range(static_p)], static_k_stationary, static_p)
        # Stationarity check removed as per user instruction.
    else: # static_k_stationary == 0
        phi_list = [] # Empty list


    # 2. T matrix
    # T = [ I_{static_k_trends}  0_{static_k_trends, static_k_stationary*static_p} ]
    #     [ 0_{static_k_stationary*static_p, static_k_trends}  AR_comp ]
    T_comp = jnp.zeros((static_k_states, static_k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:static_k_trends, :static_k_trends].set(jnp.eye(static_k_trends, dtype=_DEFAULT_DTYPE)) # Trends block
    if static_k_stationary > 0:
        # AR Companion block using Phi_list
        T_comp = T_comp.at[static_k_trends:static_k_states, static_k_trends:static_k_states].set(create_companion_matrix_jax(phi_list, static_p, static_k_stationary))


    # 3. R matrix (Shock Impact Matrix for State Space)
    # R has shape (static_k_states, static_n_trend_shocks + static_k_stationary).
    # R maps [trend_shocks_std_normal', stationary_shocks_std_normal']' to state dynamics.
    R_comp = jnp.zeros((static_k_states, static_n_shocks_state), dtype=_DEFAULT_DTYPE)

    # Trends with shocks: I_{static_n_trend_shocks} in top-left block, mapping to trend state indices with shocks
    # R[trend_state_idx, shock_idx] = 1.0
    trend_state_indices = {name: i for i, name in enumerate(trend_var_names)} # Need this map again
    trend_shock_mapping_indices = {name: i for i, name in enumerate(trend_names_with_shocks)}
    for trend_name_with_shock in trend_names_with_shocks:
        trend_state_idx = trend_state_indices[trend_name_with_shock]
        shock_idx = trend_shock_mapping_indices[trend_name_with_shock]
        R_comp = R_comp.at[trend_state_idx, shock_idx].set(1.0)

    # Stationary shocks: I_{static_k_stationary} mapping to the *current* stationary state block (indices static_k_trends to static_k_trends+static_k_stationary-1)
    if static_k_stationary > 0:
        R_comp = R_comp.at[static_k_trends:static_k_trends+static_k_stationary, static_n_trend_shocks:].set(jnp.eye(static_k_stationary, dtype=_DEFAULT_DTYPE))

    # 4. Observation Matrix (C)
    # C has shape (static_k_endog, static_k_states). Based on parsed equations and sampled measurement params.
    C_comp = jnp.zeros((static_k_endog, static_k_states), dtype=_DEFAULT_DTYPE)
    # State indices for C matrix (Trends + Current Stationary)
    state_indices_for_C = {name: i for i, name in enumerate(trend_var_names)} # Trends
    state_indices_for_C.update({name: static_k_trends + i for i, name in enumerate(stationary_var_names)}) # Current Stationary

    # Populate C matrix using the parsed structure and sampled measurement parameters
    # model_eqs_parsed is list of (obs_idx, parsed_terms)
    for obs_idx, parsed_terms in model_eqs_parsed:
        for param_name, state_name_in_eq, sign in parsed_terms:
            # State name in equation must map to index in state_indices_for_C
            state_idx = state_indices_for_C[state_name_in_eq] # Map the name in the equation to its index
            if param_name is None:
                # Direct state term
                C_comp = C_comp.at[obs_idx, state_idx].set(sign * 1.0)
            else:
                # Parameter * State term
                param_value = measurement_params_sampled[param_name]
                C_comp = C_comp.at[obs_idx, state_idx].set(sign * param_value)

    # 5. Observation Noise Covariance (H)
    # H is a zero matrix (static_k_endog, static_k_endog) based on PyMC code
    H_comp = jnp.zeros((static_k_endog, static_k_endog), dtype=_DEFAULT_DTYPE)


    # --- Initial State Distribution ---
    # Mean (init_x): Fixed values from parsed initial_conditions (means)
    init_x_comp = jnp.zeros(static_k_states, dtype=_DEFAULT_DTYPE)
    initial_state_parsed = initial_conds_parsed # Use the pre-parsed dict
    # Full state indices including lags for static_p > 1
    state_indices_full = {name: i for i, name in enumerate(trend_var_names)} # Trends 0 to static_k_trends-1
    for lag in range(static_p):
         for i, name in enumerate(stationary_var_names):
              if lag == 0: # Current cycle
                   state_indices_full[name] = static_k_trends + i
              else: # Lagged cycles
                   lagged_state_name = f"{name}_t_minus_{lag}"
                   lagged_state_idx = static_k_trends + lag * static_k_stationary + i
                   state_indices_full[lagged_state_name] = lagged_state_idx

    for state_name_parsed, parsed_config in initial_state_parsed.items():
         if state_name_parsed in state_indices_full:
             init_x_comp = init_x_comp.at[state_indices_full[state_name_parsed]].set(parsed_config['mean'])
         # Note: Initial means for lagged stationary states are implicitly 0 if not listed in initial_conditions.


    # Covariance (init_P): Diagonal matrix from parsed initial_conditions variances, repeated for lags
    init_P_comp_diag_values = jnp.zeros(static_k_states, dtype=_DEFAULT_DTYPE)
    # Populate diagonal P0 based on variances from parsed initial_conditions: states
    for state_name_parsed, parsed_config in initial_state_parsed.items():
         if state_name_parsed in state_indices_full:
             state_idx = state_indices_full[state_name_parsed]
             init_P_comp_diag_values = init_P_comp_diag_values.at[state_idx].set(parsed_config['var'])

    # For lagged stationary states (static_p>1): PyMC sets variances to the same as current state variances.
    if static_p > 1:
         for lag in range(1, static_p):
             for i, name in enumerate(stationary_var_names):
                  current_state_name = name # Name in parsed initial conditions
                  lagged_state_name = f"{name}_t_minus_{lag}" # Name in full state vector
                  if current_state_name in initial_state_parsed and lagged_state_name in state_indices_full:
                       current_var = initial_state_parsed[current_state_name]['var']
                       lagged_idx = state_indices_full[lagged_state_name]
                       init_P_comp_diag_values = init_P_comp_diag_values.at[lagged_idx].set(current_var)


    init_P_comp = jnp.diag(init_P_comp_diag_values) # Shape (static_k_states, static_k_states)

    # Ensure init_P_comp is symmetric and regularized
    init_P_comp = (init_P_comp + init_P_comp.T) / 2.0
    init_P_comp = init_P_comp + _DRAW_JITTER * jnp.eye(static_k_states, dtype=_DEFAULT_DTYPE)

    # Check if initial P is valid (not containing NaNs from parsing or negative vars)
    is_init_P_valid_computed = jnp.all(jnp.isfinite(init_P_comp)) & jnp.all(jnp.diag(init_P_comp) >= 0)


    # --- Prepare Static Args for Kalman Filter ---
    # static_valid_obs_idx, static_n_obs_actual are passed explicitly to this function
    # static_C_obs: C_comp restricted to observed rows
    static_C_obs = C_comp[static_valid_obs_idx, :] # Shape (static_n_obs_actual, static_k_states)

    # static_H_obs: H_comp restricted to observed rows/cols (will be zero matrix)
    static_H_obs = H_comp[static_valid_obs_idx[:, None], static_valid_obs_idx] # Shape (static_n_obs_actual, static_n_obs_actual)

    # static_I_obs: Identity matrix of size static_n_obs_actual
    static_I_obs = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)


    # --- Instantiate and Run Kalman Filter ---
    # Ensure all inputs to KF have the correct dtype
    T_comp = jnp.asarray(T_comp, dtype=_DEFAULT_DTYPE)
    R_comp = jnp.asarray(R_comp, dtype=_DEFAULT_DTYPE)
    C_comp = jnp.asarray(C_comp, dtype=_DEFAULT_DTYPE) # Pass the full C matrix
    H_comp = jnp.asarray(H_comp, dtype=_DEFAULT_DTYPE) # Pass the full H matrix
    init_x_comp = jnp.asarray(init_x_comp, dtype=_DEFAULT_DTYPE)
    init_P_comp = jnp.asarray(init_P_comp, dtype=_DEFAULT_DTYPE)

    # Instantiate KalmanFilter
    kf = KalmanFilter(T_comp, R_comp, C_comp, H_comp, init_x_comp, init_P_comp)

    # Run the filter. Need to pass the full `y` data matrix and the static args.
    filter_results = kf.filter(
        y, # Pass the observation data (DYNAMIC)
        static_valid_obs_idx, # Static arg
        static_n_obs_actual, # Static arg
        static_C_obs, # Derived from static args
        static_H_obs, # Derived from static args
        static_I_obs # Derived from static args
    )

    # Check if initial P is valid (not containing NaNs from parsing or negative vars)
    # We need this check for the simulation smoother path.
    is_init_P_valid_computed = jnp.all(jnp.isfinite(init_P_comp)) & jnp.all(jnp.diag(init_P_comp) >= 0)


    # If any NaNs occurred *before* the Kalman filter (e.g., during matrix construction)
    # Check some key matrices for NaNs as a safeguard for the smoother.
    matrices_to_check = [T_comp, R_comp, C_comp, H_comp, init_x_comp[None, :], init_P_comp] # Add dummy time dim for x
    any_matrix_nan = jnp.array(False)
    for mat in matrices_to_check:
        if mat.size > 0:
             any_matrix_nan |= jnp.any(jnp.isnan(mat))

    # If no observed series, cannot run filter/smoother for the initial smooth
    run_initial_smooth = is_init_P_valid_computed & (~any_matrix_nan) & (y.shape[0] > 0) & (static_n_obs_actual > 0)


    # Define default outputs for the case where initial smooth is skipped
    T_steps = y.shape[0]
    default_smoothed_states = jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE)
    default_sim_results: Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]
    if num_draws == 1:
         default_sim_results = jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE)
    else:
         default_sim_results = (
             jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE), # Mean
             jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE), # Median
             jnp.full((num_draws, T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE) # All draws
         )


    def run_smoother_and_draws_actual(operand):
        # Operand receives: (y_actual, kf_params_actual, key_actual, num_draws_actual)
        # Static arguments (static_k_endog, etc., static_valid_obs_idx, static_n_obs_actual, num_draws)
        # are directly in scope from the outer function parameters.
        y_actual, kf_params_actual, key_actual, num_draws_actual = operand
        T_comp_kf, R_comp_kf, C_comp_kf, H_comp_kf, init_x_comp_kf, init_P_comp_kf = kf_params_actual

        # Derive sliced C, H, I for the KF filter step using static args from outer scope
        static_C_obs_kf = C_comp_kf[static_valid_obs_idx, :] # <--- Uses outer scope static arg
        static_H_obs_kf = H_comp_kf[static_valid_obs_idx[:, None], static_valid_obs_idx] # <--- Uses outer scope static arg
        static_I_obs_kf = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE) # <--- Uses outer scope static arg

        # 1. Run filter on original (potentially NaN) data `y`
        kf_original = KalmanFilter(T_comp_kf, R_comp_kf, C_comp_kf, H_comp_kf, init_x_comp_kf, init_P_comp_kf)
        filter_results_original = kf_original.filter(
            y_actual, # Pass the observation data (DYNAMIC)
            static_valid_obs_idx, # Static arg
            static_n_obs_actual, # Static arg
            static_C_obs_kf,
            static_H_obs_kf,
            static_I_obs_kf
        )

        # Check if filter produced valid results
        filter_ok = jnp.all(jnp.isfinite(filter_results_original['x_filt'])) & \
                    jnp.all(jnp.isfinite(filter_results_original['P_filt'])) & \
                    jnp.all(jnp.isfinite(filter_results_original['log_likelihood_contributions']))

        # 2. Run RTS smoother on original data filter results IF filter was OK
        # Re-instantiate KalmanFilter (or reuse params) for the smoother backend
        kf_smoother = KalmanFilter(T_comp_kf, R_comp_kf, C_comp_kf, H_comp_kf, init_x_comp_kf, init_P_comp_kf)
        smoothed_states_original_means, smoothed_states_original_covs = jax.lax.cond(
            filter_ok,
            lambda res: kf_smoother.smooth(y_actual, filter_results=res), # Pass data and filter results to smooth
            lambda res: (jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE), jnp.full((T_steps, static_k_states, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE)),
            operand=filter_results_original
        )


        # 3. Run Simulation Smoother Draws IF initial smooth was OK
        sim_smoother_draws_results: Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]
        # Default to NaNs if smooth failed
        if num_draws == 1:
            sim_smoother_draws_results = jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE)
        else:
            sim_smoother_draws_results = (
                jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE), # Mean
                jnp.full((T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE), # Median
                jnp.full((num_draws, T_steps, static_k_states), jnp.nan, dtype=_DEFAULT_DTYPE) # All draws
            )

        def run_draws_if_smooth_ok(smoothed_states_means_ok):
            # Instantiate HybridDKSimulationSmoother. It needs the same state-space parameters.
            dk_smoother = HybridDKSimulationSmoother(
                T_comp_kf, R_comp_kf, C_comp_kf, H_comp_kf, init_x_comp_kf, init_P_comp_kf
            )

            # Impute NaNs in original_ys using predicted values from smoothed states
            y_imputed = jnp.where(jnp.isfinite(y_actual), y_actual, jnp.nan)
            y_predicted_from_smooth = smoothed_states_means_ok @ C_comp_kf.T # (T, static_k_states) @ (static_k_states, static_k_endog).T = (T, static_k_endog)
            y_dense_for_draws = jnp.where(jnp.isfinite(y_actual), y_actual, y_predicted_from_smooth)

            # Now call run_smoother_draws with the imputed dense data and the smoothed means
            key_draws = random.split(key_actual)[0] # Take first key from split for multiple draws
            # run_smoother_draws needs the same static arguments as the outer function
            # The HybridDKSimulationSmoother handles its own internal filter/smoother, which uses the
            # C_full, H_full, etc. passed during *its* initialization. It does *not* need the static_obs_args
            # from the original data's filter. It needs original_ys_dense and smoothed_states_original_dense.
            # Its internal filter (_filter_internal) is designed for DENSE data, which y_dense_for_draws is.
            # Its internal smoother (_rts_smoother_backend) works on DENSE filter results.
            # The D-K formula x_draw = x_star + (x_smooth_original - x_smooth_star) uses smoothed_states_original_means (from NaN-handled filter)
            # and x_smooth_star (from dense filter on simulated dense data).
            # So, dk_smoother.run_smoother_draws just needs the imputed data, key, num_draws, and smoothed_states_original_means.
            draws_results = dk_smoother.run_smoother_draws(
                y_dense_for_draws, # Use imputed dense data
                key_draws,
                num_draws_actual,
                smoothed_states_means_ok # Use the correctly smoothed states from original data
            )
            return draws_results

        sim_smoother_draws_results = jax.lax.cond(
            jnp.all(jnp.isfinite(smoothed_states_original_means)), # Only run draws if initial smooth produced finite means
            run_draws_if_smooth_ok,
            lambda smooth_means_failed: sim_smoother_draws_results, # Return default NaN results if initial smooth failed
            operand=smoothed_states_original_means # Pass smoothed means to the conditional function
        )


        return smoothed_states_original_means, sim_smoother_draws_results


    # Execute the main conditional logic
    # Pass static args explicitly to the operand tuple for the conditional function.
    # The conditional function receives (y, kf_params, key, num_draws).
    # Static observation args (static_valid_obs_idx, static_n_obs_actual) and dimensions/indices
    # are already STATIC parameters to the outer function run_simulation_smoother_single_params,
    # so they are available in the nested functions run_smoother_and_draws_actual and run_draws_if_smooth_ok.
    smoothed_states_original_means, sim_smoother_draws_results = jax.lax.cond(
        run_initial_smooth, # Check if parameters are valid and data is present for initial smooth
        run_smoother_and_draws_actual, # This branch is run if True
        lambda op: (default_smoothed_states, default_sim_results), # This branch is run if False
        # Operand for run_smoother_and_draws_actual:
        operand=(y, (T_comp, R_comp, C_comp, H_comp, init_x_comp, init_P_comp), key, num_draws)
    )


    return smoothed_states_original_means, sim_smoother_draws_results


# JIT compile the function with static arguments.
# All arguments that are dimensions, indices, counts, or config structures are static.
# Only 'params', 'y', 'key' are dynamic.
run_simulation_smoother_single_params_jit = jax.jit(
    run_simulation_smoother_single_params,
    static_argnames=(
        'static_k_endog', 'static_k_trends', 'static_k_stationary', 'static_p', 'static_k_states',
        'static_n_trend_shocks', 'static_n_shocks_state', 'static_num_off_diag',
        'static_off_diag_rows', 'static_off_diag_cols',
        'static_valid_obs_idx', 'static_n_obs_actual',
        'model_eqs_parsed', 'initial_conds_parsed',
        'trend_names_with_shocks', 'stationary_var_names', 'trend_var_names', 'measurement_params_config',
        'num_draws', # num_draws is also static
    )
)