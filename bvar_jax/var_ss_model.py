# --- var_ss_model.py (Final Corrected) ---
# NumPyro model for the BVAR with stationary prior and trends.

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jsl
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from typing import Dict, Tuple, List, Any, Sequence, Optional

# Assuming utils directory is in the Python path and contains the necessary files
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax # Removed check_stationarity_jax
from utils.Kalman_filter_jax import KalmanFilter # Use the standard KF for likelihood

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

# Add a small jitter for numerical stability
_MODEL_JITTER = 1e-8


# Helper function to get static off-diagonal indices (Moved here)
def _get_off_diagonal_indices(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates row and column indices for off-diagonal elements of an n x n matrix.
    These indices are static once n is known.
    """
    if n <= 1: # No off-diagonal elements for 0x0 or 1x1 matrices
        return jnp.empty((0,), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.int32)

    rows, cols = jnp.meshgrid(jnp.arange(n), jnp.arange(n))
    mask = jnp.eye(n, dtype=bool)
    off_diag_rows = rows[~mask]
    off_diag_cols = cols[~mask]
    return off_diag_rows, off_diag_cols


# Helper function to parse model equations (JAX compatible version of _parse_equation)
# This will be used internally during C matrix construction
def _parse_equation_jax(
    equation: str,
    trend_names: List[str],
    stationary_var_names: List[str],
    measurement_param_names: List[str],
    dtype=_DEFAULT_DTYPE
) -> List[Tuple[Optional[str], str, float]]:
    """
    Parse a measurement equation string into components with signs.
    Returns a list of tuples (param_name, state_name, sign).
    param_name is None for direct state terms.
    This function is executed *outside* the JAX graph (e.g., during model setup)
    as it relies on string parsing. The result is used to build the C matrix inside the model.
    """
    # Pre-process the equation string
    equation = equation.replace(' - ', ' + -')
    if equation.startswith('-'):
        equation = '-' + equation[1:].strip()

    # Split terms by '+'
    terms_str = [t.strip() for t in equation.split('+')]

    parsed_terms = []
    for term_str in terms_str:
        sign = 1.0
        if term_str.startswith('-'):
            sign = -1.0
            term_str = term_str[1:].strip()

        if '*' in term_str:
            parts = [p.strip() for p in term_str.split('*')]
            if len(parts) != 2:
                # In JAX model, we cannot raise Python exceptions easily within the traced code.
                # We'll assume valid equations are passed based on config validation.
                # If this was pure JAX, would need error handling via lax.cond or similar.
                # For setup, we rely on config validation.
                raise ValueError(f"Invalid term '{term_str}': Must have exactly one '*' operator")

            param, state = None, None
            if parts[0] in measurement_param_names:
                param, state = parts[0], parts[1]
            elif parts[1] in measurement_param_names:
                param, state = parts[1], parts[0]
            else:
                 raise ValueError(
                        f"Term '{term_str}' contains no valid parameter. "
                        f"Valid parameters are: {measurement_param_names}"
                    )

            if (state not in trend_names and
                state not in stationary_var_names):
                 raise ValueError(
                        f"Invalid state variable '{state}'. " # Simplified error message within JAX context
                        # f"Invalid state variable '{state}' in term '{term_str}'. "
                        # f"Valid states are: {trend_names + stationary_var_names}"
                    )

            parsed_terms.append((param, state, sign))
        else:
            if (term_str not in trend_names and
                term_str not in stationary_var_names):
                 raise ValueError(
                        f"Invalid state variable '{term_str}'. " # Simplified error message within JAX context
                        # f"Invalid state variable '{term_str}'. "
                        # f"Valid states are: {trend_names + stationary_var_names}"
                    )
            parsed_terms.append((None, term_str, sign))

    return parsed_terms


# Main NumPyro Model
def numpyro_bvar_stationary_model(
    y: jax.Array, # shape (time_steps, k_endog)
    config_data: Dict[str, Any], # Relevant parts of parsed YAML config including static indices
    static_valid_obs_idx: jax.Array, # 1D array of indices in y that are observed
    static_n_obs_actual: int, # Number of actually observed series
    # Pass variable names and equations directly for easier access in model
    trend_var_names: List[str],
    stationary_var_names: List[str],
    observable_names: List[str],
    model_eqs: List[Tuple[str, str]], # Raw model equations list
    initial_conds: Dict[str, Any], # Raw initial conditions dict
    stationary_prior_config: Dict[str, Any],
    trend_shocks_config: Dict[str, Any],
    measurement_params_config: List[Dict[str, Any]] # List of dicts with name, prior, etc.
):
    """
    NumPyro model for BVAR with trends using stationary prior.

    State vector: [trend_vars', stationary_t', stationary_{t-1}', ...]'
    For p=1: [trend_vars', stationary_t']'

    Args:
        y: Observed data, shape (time_steps, k_endog). Can contain NaNs.
           The Kalman filter handles NaNs based on static_valid_obs_idx.
        config_data: Dictionary containing the parsed YAML configuration,
                     *including* 'static_off_diag_indices' and 'num_off_diag'.
        static_valid_obs_idx: Indices of observed series (constant over time).
        static_n_obs_actual: Number of actually observed series.
        trend_var_names: List of names for trend state variables.
        stationary_var_names: List of names for stationary state variables (cycles).
        observable_names: List of names for observable variables.
        model_eqs: List of tuples (observable_name, equation_string).
        initial_conds: Dictionary from 'initial_conditions' in config.
        stationary_prior_config: Dictionary from 'stationary_prior' in config.
        trend_shocks_config: Dictionary from 'trend_shocks' in config.
        measurement_params_config: List of dictionaries for measurement parameters.
    """
    # --- Dimensions ---
    k_endog = len(observable_names)
    k_trends = len(trend_var_names)
    k_stationary = len(stationary_var_names)
    p = config_data['var_order'] # VAR order
    k_states = k_trends + k_stationary * p # Total state dimension

    # Get pre-parsed structures from config_data
    initial_conds_parsed = config_data['initial_conditions_parsed']
    parsed_model_eqs_list = config_data['model_equations_parsed'] # Already pre-parsed list of (obs_idx, parsed_terms)
    trend_names_with_shocks = config_data['trend_names_with_shocks']
    n_trend_shocks = len(trend_names_with_shocks)
    # Get static indices for off-diagonal elements of A from config_data
    static_off_diag_rows, static_off_diag_cols = config_data['static_off_diag_indices']
    num_off_diag = config_data['num_off_diag']


    n_shocks_state = n_trend_shocks + k_stationary # Shocks hitting state: trend shocks + current stationary shocks


    # --- Parameter Sampling ---

    # 1. Stationary VAR Coefficients (Phi_list)
    # Sample unconstrained A matrix (p x k_stationary x k_stationary)
    # Priors from stationary_prior_config: hyperparameters: es, fs
    hyperparams = stationary_prior_config['hyperparameters']
    es = jnp.asarray(hyperparams['es'], dtype=_DEFAULT_DTYPE)
    fs = jnp.asarray(hyperparams['fs'], dtype=_DEFAULT_DTYPE)

    # Sample diagonal and off-diagonal elements of A separately
    diag_A = numpyro.sample(
        "A_diag",
        dist.Normal(es[0], fs[0]).expand([p, k_stationary])
    )
    # Sample off-diagonal elements. Size is num_off_diag (statically known)
    if num_off_diag > 0:
        offdiag_A_flat = numpyro.sample(
            "A_offdiag",
            dist.Normal(es[1], fs[1]).expand([p, num_off_diag]) # Use static size here
        )
    else:
        offdiag_A_flat = jnp.empty((p, 0), dtype=_DEFAULT_DTYPE) # Empty if no off-diagonal


    # Reconstruct A matrix (p x k_stationary x k_stationary) using static indices
    A_unconstrained = jnp.zeros((p, k_stationary, k_stationary), dtype=_DEFAULT_DTYPE)

    # Set diagonal elements
    if k_stationary > 0:
        A_unconstrained = A_unconstrained.at[:, jnp.arange(k_stationary), jnp.arange(k_stationary)].set(diag_A)
        # Set off-diagonal elements using pre-calculated static indices
        if num_off_diag > 0:
            # Map flattened samples to the off-diagonal positions using static indices
            A_unconstrained = A_unconstrained.at[:, static_off_diag_rows, static_off_diag_cols].set(offdiag_A_flat)

    # A_draws is the (p, k_stationary, k_stationary) array
    A_draws = A_unconstrained

    # 2. Stationary Cycle Shock Variances (Diagonal of Sigma_cycles)
    # Sample variances individually with InverseGamma priors
    stationary_variances_list = []
    stationary_shocks_spec = stationary_prior_config.get('stationary_shocks', {})

    # Need to iterate in the specific order of stationary_var_names
    for stat_var_name in stationary_var_names:
        if stat_var_name not in stationary_shocks_spec:
             # This should be caught by config validation ideally
             raise ValueError(f"Missing shock specification for stationary variable '{stat_var_name}' in config.")

        shock_spec = stationary_shocks_spec[stat_var_name]
        if shock_spec['distribution'].lower() == 'inverse_gamma':
            params = shock_spec['parameters']
            var = numpyro.sample(
                f"stationary_var_{stat_var_name}",
                dist.InverseGamma(params['alpha'], params['beta'])
            )
            stationary_variances_list.append(var)
        else:
             raise NotImplementedError(f"Prior distribution '{shock_spec['distribution']}' not supported for stationary shock {stat_var_name}")

    # Stack sampled variances into a diagonal matrix of *standard deviations*
    if k_stationary == 0:
        stationary_D_sds = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
        stationary_variances = jnp.empty((0,), dtype=_DEFAULT_DTYPE)
    else:
        stationary_variances = jnp.stack(stationary_variances_list, axis=-1) # shape (k_stationary,)
        stationary_D_sds = jnp.diag(jnp.sqrt(jnp.maximum(stationary_variances, 1e-12))) # Ensure non-negative variance before sqrt! (k_stationary, k_stationary)


    # 3. Stationary Cycle Correlation Cholesky
    # LKJCholesky prior on the correlation matrix
    stationary_lkj_concentration = stationary_prior_config.get('covariance_prior', {}).get('eta', 1.0) # Default 1.0 if not specified
    
    # Only sample if k_stationary > 1 (correlation matrix is 1x1 for k_stationary=1)
    # if k_stationary > 1:
    #     stationary_chol = numpyro.sample(
    #         "stationary_chol",
    #         dist.LKJCholesky(k_stationary, concentration=stationary_lkj_concentration)
    #     ) # Samples the Cholesky factor of a correlation matrix (L_corr)
    # elif k_stationary == 1:
    #     # For 1x1 matrix, Cholesky of correlation is just 1. No sampling needed.
    #     stationary_chol = jnp.eye(1, dtype=_DEFAULT_DTYPE)
    # else: # k_stationary == 0
    #     stationary_chol = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE) # Empty if k_stationary == 0


    # # Construct Sigma_cycles using the specific PyMC formula: Sigma_cycles = L_corr @ D_sds @ L_corr.T
    # if k_stationary > 0:
    #      Sigma_cycles = stationary_chol @ stationary_D_sds @ stationary_chol.T
    #      Sigma_cycles = jnp.where(jnp.all(jnp.isfinite(Sigma_cycles)), Sigma_cycles, jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE)*1e6) # Handle potential NaN/Inf
    #      Sigma_cycles = (Sigma_cycles + Sigma_cycles.T)/2.0 # Ensure symmetry
    # else:
    #      Sigma_cycles = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    # This directly samples the Cholesky factor of the covariance matrix
    # Your approach is correct - just clean it up slightly
    if k_stationary > 1:
        # Sample correlation Cholesky factor
        stationary_chol = numpyro.sample(
            "stationary_chol",
            dist.LKJCholesky(k_stationary, concentration=stationary_lkj_concentration)
        )
        
        # Construct covariance matrix
        Sigma_cycles = stationary_chol @ stationary_D_sds @ stationary_chol.T
        
        # Your safety checks are good
        Sigma_cycles = jnp.where(
            jnp.all(jnp.isfinite(Sigma_cycles)), 
            Sigma_cycles, 
            jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE) * 1e6
        )
        Sigma_cycles = (Sigma_cycles + Sigma_cycles.T) / 2.0  # Ensure symmetry
        
    # 4. Trend Shock Variances (Diagonal of Sigma_trends)
    # Sample variances individually with InverseGamma priors for trends *with shocks defined*
    trend_variances_list = []
    trend_shocks_spec = trend_shocks_config.get('trend_shocks', {})

    # Need to iterate in the specific order of trend_names_with_shocks (from config_data)
    for trend_name in trend_names_with_shocks:
        # We already filtered to ensure these names are in trend_shocks_spec and have distribution
        shock_spec = trend_shocks_spec[trend_name]
        # Distribution is guaranteed to be inverse_gamma based on filtering in load_config_and_parse
        params = shock_spec['parameters']
        var = numpyro.sample(
            f"trend_var_{trend_name}",
            dist.InverseGamma(params['alpha'], params['beta'])
        )
        trend_variances_list.append(var)

    if n_trend_shocks == 0:
         # If no trends have shocks defined, Sigma_trends is 0x0
         Sigma_trends = jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)
    else:
         # Stack sampled variances into a diagonal matrix
         trend_variances = jnp.stack(trend_variances_list, axis=-1) # shape (n_trend_shocks,)
         Sigma_trends = jnp.diag(jnp.maximum(trend_variances, 1e-12)) # Ensure non-negative variance! (n_trend_shocks, n_trend_shocks)
         Sigma_trends = (Sigma_trends + Sigma_trends.T) / 2.0 # Ensure symmetry


    # 5. Measurement Parameters
    # Sample individual measurement parameters based on config
    measurement_params_sampled = {}
    measurement_param_names = [p['name'] for p in measurement_params_config]
    for param_spec in measurement_params_config:
        param_name = param_spec['name']
        prior_spec = param_spec['prior']
        if prior_spec['distribution'].lower() == 'normal':
            params = prior_spec['parameters']
            measurement_params_sampled[param_name] = numpyro.sample(
                param_name,
                dist.Normal(params['mu'], params['sigma'])
            )
        elif prior_spec['distribution'].lower() == 'half_normal':
            params = prior_spec['parameters']
            measurement_params_sampled[param_name] = numpyro.sample(
                param_name,
                dist.HalfNormal(params['sigma']) # Note: HalfNormal takes scale param, often sigma
            )
        else:
             # This should be caught by config validation ideally
             raise NotImplementedError(f"Prior distribution '{prior_spec['distribution']}' not supported for measurement parameter {param_name}")


    # --- Transformations to State-Space Matrices ---

    # 1. Transform unconstrained A to Phi_list
    # Requires Sigma_cycles (VAR shock covariance) and A_draws
    # Only perform transform if k_stationary > 0
    if k_stationary > 0:
        phi_list, _ = make_stationary_var_transformation_jax(Sigma_cycles, [A_draws[i] for i in range(p)], k_stationary, p)
        # Check stationarity is removed as per user instruction
        # is_stationary = check_stationarity_jax(phi_list, k_stationary, p)
    else:
        # If no stationary variables, phi_list is empty
        phi_list = []
        # is_stationary = jnp.array(True, dtype=jnp.bool_) # Process is trivially stationary


    # 2. State Transition Matrix (T)
    # T = [ I_{k_trends}  0_{k_trends, k_stationary*p} ]
    #     [ 0_{k_stationary*p, k_trends}  AR_comp ]
    T_comp = jnp.zeros((k_states, k_states), dtype=_DEFAULT_DTYPE)
    T_comp = T_comp.at[:k_trends, :k_trends].set(jnp.eye(k_trends, dtype=_DEFAULT_DTYPE)) # Trends block
    if k_stationary > 0:
        # AR Companion block using Phi_list
        T_comp = T_comp.at[k_trends:k_states, k_trends:k_states].set(create_companion_matrix_jax(phi_list, p, k_stationary))


    # 3. State Shock Impact Matrix (R)
    # R has shape (k_states, n_trend_shocks + k_stationary).
    # R maps [trend_shocks_std_normal', stationary_shocks_std_normal']' to state dynamics.
    R_comp = jnp.zeros((k_states, n_shocks_state), dtype=_DEFAULT_DTYPE)

    # Trends with shocks: I_{n_trend_shocks} in top-left block, mapping to trend state indices with shocks
    # R[trend_state_idx, shock_idx] = 1.0
    trend_state_indices = {name: i for i, name in enumerate(trend_var_names)} # Need this map again
    trend_shock_mapping_indices = {name: i for i, name in enumerate(trend_names_with_shocks)}
    for trend_name_with_shock in trend_names_with_shocks:
        trend_state_idx = trend_state_indices[trend_name_with_shock]
        shock_idx = trend_shock_mapping_indices[trend_name_with_shock]
        R_comp = R_comp.at[trend_state_idx, shock_idx].set(1.0)

    # Stationary shocks: I_{k_stationary} mapping to the *current* stationary state block (indices k_trends to k_trends+k_stationary-1)
    if k_stationary > 0:
        R_comp = R_comp.at[k_trends:k_trends+k_stationary, n_trend_shocks:].set(jnp.eye(k_stationary, dtype=_DEFAULT_DTYPE))

    # 4. Observation Matrix (C)
    # C has shape (k_endog, k_states). Based on parsed equations and sampled measurement params.
    C_comp = jnp.zeros((k_endog, k_states), dtype=_DEFAULT_DTYPE)
    # State indices for C matrix (Trends + Current Stationary)
    state_indices_for_C = {name: i for i, name in enumerate(trend_var_names)} # Trends
    state_indices_for_C.update({name: k_trends + i for i, name in enumerate(stationary_var_names)}) # Current Stationary

    # Populate C matrix using the parsed structure and sampled measurement parameters
    # parsed_model_eqs_list is list of (obs_idx, parsed_terms)
    for obs_idx, parsed_terms in parsed_model_eqs_list:
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
    # H is a zero matrix (k_endog, k_endog) based on PyMC code
    H_comp = jnp.zeros((k_endog, k_endog), dtype=_DEFAULT_DTYPE)


    # --- Initial State Distribution ---
    # Mean (init_x): Fixed values from parsed initial_conditions (means)
    init_x_comp = jnp.zeros(k_states, dtype=_DEFAULT_DTYPE)
    initial_state_parsed = initial_conds_parsed # Use the pre-parsed dict
    # Full state indices including lags for p > 1 (same as state_indices_full)
    state_indices_full = {name: i for i, name in enumerate(trend_var_names)} # Trends 0 to k_trends-1
    for lag in range(p):
         for i, name in enumerate(stationary_var_names):
              if lag == 0: # Current cycle
                   state_indices_full[name] = k_trends + i
              else: # Lagged cycles
                   lagged_state_name = f"{name}_t_minus_{lag}"
                   lagged_state_idx = k_trends + lag * k_stationary + i
                   state_indices_full[lagged_state_name] = lagged_state_idx

    for state_name_parsed, parsed_config in initial_state_parsed.items():
         if state_name_parsed in state_indices_full:
             init_x_comp = init_x_comp.at[state_indices_full[state_name_parsed]].set(parsed_config['mean'])
         # Note: Initial means for lagged stationary states are implicitly 0 if not listed in initial_conditions.


    # Covariance (init_P): Diagonal matrix from parsed initial_conditions variances, repeated for lags
    init_P_comp_diag_values = jnp.zeros(k_states, dtype=_DEFAULT_DTYPE)
    # Populate diagonal P0 based on variances from parsed initial_conditions: states
    for state_name_parsed, parsed_config in initial_state_parsed.items():
         if state_name_parsed in state_indices_full:
             state_idx = state_indices_full[state_name_parsed]
             init_P_comp_diag_values = init_P_comp_diag_values.at[state_idx].set(parsed_config['var'])

    # For lagged stationary states (p>1): PyMC sets variances to the same as current state variances.
    if p > 1:
         for lag in range(1, p):
             for i, name in enumerate(stationary_var_names):
                  current_state_name = name # Name in parsed initial conditions
                  lagged_state_name = f"{name}_t_minus_{lag}" # Name in full state vector
                  if current_state_name in initial_state_parsed and lagged_state_name in state_indices_full:
                       current_var = initial_state_parsed[current_state_name]['var']
                       lagged_idx = state_indices_full[lagged_state_name]
                       init_P_comp_diag_values = init_P_comp_diag_values.at[lagged_idx].set(current_var)


    init_P_comp = jnp.diag(init_P_comp_diag_values) # Shape (k_states, k_states)

    # Ensure init_P_comp is symmetric and regularized
    init_P_comp = (init_P_comp + init_P_comp.T) / 2.0
    init_P_comp = init_P_comp + _MODEL_JITTER * jnp.eye(k_states, dtype=_DEFAULT_DTYPE)

    # Check if initial P is valid (not containing NaNs from parsing or negative vars)
    is_init_P_valid_computed = jnp.all(jnp.isfinite(init_P_comp)) & jnp.all(jnp.diag(init_P_comp) >= 0)


    # --- Prepare Static Args for Kalman Filter ---
    # static_valid_obs_idx, static_n_obs_actual are passed directly
    # static_C_obs: C_comp restricted to observed rows
    static_C_obs = C_comp[static_valid_obs_idx, :] # Shape (n_obs_actual, k_states)

    # static_H_obs: H_comp restricted to observed rows/cols (will be zero matrix)
    static_H_obs = H_comp[static_valid_obs_idx[:, None], static_valid_obs_idx] # Shape (n_obs_actual, n_obs_actual)

    # static_I_obs: Identity matrix of size n_obs_actual
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
        y,
        static_valid_obs_idx,
        static_n_obs_actual,
        static_C_obs, # Use the sliced C
        static_H_obs, # Use the sliced H (zero)
        static_I_obs # Use the sliced I
    )

    # --- Compute Total Log-Likelihood ---
    total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])

    # --- Add Likelihood and Penalties ---
    # Stationarity check and penalty removed as per user instruction.

    # Add penalty if the initial P matrix is invalid (unlikely with diagonal from config)
    penalty_init_P = jnp.where(is_init_P_valid_computed, 0.0, -1e10)

    # If any NaNs occurred *before* the Kalman filter (e.g., during matrix construction)
    # This could happen if transformation or covariance construction failed numerically.
    # Check some key matrices for NaNs as a final safeguard before likelihood.
    # Add a penalty if T, R, C, H, init_x, or init_P contain NaNs
    matrices_to_check = [T_comp, R_comp, C_comp, H_comp, init_x_comp[None, :], init_P_comp] # Add dummy time dim for x
    any_matrix_nan = jnp.array(False)
    for mat in matrices_to_check:
        any_matrix_nan |= jnp.any(jnp.isnan(mat))

    penalty_matrix_nan = jnp.where(any_matrix_nan, -1e10, 0.0)


    numpyro.factor("log_likelihood", total_log_likelihood + penalty_init_P + penalty_matrix_nan) # Include penalty for invalid P and NaNs


    # --- Expose Transformed Parameters ---
    numpyro.deterministic("Phi_list", phi_list) # Return phi_list (list of matrices) for p>1
    numpyro.deterministic("Sigma_cycles", Sigma_cycles)
    numpyro.deterministic("Sigma_trends", Sigma_trends) # Diagonal (n_trend_shocks, n_trend_shocks)
    numpyro.deterministic("T_comp", T_comp)
    numpyro.deterministic("R_comp", R_comp) # (k_states, n_trend_shocks + k_stationary)
    numpyro.deterministic("C_comp", C_comp) # (k_endog, k_states)
    numpyro.deterministic("H_comp", H_comp) # Zero matrix (k_endog, k_endog)
    numpyro.deterministic("init_x_comp", init_x_comp) # (k_states,)
    numpyro.deterministic("init_P_comp", init_P_comp) # (k_states, k_states)
    # is_stationary removed as per user instruction
    numpyro.deterministic("k_states", k_states) # Useful for downstream

    # Expose sampled variances and correlation Cholesky for checking
    # Stationary
    if k_stationary > 0:
         for i, name in enumerate(stationary_var_names):
              # No need to check if sampled, just define the deterministic using the sampled variable
              numpyro.deterministic(f"stationary_var_{name}_det", stationary_variances_list[i])
         # Sampled stationary_chol only if k_stationary > 1, deterministic if k_stationary == 1
         numpyro.deterministic("stationary_chol_det", stationary_chol)


    # Trends
    if n_trend_shocks > 0:
        for i, name in enumerate(trend_names_with_shocks): # Only trends with shocks
             # No need to check if sampled
             numpyro.deterministic(f"trend_var_{name}_det", trend_variances_list[i])


    # Measurement params
    for param_name, param_value in measurement_params_sampled.items():
         numpyro.deterministic(f"{param_name}_det", param_value)


# Helper function to parse config initial states for NumPyro
# Keep this function here or import it if it lives in config.py
def parse_initial_state_config(initial_conditions_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Parse initial state configuration (means and variances) from config.yaml.
    Replicates BVARConfig._parse_initial_state logic.
    """
    parsed_states = {}
    raw_states_config = initial_conditions_config.get('states', {})

    for state_name, state_config in raw_states_config.items():
        parsed = {}
        if isinstance(state_config, (int, float)):
            parsed = {"mean": float(state_config), "var": 1.0}
        elif isinstance(state_config, dict) and "mean" in state_config:
            parsed = state_config
        elif isinstance(state_config, str):
            parts = state_config.split()
            temp_result = {}
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    key = parts[i].strip().rstrip(':')
                    value = float(parts[i + 1])
                    temp_result[key] = value
            if "mean" in temp_result:
                if "var" not in temp_result:
                    temp_result["var"] = 1.0
                parsed = temp_result

        if "mean" not in parsed:
            # This case should ideally be caught by config validation
            # In JAX, better to return NaN or large value if parsing fails unexpectedly
            # For this helper used in setup, raising an error is acceptable.
            raise ValueError(f"Could not parse mean for initial state '{state_name}'.")
        if "var" not in parsed:
             # This case should ideally be caught by config validation
             raise ValueError(f"Could not parse variance for initial state '{state_name}'.")

        # Ensure var is non-negative, clip at a small value
        var_val = jnp.maximum(jnp.array(parsed["var"], dtype=_DEFAULT_DTYPE), 1e-12) # Ensure positive variance
        parsed_states[state_name] = {"mean": jnp.array(parsed["mean"], dtype=_DEFAULT_DTYPE), "var": var_val}

    return parsed_states