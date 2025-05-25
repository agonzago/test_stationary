# --- run_single_draw.py ---
# This file contains the routine to run the simulation smoother for a single parameter vector.

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from typing import Dict, Tuple, Union, List, Any

# Assuming utils directory is in the Python path
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax
from utils.hybrid_dk_smoother import HybridDKSimulationSmoother # Use the DK smoother class

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

# Add a small jitter consistent with model/filter
_DRAW_JITTER = 1e-8


def run_simulation_smoother_single_params(
    params: Dict[str, jax.Array],
    y: jax.Array,
    m: int,
    p: int,
    key: jax.random.PRNGKey,
    num_draws: int = 1,
) -> Tuple[jax.Array, Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]]:
    """
    Runs the simulation smoother using a fixed parameter vector.

    Args:
        params: A dictionary containing the parameter values, e.g., from posterior mean/median.
                Expected keys: 'a_flat', 'sigma_chol_packed', 'h_chol_packed'.
        y: Observed data, shape (time_steps, full_obs_dim).
           Note: The HybridDKSimulationSmoother expects dense observations
           for its internal filter/smoother used in the D-K loop.
           If your original `y` has NaNs, you might need to impute them
           for the *initial* filter/smooth run within this function,
           before calling `smoother.smooth_and_simulate`.
           The `smooth_and_simulate` method itself calls a DENSE filter.
           For this function, let's assume `y` is the *original* data,
           and we will use the `HybridDKSimulationSmoother` which calls its
           `_filter_internal` and `_rts_smoother_backend` that are designed
           for DENSE data. This means if `y` has NaNs, the initial smooth
           will likely be incorrect.
           A robust solution would pass the static observation info here
           and modify HybridDKSimulationSmoother to accept/use it in its
           initial filter/smooth step, or impute `y` first.
           Let's proceed assuming `y` is the original data, and the
           HybridDKSimulationSmoother's internal filter/smoother works
           on this (potentially dense-view of) data.
           For the purpose of this routine, it's often run on posterior MEANS,
           and the resulting smoothed path (x_smooth_original_dense) is then
           used for simulation draws, where the simulated data *is* dense.
           So the DENSE assumption in HybridDKSimulationSmoother methods is okay
           for the simulation loop, but might be problematic for the *initial* smooth of `y`.

           **Important:** For correct initial smoothing with original data containing NaNs,
           the HybridDKSimulationSmoother would need to use a filter that handles NaNs
           correctly (like the one in Kalman_filter_jax.py). As currently implemented,
           `_filter_internal` in `hybrid_dk_smoother.py` is for dense data.
           Let's use the `KalmanFilter` from `utils/Kalman_filter_jax.py`
           for the initial smooth of the *original* data `y`, and then pass
           its smoothed output to `HybridDKSimulationSmoother` which will use
           its internal DENSE filter/smoother for the simulation loop.

        m: Dimension of the VAR process.
        p: Order of the VAR process.
        key: JAX random key for simulation draws.
        num_draws: Number of simulation draws to perform. Defaults to 1.

    Returns:
        Tuple: (smoothed_states_original, simulation_results)
        smoothed_states_original: Smoothed state means from the original data `y`
                                   using the provided parameters. Shape (T, n_state).
        simulation_results: Depends on num_draws:
                            If num_draws == 1: Single draw (T, n_state).
                            If num_draws > 1: Tuple (mean_draws, median_draws, all_draws)
                                             Shapes: (T, n_state), (T, n_state), (num_draws, T, n_state).
    """
    full_obs_dim = y.shape[-1]
    T_steps = y.shape[0]
    mp = m * p # State dimension

    # --- Extract and Transform Parameters ---
    a_flat = params['a_flat']
    sigma_chol_packed = params['sigma_chol_packed']
    h_chol_packed = params['h_chol_packed']

    # Reconstruct Sigma
    L_sigma = promote_to_dense(triangular_to_full(sigma_chol_packed, m, m, False))
    Sigma = L_sigma @ L_sigma.T
    Sigma = (Sigma + Sigma.T) / 2.0 # Ensure symmetry

    # Reconstruct H_full
    L_H_full = promote_to_dense(triangular_to_full(h_chol_packed, full_obs_dim, full_obs_dim, False))
    H_full = L_H_full @ L_H_full.T
    H_full = (H_full + H_full.T) / 2.0 # Ensure symmetry

    # Reshape A_list
    A_list_unconstrained = [
        jnp.reshape(a_flat[i * m * m : (i + 1) * m * m], (m, m))
        for i in range(p)
    ]

    # Transform A to Phi using the stationary prior logic
    # This also provides the VAR shock covariance Sigma if needed, but we use the sampled one.
    phi_list, _ = make_stationary_var_transformation_jax(Sigma, A_list_unconstrained, m, p)

    # Create state-space matrices T, R, C
    T_comp = create_companion_matrix_jax(phi_list, p, m) # Shape (mp, mp)

    # R_comp = [L_sigma; 0]
    R_comp = jnp.vstack([L_sigma, jnp.zeros((m * (p - 1), m), dtype=_DEFAULT_DTYPE)]) # Shape (mp, m)

    # C_for_kf maps state to full observation space (full_obs_dim, mp)
    C_for_kf = jnp.vstack([
        jnp.hstack([jnp.eye(m, dtype=_DEFAULT_DTYPE), jnp.zeros((m, m * (p - 1)), dtype=_DEFAULT_DTYPE)]),
        jnp.zeros((full_obs_dim - m, mp), dtype=_DEFAULT_DTYPE)
    ]) # Shape (full_obs_dim, mp)


    # Compute stationary initial state distribution (mean and covariance)
    init_x_comp = jnp.zeros(mp, dtype=_DEFAULT_DTYPE) # Assuming zero mean VAR state

    # Compute Q_comp = R_comp @ R_comp.T = [Sigma, 0; 0, 0]
    Q_comp = R_comp @ R_comp.T
    Q_comp = (Q_comp + Q_comp.T) / 2.0
    Q_comp_reg = Q_comp + _DRAW_JITTER * jnp.eye(mp, dtype=_DEFAULT_DTYPE)

    # Solve Lyapunov equation for stationary covariance: P = T @ P @ T.T + Q
    # solve_discrete_lyapunov(A.T, Q) solves X = A @ X @ A.T + Q
    # Need to check if T_comp is stable first. Use check_stationarity_jax on the derived phi_list.
    is_stationary = check_stationarity_jax(phi_list, m, p)

    init_P_comp = jnp.eye(mp, dtype=_DEFAULT_DTYPE) * 1e6 # Default to large covariance if solve fails

    def solve_lyapunov_draw(T_comp_stable_arg, Q_comp_reg_arg):
         try:
              P_sol = jsl.solve_discrete_lyapunov(T_comp_stable_arg, Q_comp_reg_arg)
              solution_valid = jnp.all(jnp.isfinite(P_sol))
              # Ensure PSD property explicitly if solve_discrete_lyapunov doesn't guarantee it
              # evals = jnp.linalg.eigvalsh(P_sol)
              # is_psd = jnp.all(evals > -_DRAW_JITTER)
              # return jnp.where(solution_valid & is_psd, P_sol, jnp.eye(mp)*1e6)
              return jnp.where(solution_valid, P_sol, jnp.eye(mp, dtype=_DEFAULT_DTYPE)*1e6)
         except Exception:
              return jnp.eye(mp, dtype=_DEFAULT_DTYPE)*1e6

    # Solve Lyapunov equation only if T_comp is stable
    init_P_comp_unreg = jax.lax.cond(
         is_stationary, # Check stationarity of the derived phi_list
         lambda op: solve_lyapunov_draw(op[0], op[1]),
         lambda op: jnp.eye(mp, dtype=_DEFAULT_DTYPE)*1e6, # Return default if not stationary
         operand=(T_comp.T, Q_comp_reg)
    )

    # Ensure init_P_comp is symmetric and regularized
    init_P_comp = (init_P_comp_unreg + init_P_comp_unreg.T) / 2.0
    init_P_comp = init_P_comp + _DRAW_JITTER * jnp.eye(mp, dtype=_DEFAULT_DTYPE)

    # Check if initial covariance is valid (not the default large value)
    is_init_P_valid_computed = jnp.all(jnp.isfinite(init_P_comp)) & (jnp.max(jnp.abs(init_P_comp)) < 1e5)

    # --- Run Initial Filter/Smoother on Original Data `y` ---
    # This uses the standard KalmanFilter from Kalman_filter_jax.py which
    # handles time-varying NaNs if static info is correctly derived.
    # However, the HybridDKSimulationSmoother assumes DENSE data for its internal filter.
    # To be safe and correct with original data `y` which might have NaNs,
    # we *must* use the NaN-handling filter from Kalman_filter_jax.py for this step.

    # Prepare static args for KalmanFilter on original data `y`
    # These must match the logic in the NumPyro model.
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]

    static_C_obs = C_for_kf[static_valid_obs_idx, :] # Shape (n_obs_actual, mp)
    static_H_obs = H_full[static_valid_obs_idx[:, None], static_valid_obs_idx] # Shape (n_obs_actual, n_obs_actual)
    static_I_obs = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)


    # Instantiate and run KalmanFilter from utils/Kalman_filter_jax.py
    # Use the computed parameters (T_comp, R_comp, C_for_kf, H_full, init_x_comp, init_P_comp)
    # Only proceed if parameters are finite and stationary covariance was computed correctly.
    # If not stationary or init_P failed, the smoothed states will be meaningless (likely NaNs or zeros).
    # The simulation draws depend on the difference (x_smooth_original - x_smooth_sim),
    # so if the initial smooth fails, the draws will also be garbage.

    run_initial_smooth = is_stationary & is_init_P_valid_computed & (T_steps > 0)

    # Define default outputs for the case where initial smooth is skipped
    default_smoothed_states = jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE)
    default_sim_results: Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]
    if num_draws == 1:
         default_sim_results = jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE)
    else:
         default_sim_results = (
             jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE), # Mean
             jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE), # Median
             jnp.full((num_draws, T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE) # All draws
         )

    def run_smoother_and_draws_actual(operand):
        y_actual, static_args_actual, kf_params_actual, key_actual, num_draws_actual = operand
        T_comp_kf, R_comp_kf, C_for_kf_kf, H_full_kf, init_x_comp_kf, init_P_comp_kf = kf_params_actual
        static_valid_obs_idx_kf, static_n_obs_actual_kf, static_C_obs_kf, static_H_obs_kf, static_I_obs_kf = static_args_actual

        # 1. Run filter on original (potentially NaN) data `y`
        kf_original = KalmanFilter(T_comp_kf, R_comp_kf, C_for_kf_kf, H_full_kf, init_x_comp_kf, init_P_comp_kf)
        filter_results_original = kf_original.filter(
            y_actual,
            static_valid_obs_idx_kf,
            static_n_obs_actual_kf,
            static_C_obs_kf,
            static_H_obs_kf,
            static_I_obs_kf
        )

        # Check if filter produced valid results
        filter_ok = jnp.all(jnp.isfinite(filter_results_original['x_filt'])) & \
                    jnp.all(jnp.isfinite(filter_results_original['P_filt'])) & \
                    jnp.all(jnp.isfinite(filter_results_original['log_likelihood_contributions']))

        # 2. Run RTS smoother on original data filter results IF filter was OK
        smoothed_states_original_means = jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE)
        smoothed_states_original_covs = jnp.full((T_steps, mp, mp), jnp.nan, dtype=_DEFAULT_DTYPE)

        def smooth_if_filter_ok(filter_results_ok):
             # Pass the filter results dictionary to the smoother
             kf_smoother = KalmanFilter(T_comp_kf, R_comp_kf, C_for_kf_kf, H_full_kf, init_x_comp_kf, init_P_comp_kf) # Re-instantiate or reuse
             smoothed_means, smoothed_covs = kf_smoother.smooth(y_actual, filter_results=filter_results_ok) # Pass data and filter results
             return smoothed_means, smoothed_covs

        smoothed_states_original_means, smoothed_states_original_covs = jax.lax.cond(
            filter_ok,
            smooth_if_filter_ok,
            lambda res: (jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE), jnp.full((T_steps, mp, mp), jnp.nan, dtype=_DEFAULT_DTYPE)),
            operand=filter_results_original # Pass the filter results
        )

        # 3. Run Simulation Smoother Draws IF initial smooth was OK
        # The HybridDKSimulationSmoother's draw_simulation method expects *dense* data and smooth states.
        # We need dense original observations for the D-K loop (y_star = C x_star + eta_star).
        # Here's the catch: If `y` has NaNs, the D-K loop in HybridDKSimulationSmoother (specifically _draw_simulation_small/large)
        # will simulate dense y_star and run its internal *dense* filter/smoother on y_star. This is correct.
        # However, the formula is x_draw = x_star + (x_smooth_original - x_smooth_star).
        # `x_smooth_original` *must* be from a filter/smoother run on the original data `y` correctly handling NaNs.
        # The `smoothed_states_original_means` computed above *is* from such a filter/smoother (using KalmanFilter from utils).
        # But `HybridDKSimulationSmoother.draw_simulation` also requires the original observations (`original_ys_dense`).
        # It uses this `original_ys_dense` only to get the shape and pass it to `_filter_internal` inside the D-K loop,
        # which expects dense data.
        # This means for the D-K loop *itself*, we don't need the *original* `y`. We only need its shape `(T_steps, full_obs_dim)`.
        # The `HybridDKSimulationSmoother` needs T, R, C_full, H_full, init_x, init_P, and the *smoothed states* from the *original* data.
        # Let's pass the actual original `y` data, but the `HybridDKSimulationSmoother` will internally use a DENSE filter on simulated data.

        sim_smoother_draws_results: Union[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]
        if num_draws == 1:
            sim_smoother_draws_results = jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE)
        else:
            sim_smoother_draws_results = (
                jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE), # Mean
                jnp.full((T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE), # Median
                jnp.full((num_draws, T_steps, mp), jnp.nan, dtype=_DEFAULT_DTYPE) # All draws
            )


        def run_draws_if_smooth_ok(smoothed_states_means_ok):
            # Instantiate HybridDKSimulationSmoother. It needs the same parameters.
            dk_smoother = HybridDKSimulationSmoother(
                T_comp_kf, R_comp_kf, C_for_kf_kf, H_full_kf, init_x_comp_kf, init_P_comp_kf
            )
            # Pass the original *potentially NaN* data `y_actual` to `run_smoother_draws`.
            # The `run_smoother_draws` method takes `original_ys_dense`. Its internal `_filter_internal` expects dense data.
            # This seems like a potential mismatch. The HybridDKSimulationSmoother should ideally take the original data `y`
            # *and* the static observation info, and use a NaN-handling filter internally for both the initial smooth (which it currently doesn't do)
            # and for the filter step on *simulated* data (which is dense, so its current dense filter is fine there).
            # Let's *impute* the original data `y` with the smoothed means *before* passing it to `run_smoother_draws`.
            # This is a common workaround: y_imputed = y_original; y_imputed[NaNs] = C @ x_smooth[NaN indices].
            # The C matrix for the *full* observation space (C_for_kf) is needed for this imputation.
            y_imputed = jnp.where(jnp.isfinite(y_actual), y_actual, jnp.nan) # Start with original, NaNs preserved
            # Create predicted observations from smoothed states
            y_predicted_from_smooth = smoothed_states_means_ok @ C_for_kf_kf.T # (T, mp) @ (mp, full_obs_dim).T = (T, full_obs_dim)
            # Impute NaNs in original_ys using predicted values from smoothed states
            y_dense_for_draws = jnp.where(jnp.isfinite(y_actual), y_actual, y_predicted_from_smooth) # Replace NaNs with predicted/smoothed means

            # Now call run_smoother_draws with the imputed dense data and the smoothed means from original data
            # Pass a split key for the draws
            key_draws = random.split(key_actual)[0] # Take first key from split for multiple draws
            draws_results = dk_smoother.run_smoother_draws(
                y_dense_for_draws, # Use imputed dense data for the D-K loop's internal filter calls
                key_draws,
                num_draws_actual,
                smoothed_states_means_ok # Use the correctly smoothed states from original data
            )
            return draws_results

        sim_smoother_draws_results = jax.lax.cond(
            jnp.all(jnp.isfinite(smoothed_states_original_means)), # Only run draws if initial smooth produced finite means
            run_draws_if_smooth_ok,
            lambda smooth_means_failed: default_sim_results, # Return default NaN results if initial smooth failed
            operand=smoothed_states_original_means # Pass smoothed means to the conditional function
        )


        return smoothed_states_original_means, sim_smoother_draws_results


    # Execute the main conditional logic
    smoothed_states_original_means, sim_smoother_draws_results = jax.lax.cond(
        run_initial_smooth, # Check if parameters are valid for initial smooth
        run_smoother_and_draws_actual,
        lambda op: (default_smoothed_states, default_sim_results), # Return defaults if initial smooth conditions not met
        operand=(y, (static_valid_obs_idx, static_n_obs_actual, static_C_obs, static_H_obs, static_I_obs),
                 (T_comp, R_comp, C_for_kf, H_full, init_x_comp, init_P_comp), key, num_draws)
    )


    return smoothed_states_original_means, sim_smoother_draws_results


# JIT compile the function with static arguments if num_draws is static.
# If num_draws can be dynamic, the internal Python loop prevents JIT.
# Let's assume num_draws is static for typical use (e.g. num_draws=1 for testing).
# If you need dynamic num_draws, remove `static_argnames`.
run_simulation_smoother_single_params_jit = jax.jit(
    run_simulation_smoother_single_params,
    static_argnames=('m', 'p', 'num_draws')
)