import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_to_dense, triangular_to_full
from typing import Dict, Tuple

# Assuming utils directory is in the Python path
from utils.stationary_prior_jax_simplified import make_stationary_var_transformation_jax, create_companion_matrix_jax, check_stationarity_jax
from utils.Kalman_filter_jax import KalmanFilter # Use the standard KF for likelihood

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

# Add a small jitter for numerical stability, especially in Lyapunov solver
_MODEL_JITTER = 1e-8

def numpyro_var_ss_model(y: jax.Array, m: int, p: int):
    """
    NumPyro model for a stationary VAR(p) process in state-space form.

    State space form:
    x_t = T x_{t-1} + R epsilon_t, epsilon_t ~ N(0, I)
    y_t = C x_t + eta_t, eta_t ~ N(0, H)

    The state vector x_t is [y_t', y_{t-1}', ..., y_{t-p+1}']'.
    T is the companion matrix derived from VAR coefficients Phi_1, ..., Phi_p.
    R is related to the shock covariance Sigma of the VAR process y_t = Phi_1 y_{t-1} + ... + noise_t.
    C selects the first m components of the state vector (y_t).

    Priors are placed on unconstrained parameters (A matrices for VAR, Cholesky factors for Sigma and H).
    Stationarity is ensured by transforming A matrices to Phi matrices.
    The initial state distribution is the stationary distribution derived from T and R@R.T.

    Args:
        y: Observed data, shape (time_steps, full_obs_dim). Can contain NaNs.
           Note: This model currently uses a Kalman filter implementation
           that assumes a *constant* subset of observed series over time.
           The 'full_obs_dim' is the total potential observation dimension.
           Missing values (NaNs) are handled by identifying the subset of
           series that are *always* observed and filtering only those.
        m: Dimension of the VAR process (number of variables).
        p: Order of the VAR process (number of lags).
    """
    mp = m * p # Dimension of the state vector

    # --- Identify Observed Subset ---
    # This logic assumes that some columns in `y` are *entirely* NaN (never observed)
    # and some are *partially* or fully observed (at least one non-NaN value).
    # The filter works on the subset of columns that are *always* potentially observed.
    # Time-varying missing data (NaNs within an otherwise observed column) are handled
    # by the filter's internal jnp.isfinite checks, but the structure of C and H
    # is fixed based on the initial assessment here.

    full_obs_dim = y.shape[-1] # Total number of potential observation series
    T_steps = y.shape[0]

    # Find columns with at least one non-NaN value
    valid_obs_mask_cols = jnp.any(jnp.isfinite(y), axis=0)
    static_valid_obs_idx = jnp.where(valid_obs_mask_cols)[0]
    static_n_obs_actual = static_valid_obs_idx.shape[0]

    # --- Sample Parameters ---

    # 1. Unconstrained parameters for VAR coefficients (A matrices)
    # Sample p matrices A_1, ..., A_p, each m x m.
    # The transformation make_stationary_var_transformation_jax takes a list of m x m matrices.
    # We sample a flattened array and reshape it.
    a_flat = numpyro.sample("a_flat", dist.Normal(0., 1.).expand([p * m * m]))
    A_list_unconstrained = [
        jnp.reshape(a_flat[i * m * m : (i + 1) * m * m], (m, m))
        for i in range(p)
    ]

    # 2. Shock Covariance Matrix (Sigma) for the VAR noise term
    # Sample the Cholesky decomposition L_sigma where Sigma = L_sigma @ L_sigma.T
    # L_sigma is lower triangular m x m. It has m*(m+1)/2 non-zero elements.
    sigma_chol_packed = numpyro.sample(
        "sigma_chol_packed",
        dist.Normal(0., 1.).expand([m * (m + 1) // 2]) # Prior on unconstrained elements
    )
    L_sigma = triangular_to_full(sigma_chol_packed, m, m, False) # False for lower triangular
    Sigma = L_sigma @ L_sigma.T # Ensure Sigma is symmetric PSD

    # 3. Observation Noise Covariance Matrix (H)
    # The observation equation is y_t = C x_t + eta_t.
    # In this VAR state-space form, C x_t just selects y_t, y_{t-1}, etc.
    # If y_t = [actual data], then H represents measurement error.
    # If the state is [y_t, ...], and observation is [y_t] with *no* extra noise, H should be zero.
    # However, the provided KalmanFilter takes H as input, and bvar_with_trends often includes measurement error.
    # Let's assume H is a parameter for measurement error on the *full* potential observation vector y.
    # H will be full_obs_dim x full_obs_dim.
    # Sample the Cholesky decomposition L_H where H = L_H @ L_H.T
    # L_H is lower triangular full_obs_dim x full_obs_dim.

    h_chol_packed = numpyro.sample(
        "h_chol_packed",
        # Use HalfNormal for scale elements (diagonal of Cholesky) to enforce positivity
        # And Normal for off-diagonal elements. Pack them appropriately.
        dist.HalfNormal(1.).expand([full_obs_dim * (full_obs_dim + 1) // 2])
    )
    L_H_full = triangular_to_full(h_chol_packed, full_obs_dim, full_obs_dim, False)
    H_full = L_H_full @ L_H_full.T # Ensure H_full is symmetric PSD

    # --- Transformations to State-Space Matrices ---

    # 1. Transform unconstrained A matrices to VAR coefficients (Phi) and Gamma
    # make_stationary_var_transformation_jax requires Sigma to be m x m,
    # and returns phi_list = [Phi_1, ..., Phi_p] (each m x m) and gamma_list.
    # Gamma is not directly used in the standard state-space formulation likelihood,
    # but might be related to the stationary covariance derivation or interpretation.
    # We need Phi_list for the companion matrix T.
    phi_list, _ = make_stationary_var_transformation_jax(Sigma, A_list_unconstrained, m, p)

    # Ensure transformed coefficients are finite (check_stationarity includes NaN check)
    # If transformation resulted in NaNs, the process is non-stationary or transformation failed.
    # We can factor a -infinity log-likelihood in this case.
    is_stationary = check_stationarity_jax(phi_list, m, p)

    # If not stationary or transformation failed, return -infinity likelihood
    # This guides the sampler away from non-stationary regions.
    # We can use a dummy likelihood contribution of -large_value if not stationary.
    # We need to compute the full likelihood first, and then scale it or add a penalty.
    # Or, use a numpyro.factor that is conditionally -inf.
    # Let's add a large penalty if not stationary *after* computing the KF likelihood.

    # 2. State Transition Matrix (T) - The Companion Matrix
    # T maps x_{t-1} to E[x_t | x_{t-1}].
    # x_t = [y_t', y_{t-1}', ..., y_{t-p+1}']'
    # E[y_t | ...] = Phi_1 y_{t-1} + ... + Phi_p y_{t-p}
    # E[y_{t-1} | ...] = y_{t-1}
    # ...
    # E[y_{t-p+1} | ...] = y_{t-p+1}
    # T = [Phi_1 Phi_2 ... Phi_p ]
    #     [I_m   0     ... 0     ]
    #     [0     I_m   ... 0     ]
    #     [...                   ]
    #     [0     ... I_m   0     ]
    T_comp = create_companion_matrix_jax(phi_list, p, m) # Shape (mp, mp)

    # 3. State Shock Impact Matrix (R)
    # x_t = T x_{t-1} + R epsilon_t.
    # The state shock comes from the VAR noise term in the first m equations (for y_t).
    # y_t = Phi_1 y_{t-1} + ... + Phi_p y_{t-p} + noise_t
    # noise_t = L_sigma @ epsilon_t_m (where epsilon_t_m is m x 1 standard normal)
    # The state shock vector should map the standard normal shock epsilon_t_mp (mp x 1)
    # to the state dynamics.
    # The state equation is:
    # y_t = Phi_1 y_{t-1} + ... + Phi_p y_{t-p} + L_sigma epsilon_t_m
    # y_{t-1} = y_{t-1}
    # ...
    # y_{t-p+1} = y_{t-p+1}
    # So, the state shock vector is [L_sigma epsilon_t_m', 0, ..., 0]', where the 0s are m-dimensional zero vectors.
    # The standard normal shock vector for the state space is epsilon_t_mp, size mp.
    # R should be (mp, n_shocks_state). The non-zero shocks only hit the first m state elements.
    # If n_shocks_state == m, R should map epsilon_t_m to [L_sigma epsilon_t_m', 0, ..., 0]'.
    # This implies R is [L_sigma; 0_{m*(p-1) x m}].
    R_comp = jnp.vstack([L_sigma, jnp.zeros((m * (p - 1), m), dtype=_DEFAULT_DTYPE)]) # Shape (mp, m)
    # The covariance of R epsilon_t is R @ R.T = [L_sigma @ L_sigma.T, 0; 0, 0] = [Sigma, 0; 0, 0].
    # This matches the structure: Q_comp = [Sigma, 0; 0, 0], shape (mp, mp).

    # 4. Observation Matrix (C)
    # y_t = C x_t + eta_t
    # x_t = [y_t', y_{t-1}', ..., y_{t-p+1}']'
    # The observation is just y_t (the first m elements of the state).
    # C selects these: C = [I_m, 0, ..., 0], shape (m, mp).
    C_comp_full_obs_dim = jnp.hstack([jnp.eye(m, dtype=_DEFAULT_DTYPE), jnp.zeros((m, m * (p - 1)), dtype=_DEFAULT_DTYPE)]) # Shape (m, mp)

    # The observation matrix C in the Kalman filter takes state to the *full* observation space.
    # If full_obs_dim > m, the observation equation implies we observe more than just y_t.
    # However, the VAR state definition [y_t, y_{t-1}, ...] only naturally produces y_t.
    # The C matrix should map state to the *full* observed space, using the static_valid_obs_idx logic.
    # The standard VAR setup often implies observing only y_t variables. If full_obs_dim > m,
    # it suggests either some observed series are *not* in the state vector (unlikely in standard form)
    # or the observation model is different. Let's assume the standard VAR state-space where we
    # potentially observe the first m components of the state, possibly with measurement error.
    # The C matrix passed to the KF should map (mp,) state to (full_obs_dim,) observation space.
    # It should be an (full_obs_dim, mp) matrix.
    # It selects the first m state variables (y_t) for the relevant observation series.
    # If the first m series of `y` correspond to y_t, and other series are also observed... this state space
    # model doesn't explain them.
    # Revisit bvar_with_trends structure: It uses a state vector of size (mp + n_deterministic),
    # where the first mp are VAR states, and the rest are deterministic. The observation matrix
    # maps this extended state to the observed series. The first m rows of the observation matrix
    # map to the VAR variables. The state vector IS [y_t, y_{t-1}, ...]. So C should map state to
    # observations.
    # Let's assume the full observation space (full_obs_dim) includes *at least* the m VAR variables.
    # C_full_obs_dim should be (full_obs_dim, mp). It should select the first m state elements
    # for the first m observation dimensions, and handle other observation dimensions if needed.
    # For a standard VAR(p) state space, C maps [y_t', ...] to y_t'. So C is [I_m | 0].
    # If we observe more than m series, the state space model needs to be extended or H needs to cover cross-covariances.
    # Let's assume the model observes the first m state variables (y_t) and the rest are measurement error or zero.
    # A simple approach: C maps state to first m obs dims. Other obs dims have zero mapping from state.
    # C_full_obs_dim = [ [I_m, 0_{m, m*(p-1)}], [0_{full_obs_dim-m, m*p}] ]
    # This seems wrong. The state vector is defined based on the m variables. The observation model
    # connects this state to the *actual* observed series.
    # The C matrix should have shape (full_obs_dim, mp).
    # It should pick out the relevant state elements for each observation series.
    # For the first m series (y_t), it picks the first m state elements.
    # C_full_obs_dim = [I_m | 0] for the first m rows.
    # How the other full_obs_dim - m rows look depends on what those series observe.
    # If we are only modeling the m VAR series, but observe more, the model is misspecified or needs extension.
    # Let's assume the model only explains the first `m` observed series. The filter can still run
    # with `full_obs_dim`, but the structure of C and H needs careful thought.
    # Standard VAR state-space: observation is the first m state variables. C = [I_m, 0].
    # The Kalman Filter in utils/Kalman_filter_jax.py takes C shape (n_obs_full, n_state).
    # This means C_comp needs shape (full_obs_dim, mp).
    # The simplest interpretation consistent with a VAR state vector is that C maps the state
    # to a *potential* observation vector of size full_obs_dim, where the first m elements are y_t,
    # and the rest are 0 or pick other state variables if the state vector was expanded (e.g. trends).
    # Given the prompt is about a standard stationary VAR, let's assume C maps state to *only* the m VAR variables.
    # Then n_obs_full = m. If y has more columns, they must be handled differently or ignored.
    # Let's stick to the standard VAR state space definition.
    # State: x_t = [y_t', ..., y_{t-p+1}']' (mp x 1)
    # Observation: y_t_obs = C x_t + eta_t. If y_t_obs is the m x 1 vector y_t, then C = [I_m, 0].
    # The provided KalmanFilter takes `C` as `n_obs_full x n_state`. This suggests n_obs_full is the full dim.
    # Let's assume `y` contains the `m` VAR variables *first*, possibly followed by other series.
    # The Kalman filter needs `C` (full_obs_dim, mp).
    # For the first `m` rows of C, it's `[I_m | 0]`. For the remaining `full_obs_dim - m` rows, it's `[0 | 0]`
    # assuming those other series are not explained by the VAR state.
    C_for_kf = jnp.vstack([
        jnp.hstack([jnp.eye(m, dtype=_DEFAULT_DTYPE), jnp.zeros((m, m * (p - 1)), dtype=_DEFAULT_DTYPE)]),
        jnp.zeros((full_obs_dim - m, mp), dtype=_DEFAULT_DTYPE)
    ]) # Shape (full_obs_dim, mp)

    # 5. Observation Noise Covariance (H)
    # This was sampled as H_full (full_obs_dim, full_obs_dim). This seems consistent
    # with the Kalman filter taking H of shape (n_obs_full, n_obs_full).

    # --- Initial State Distribution ---
    # Use the stationary distribution: x_0 ~ N(init_x, init_P)
    # init_x = 0 for a zero-mean VAR (no constant term in the state equation)
    # init_P satisfies the discrete-time Lyapunov equation: P = T @ P @ T.T + Q
    # where Q = R @ R.T is the state shock covariance [Sigma, 0; 0, 0].
    # We need to solve P - T @ P @ T.T = Q for P.
    # jax.scipy.linalg.solve_discrete_lyapunov(A.T, Q) solves X = A @ X @ A.T + Q for X.
    # So we need solve_discrete_lyapunov(T_comp.T, Q_comp).
    init_x_comp = jnp.zeros(mp, dtype=_DEFAULT_DTYPE)

    # Construct Q_comp = R_comp @ R_comp.T = [Sigma, 0; 0, 0]
    Q_comp = R_comp @ R_comp.T # Shape (mp, mp)
    # Ensure Q_comp is symmetric and regularized before solving Lyapunov
    Q_comp = (Q_comp + Q_comp.T) / 2.0
    Q_comp_reg = Q_comp + _MODEL_JITTER * jnp.eye(mp, dtype=_DEFAULT_DTYPE)

    # Check if T_comp is stable (eigenvalues inside unit circle) BEFORE solving Lyapunov.
    # The stationary prior transformation is supposed to ensure this for the Phi coefficients,
    # but numerical precision or issues in transformation could lead to T_comp being unstable.
    # solve_discrete_lyapunov requires A (here T_comp.T) to be stable.
    # If not stable, the stationary covariance doesn't exist, or is infinite.
    # In the non-stationary case, the filter likelihood is mathematically zero (log-likelihood is -inf).
    # We should catch this and return -inf likelihood.
    T_comp_stable = True
    if not is_stationary: # check_stationarity_jax already checks eigenvalues < 1
         T_comp_stable = False # Flag T_comp as unstable if Phi were non-stationary/NaN

    init_P_comp = jnp.eye(mp, dtype=_DEFAULT_DTYPE) * 1e6 # Default to large covariance if solve fails

    # Solve Lyapunov equation only if T_comp is stable
    def solve_lyapunov(T_comp_stable_arg, Q_comp_reg_arg):
         try:
              # T_comp_stable_arg is T_comp.T
              P_sol = jsl.solve_discrete_lyapunov(T_comp_stable_arg, Q_comp_reg_arg)
              # Check for NaNs/Infs in the solution
              solution_valid = jnp.all(jnp.isfinite(P_sol))
              # Check if the solution is positive semidefinite (or close) - solve_discrete_lyapunov should ensure this if T is stable
              # evals = jnp.linalg.eigvalsh(P_sol) # eigvalsh for symmetric matrix
              # is_psd = jnp.all(evals > -_MODEL_JITTER) # Check eigenvalues are non-negative
              # return jnp.where(solution_valid & is_psd, P_sol, jnp.eye(mp)*1e6)
              return jnp.where(solution_valid, P_sol, jnp.eye(mp, dtype=_DEFAULT_DTYPE)*1e6) # Just check for finite solution for now
         except Exception:
              # If solver raises an error (e.g., T is not stable), return default
              return jnp.eye(mp, dtype=_DEFAULT_DTYPE)*1e6

    init_P_comp_unreg = jax.lax.cond(
         jnp.array(T_comp_stable), # Only try solving if the transform yielded a stationary T
         lambda op: solve_lyapunov(op[0], op[1]),
         lambda op: jnp.eye(mp, dtype=_DEFAULT_DTYPE)*1e6, # Return default if T is not stable
         operand=(T_comp.T, Q_comp_reg)
    )

    # Ensure init_P_comp is symmetric and regularized
    init_P_comp = (init_P_comp_unreg + init_P_comp_unreg.T) / 2.0
    init_P_comp = init_P_comp + _MODEL_JITTER * jnp.eye(mp, dtype=_DEFAULT_DTYPE)


    # --- Prepare Static Args for Kalman Filter ---
    # The KalmanFilter expects static args derived from the observed subset.
    # These must match the structure identified at the beginning.
    # static_C_obs: C_for_kf restricted to rows corresponding to static_valid_obs_idx
    static_C_obs = C_for_kf[static_valid_obs_idx, :] # Shape (n_obs_actual, mp)

    # static_H_obs: H_full restricted to rows/cols corresponding to static_valid_obs_idx
    static_H_obs = H_full[static_valid_obs_idx[:, None], static_valid_obs_idx] # Shape (n_obs_actual, n_obs_actual)

    # static_I_obs: Identity matrix of size n_obs_actual
    static_I_obs = jnp.eye(static_n_obs_actual, dtype=_DEFAULT_DTYPE)


    # --- Instantiate and Run Kalman Filter ---
    # The filter takes T, R, C, H, init_x, init_P
    # T = T_comp (mp, mp)
    # R = R_comp (mp, m) - maps m standard normal shocks to state shocks
    # C = C_for_kf (full_obs_dim, mp) - maps state to *full* observation space
    # H = H_full (full_obs_dim, full_obs_dim) - covariance of *full* observation noise
    # init_x = init_x_comp (mp,)
    # init_P = init_P_comp (mp, mp)

    # Ensure all inputs to KF have the correct dtype
    T_comp = jnp.asarray(T_comp, dtype=_DEFAULT_DTYPE)
    R_comp = jnp.asarray(R_comp, dtype=_DEFAULT_DTYPE)
    C_for_kf = jnp.asarray(C_for_kf, dtype=_DEFAULT_DTYPE)
    H_full = jnp.asarray(H_full, dtype=_DEFAULT_DTYPE)
    init_x_comp = jnp.asarray(init_x_comp, dtype=_DEFAULT_DTYPE)
    init_P_comp = jnp.asarray(init_P_comp, dtype=_DEFAULT_DTYPE)


    # Instantiate KalmanFilter
    kf = KalmanFilter(T_comp, R_comp, C_for_kf, H_full, init_x_comp, init_P_comp)

    # Run the filter. Need to pass the full `y` data matrix and the static args.
    # Note: The filter expects `y` shape (T, n_obs_full).
    filter_results = kf.filter(
        y,
        static_valid_obs_idx,
        static_n_obs_actual,
        static_C_obs,
        static_H_obs,
        static_I_obs
    )

    # --- Compute Total Log-Likelihood ---
    # The filter returns log-likelihood contributions for each time step where an update occurred.
    # We sum these contributions.
    total_log_likelihood = jnp.sum(filter_results['log_likelihood_contributions'])

    # --- Add Likelihood to Model ---
    # numpyro.factor adds an arbitrary contribution to the log-probability.
    # This is where the observed data likelihood is incorporated.

    # Add a penalty if the process is not stationary or Lyapunov failed significantly.
    # The check_stationarity_jax function returning False or init_P_comp being large (~default)
    # indicates non-stationarity or numerical issues.
    # A large positive penalty subtracts from the log-likelihood.
    penalty = jnp.where(is_stationary, 0.0, -1e10) # Apply a large penalty if not stationary

    # Check if init_P_comp is potentially invalid (e.g., default large value)
    # This can happen if Lyapunov solve failed.
    is_init_P_valid = jnp.all(jnp.isfinite(init_P_comp)) # Check for NaNs/Infs
    is_init_P_large = jnp.max(jnp.abs(init_P_comp)) > 1e5 # Check if it's the default large value

    # Add another penalty if the stationary initial covariance could not be computed correctly
    penalty += jnp.where(is_init_P_valid & (~is_init_P_large), 0.0, -1e10)


    # Add the log likelihood contribution
    numpyro.factor("log_likelihood", total_log_likelihood + penalty)

    # --- (Optional) Expose Transformed Parameters ---
    # We can use numpyro.deterministic to save transformed parameters in the posterior.
    numpyro.deterministic("phi_list", phi_list)
    numpyro.deterministic("Sigma", Sigma)
    numpyro.deterministic("H_full", H_full)
    numpyro.deterministic("T_comp", T_comp)
    numpyro.deterministic("R_comp", R_comp)
    numpyro.deterministic("C_for_kf", C_for_kf)
    numpyro.deterministic("init_x_comp", init_x_comp)
    numpyro.deterministic("init_P_comp", init_P_comp)
    numpyro.deterministic("is_stationary", is_stationary)