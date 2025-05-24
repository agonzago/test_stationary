# stationary_prior_jax.py

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import lax
from jax.typing import ArrayLike
from typing import List, Tuple

# Ensure JAX is configured for float64 if enabled
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# Numerical stability jitter
# Increased jitter for robustness
_JITTER_PRIOR = 1e-8 # <-- Changed from 1e-12

def symmetrize_jax(A: ArrayLike) -> jax.Array:
    """JAX: Symmetrize a matrix by computing (A + A.T)/2."""
    A_jax = jnp.asarray(A, dtype=_DEFAULT_DTYPE)
    return 0.5 * (A_jax + A_jax.T)

def sqrtm_jax(A: ArrayLike) -> jax.Array:
    """
    JAX: Matrix square root computation using eigendecomposition.
    Equivalent to Stan's sqrtm function. Handles potential non-PSD input numerically.
    """
    A_sym = symmetrize_jax(A) # Ensure matrix is symmetric

    # Compute eigenvalues and eigenvectors
    # eigvals are real for symmetric matrices
    # Use a small jitter on the diagonal before eigh for robustness against near-singularity
    # Add jitter *before* eigenvalue decomposition
    A_sym_reg = A_sym + _JITTER_PRIOR * jnp.eye(A_sym.shape[0], dtype=_DEFAULT_DTYPE)
    evals, evecs = jsl.eigh(A_sym_reg)

    # Handle potential negative eigenvalues due to numerical issues or non-PSD input
    # Clip eigenvalues to a small positive value
    evals_clipped = jnp.maximum(evals, _JITTER_PRIOR) # <-- Use the same jitter for clipping floor

    # Compute the fourth root of eigenvalues (element-wise)
    # This is sqrt(sqrt(evals_clipped)) as per Stan code (root_root_evals)
    root_root_evals = jnp.sqrt(jnp.sqrt(evals_clipped))

    # Construct the result: evecs @ diag(root_root_evals) @ evecs.T
    # This is equivalent to tcrossprod(evecs @ diag(root_root_evals))
    eprod = evecs @ jnp.diag(root_root_evals)

    # Return tcrossprod(eprod) = eprod @ eprod.T, which corresponds to Stan's sqrtm output
    return eprod @ eprod.T

def tcrossprod_jax(A: ArrayLike) -> jax.Array:
    """JAX: Compute A @ A.T (equivalent to Stan's tcrossprod)."""
    A_jax = jnp.asarray(A, dtype=_DEFAULT_DTYPE)
    return A_jax @ A_jax.T

def quad_form_sym_jax(A: ArrayLike, B: ArrayLike) -> jax.Array:
    """
    JAX: Compute B.T @ A @ B (symmetric quadratic form).
    Ensures A is treated as symmetric.
    """
    # Ensure matrix is symmetric for numerical stability, although B.T @ A @ B is symmetric if A is.
    A_sym = symmetrize_jax(A)
    B_jax = jnp.asarray(B, dtype=_DEFAULT_DTYPE)
    return B_jax.T @ A_sym @ B_jax

def AtoP_jax(A: ArrayLike, m: int) -> jax.Array:
    """
    JAX: Transform A to P (equivalent to Stan's AtoP function).
    
    Args:
        A: m x m matrix
        m: Dimension of the VAR
        
    Returns:
        m x m matrix P
    """
    A_jax = jnp.asarray(A, dtype=_DEFAULT_DTYPE)
    
    B = tcrossprod_jax(A_jax) # B = A @ A.T
    
    # Add 1.0 to diagonal elements of B
    B = B + jnp.eye(m, dtype=_DEFAULT_DTYPE)
    
    # Compute matrix square root of B
    # Use our JAX sqrtm_jax implementation which handles potential non-PSD numerically
    sqrtB = sqrtm_jax(B)
    
    # Compute sqrtB^(-1) @ A
    # Use jsl.solve for robust solving. Use assume_a='pos' because B + I is SPD if A is finite.
    # If sqrtm_jax output is exactly symmetric positive semidefinite, 'pos' is appropriate.
    # Given how sqrtm_jax is built from eigendecomposition of a symmetric matrix, it should be SPD.
    # Let's try assuming 'pos' as it matches the likely assumption in the original implementation
    # and can be faster/more stable if the matrix *is* SPD.
    try:
       # Add jitter *before* solve for robustness
       sqrtB_reg = sqrtB + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
       P = jsl.solve(sqrtB_reg, A_jax, assume_a='pos') # <-- Added jitter here
       return P
    except Exception:
       # Fallback to general solve if 'pos' fails, or return NaN
       # Given the try/except in the test function, returning NaN is fine here
       return jnp.full_like(A_jax, jnp.nan)


def rev_mapping_jax(P_list: ArrayLike, Sigma: ArrayLike, m: int, p: int) -> Tuple[List[jax.Array], List[jax.Array]]:
    """
    JAX: Equivalent to Stan's rev_mapping function using lax.scan.
    Implements the backward Levinson-Durbin recursion.

    Args:
        P_list: Stacked JAX array of p matrices, shape (p, m, m), [P_0, ..., P_{p-1}].
        Sigma: m×m covariance matrix (JAX array).
        m: Dimension of the VAR process.
        p: Order of the VAR process.
        
    Returns:
        Tuple of (phi_list, gamma_list):
            - phi_list: List of p VAR coefficient matrices [phi_0, ..., phi_{p-1}].
                        These are the final VAR coefficients phi_{p-1}[0..p-1] in Stan's notation.
            - gamma_list: List of p autocovariance matrices [Gamma_{p-1}, ..., Gamma_0].
                          Corresponds to Stan's Gamma_trans[1]...Gamma_trans[p].
                          (Note: The Stan code lists Gamma_trans[1..p], which seems to correspond
                           to Gamma_{p-1} down to Gamma_0 in standard VAR literature notation,
                           where Gamma_k is the autocovariance at lag k). Let's return [Gamma_0, .. Gamma_{p-1}]
                           for standard order. No, Stan returns Gamma_trans[1..p]. Let's return [Gamma_1..Gamma_p]
                           corresponding to Stan's Gamma_trans[1..p]. Gamma_trans[0] is Sigma_for[0].
                           Let's clarify the output format needed later for NumPyro, but match Stan for now:
                           gamma_list = [Gamma_trans[1], ..., Gamma_trans[p]].
    """
    Sigma_jax = jnp.asarray(Sigma, dtype=_DEFAULT_DTYPE)
    P_list_jax = jnp.asarray(P_list, dtype=_DEFAULT_DTYPE) # Expects stacked array (p, m, m)

    # Handle VAR(0) case
    if p == 0:
        # phi_list is empty, gamma_list contains only Gamma_0 = Sigma
        return [], [Sigma_jax] # Gamma_0 is the autocovariance at lag 0

    # --- Step 1: Forward Pass (Compute Sigma_for and S_for_list) ---
    # Stan loop s=1..p. Computes Sigma_for[p-s] and S_for_list[p-s].
    # Uses P[p-s+1], Sigma_for[p-s+1], S_for_list[p-s+1].
    # Initial: Sigma_for[p] = Sigma, S_for_list[p] = sqrtm(Sigma).
    # Scan iterates k=0..p-1, corresponding to Stan's s=k+1.
    # Input P matrices should be in reverse order: P_{p-1}, P_{p-2}, ..., P_0 (0-indexed).
    # Stan's P[p-s+1] for s=1..p corresponds to P_list_jax[p-s] (0-indexed).
    # s=1 -> P[p] -> P_list_jax[p-1]
    # s=p -> P[1] -> P_list_jax[0]
    # So the input to the scan needs to be P_list_jax[::-1].

    # Initial carry for scan step k=0 (Stan s=1): (Sigma_for[p], S_for_list[p])
    init_carry_step1 = (Sigma_jax, sqrtm_jax(Sigma_jax))

    # Scan body for Step 1: Computes Sigma_for[s-1] and S_for_list[s-1] from s
    # carry: (Sigma_for_curr_idx, S_for_list_curr_idx)
    # input: P_matrix corresponding to P_{s-1} (in our 0-based index). Scan inputs are P_list_jax[::-1].
    # Scan runs k=0..p-1. Input element is P_list_jax[p-1-k]. This corresponds to P_{s-1} where s-1=p-1-k => s = p-k.
    # So the carry should be (Sigma_for[p-k], S_for_list[p-k]).
    # Input P matrix is P_{p-1-k}.
    # Output computed are (Sigma_for[p-k-1], S_for_list[p-k-1]).

    def step1_scan_body(carry, P_matrix_input): # P_matrix_input is P_list_jax[p-1-k]
        # carry: (Sigma_for[p-k], S_for_list[p-k]) from Stan step s=p-k
        Sigma_for_curr_idx, S_for_list_curr_idx = carry

        # The P matrix needed is P[s] from Stan notation. In 0-indexed, this is P_{s-1}.
        # When k runs 0..p-1, s=p-k runs p..1. P[s] runs P[p]..P[1]. In 0-index, P_{p-1}..P_0.
        # So the input P_matrix_input = P_list_jax[p-1-k] is correct. This IS P_{s-1}.
        P_s_minus_1 = P_matrix_input

        # Compute S_for corresponding to P[s-1]
        S_for_val = -tcrossprod_jax(P_s_minus_1) + jnp.eye(m, dtype=_DEFAULT_DTYPE)

        # Check S_for_val for NaNs/Infs before sqrtm
        S_for_val = jnp.where(jnp.all(jnp.isfinite(S_for_val)), S_for_val, jnp.full_like(S_for_val, jnp.nan))
        S_rev_curr = sqrtm_jax(S_for_val) # sqrtm_jax handles non-PSD numerically

        # Check S_rev_curr for NaNs/Infs before quad_form
        S_rev_curr = jnp.where(jnp.all(jnp.isfinite(S_rev_curr)), S_rev_curr, jnp.full_like(S_rev_curr, jnp.nan))
        # Compute Q = S_rev_curr.T @ Sigma_for[s] @ S_rev_curr (Symmetric form)
        Q = quad_form_sym_jax(Sigma_for_curr_idx, S_rev_curr)

        # Check Q for NaNs/Infs before sqrtm
        Q = jnp.where(jnp.all(jnp.isfinite(Q)), Q, jnp.full_like(Q, jnp.nan))
        sqrtQ = sqrtm_jax(Q) # sqrtm_jax handles non-PSD numerically

        # Check sqrtQ for NaNs/Infs before solve
        sqrtQ = jnp.where(jnp.all(jnp.isfinite(sqrtQ)), sqrtQ, jnp.full_like(sqrtQ, jnp.nan))
        # Compute MID = S_rev_curr^(-1) @ sqrtQ
        # Use jsl.solve(A, B) -> A^(-1) B
        # Add jitter *before* solve
        S_rev_curr_reg = S_rev_curr + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
            MID = jsl.solve(S_rev_curr_reg, sqrtQ, assume_a='pos') # Assuming S_rev_curr is SPD after sqrtm_jax
        except Exception:
            MID = jnp.full_like(MID, jnp.nan)

        # Check MID for NaNs/Infs before solve
        MID = jnp.where(jnp.all(jnp.isfinite(MID)), MID, jnp.full_like(MID, jnp.nan))
        # Compute S_for_list[s-1] = MID @ S_rev_curr^(-1)
        # Use jsl.solve(A.T, B.T).T -> (A^(-1) B).T -> B.T @ A.T^(-1) -> (A.T @ B.T^(-1)).T
        # S_for_list[s-1] = solve(S_rev_curr.T, MID.T, T).T
        S_rev_curr_T_reg = S_rev_curr.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
            S_for_list_s_minus_1 = jsl.solve(S_rev_curr_T_reg, MID.T, assume_a='pos').T # Assuming S_rev_curr.T is SPD
        except Exception:
            S_for_list_s_minus_1 = jnp.full_like(S_for_list_s_minus_1, jnp.nan)


        # Check S_for_list_s_minus_1 for NaNs/Infs before tcrossprod
        S_for_list_s_minus_1 = jnp.where(jnp.all(jnp.isfinite(S_for_list_s_minus_1)), S_for_list_s_minus_1, jnp.full_like(S_for_list_s_minus_1, jnp.nan))
        # Compute Sigma_for[s-1] = tcrossprod(S_for_list[s-1])
        Sigma_for_s_minus_1 = tcrossprod_jax(S_for_list_s_minus_1)

        # Check final outputs for NaNs/Infs before returning in carry
        Sigma_for_s_minus_1 = jnp.where(jnp.all(jnp.isfinite(Sigma_for_s_minus_1)), Sigma_for_s_minus_1, jnp.full_like(Sigma_for_s_minus_1, jnp.nan))
        S_for_list_s_minus_1 = jnp.where(jnp.all(jnp.isfinite(S_for_list_s_minus_1)), S_for_list_s_minus_1, jnp.full_like(S_for_list_s_minus_1, jnp.nan))

        next_carry = (Sigma_for_s_minus_1, S_for_list_s_minus_1)

        # Store computed values for index s-1 (which is p-k-1)
        return next_carry, next_carry # Store results for index p-k-1

    # Run scan for Step 1
    # Input P matrices in reverse order: [P_{p-1}, P_{p-2}, ..., P_0] (shape p, m, m)
    # Initial carry: (Sigma_for[p], S_for_list[p])
    (final_Sigma_for_0, final_S_for_list_0), (sigma_for_scan_results, s_for_list_scan_results) = lax.scan(
        step1_scan_body, init_carry_step1, P_list_jax[::-1], length=p
    )

    # Collect results:
    # sigma_for_scan_results is stacked [Sigma_for[p-1], ..., Sigma_for[0]]
    # s_for_list_scan_results is stacked [S_for_list[p-1], ..., S_for_list[0]]

    # Sigma_for_full_list = [Sigma_for[0], ..., Sigma_for[p-1], Sigma_for[p]] (p+1 elements)
    # Need to reverse the stacked scan results, then add the initial Sigma_for[p]
    Sigma_for_full_list = [sigma_for_scan_results[i, ...] for i in range(p)][::-1] # Indices 0 to p-1
    Sigma_for_full_list.append(Sigma_jax) # Index p

    # S_for_list_full_list = [S_for_list[0], ..., S_for_list[p-1]] (p elements)
    # Need to reverse the stacked scan results
    S_for_list_full_list = [s_for_list_scan_results[i, ...] for i in range(p)][::-1]

    # --- Step 2: Backward Pass (Compute phi and Gamma_trans) ---
    # Stan loop s=0..p-1.
    # Computes phi_for[s][s], phi_rev[s][s], phi_for[s][k] (k<s), phi_rev[s][k] (k<s), Gamma_trans[s+1], Sigma_rev[s+1].
    # Uses Sigma_rev[s], S_for_list[s], P[s], Sigma_for[s], phi_for[s-1][.], phi_rev[s-1][.], Gamma_trans[.].

    # This requires carrying the state for phi (padded layers) and Gamma_trans (padded history).
    # Carry state: (phi_for_padded, phi_rev_padded, Gamma_trans_padded, Sigma_rev_s_val)
    # phi_for_padded: (p, p, m, m) array. Stores phi_for[s][k] at index [s, k]. Zero padded.
    # phi_rev_padded: (p, p, m, m) array. Stores phi_rev[s][k] at index [s, k]. Zero padded.
    # Gamma_trans_padded: (p+1, m, m) array. Stores Gamma_trans[s] at index [s]. Zero padded.
    # Sigma_rev_s_val: m x m matrix. Stores Sigma_rev[s].

    # Initial carry (k=0, Stan s=0):
    # Sigma_rev[0] = Sigma_for[0]
    Sigma_rev_0 = Sigma_for_full_list[0] # This is the Sigma_for[0] computed at the *end* of step 1

    # Gamma_trans[0] = Sigma_for[0] = Sigma_rev_0
    Gamma_trans_0 = Sigma_rev_0

    # phi_for/rev[0][0] computed using s=0 inputs: Sigma_rev[0], S_for_list[0], P[0].
    S_rev_0 = sqrtm_jax(Sigma_rev_0) # S_rev[0] = sqrtm(Sigma_rev[0])
    S_for_list_0 = S_for_list_full_list[0] # S_for_list[0]
    P_0 = P_list_jax[0] # P[0]

    # Check inputs for phi_for/rev[0][0] solve for NaNs/Infs
    S_rev_0 = jnp.where(jnp.all(jnp.isfinite(S_rev_0)), S_rev_0, jnp.full_like(S_rev_0, jnp.nan))
    S_for_list_0 = jnp.where(jnp.all(jnp.isfinite(S_for_list_0)), S_for_list_0, jnp.full_like(S_for_list_0, jnp.nan))
    P_0 = jnp.where(jnp.all(jnp.isfinite(P_0)), P_0, jnp.full_like(P_0, jnp.nan))


    # Compute phi_for[0][0] = solve(S_rev_0.T, (S_for_list_0 @ P_0.T).T, T).T
    S_rev_0_T_reg = S_rev_0.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
    try:
       phi_for_00 = jsl.solve(S_rev_0_T_reg, (S_for_list_0 @ P_0.T).T, assume_a='pos').T
    except Exception:
       phi_for_00 = jnp.full_like(phi_for_00, jnp.nan)


    # Compute phi_rev[0][0] = solve(S_for_list_0.T, (S_rev_0 @ P_0.T).T, T).T
    S_for_list_0_T_reg = S_for_list_0.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
    try:
       phi_rev_00 = jsl.solve(S_for_list_0_T_reg, (S_rev_0 @ P_0.T).T, assume_a='pos').T
    except Exception:
       phi_rev_00 = jnp.full_like(phi_rev_00, jnp.nan)


    initial_phi_for_padded = jnp.zeros((p, p, m, m), dtype=_DEFAULT_DTYPE)
    initial_phi_rev_padded = jnp.zeros((p, p, m, m), dtype=_DEFAULT_DTYPE)
    initial_Gamma_trans_padded = jnp.zeros((p+1, m, m), dtype=_DEFAULT_DTYPE) # Stores indices 0..p

    # Check phi_for_00 and phi_rev_00 for NaNs/Infs before setting
    phi_for_00 = jnp.where(jnp.all(jnp.isfinite(phi_for_00)), phi_for_00, jnp.full_like(phi_for_00, jnp.nan))
    phi_rev_00 = jnp.where(jnp.all(jnp.isfinite(phi_rev_00)), phi_rev_00, jnp.full_like(phi_rev_00, jnp.nan))
    Gamma_trans_0 = jnp.where(jnp.all(jnp.isfinite(Gamma_trans_0)), Gamma_trans_0, jnp.full_like(Gamma_trans_0, jnp.nan))

    if p > 0:
        initial_phi_for_padded = initial_phi_for_padded.at[0, 0, :, :].set(phi_for_00)
        initial_phi_rev_padded = initial_phi_rev_padded.at[0, 0, :, :].set(phi_rev_00)
        initial_Gamma_trans_padded = initial_Gamma_trans_padded.at[0, :, :].set(Gamma_trans_0)


    init_carry_step2_combined = (
        initial_phi_for_padded,
        initial_phi_rev_padded,
        initial_Gamma_trans_padded,
        Sigma_rev_0 # Sigma_rev[0]
    )

    # Scan input for k = 0..p-1 (Stan s=k): Uses S_for_list[k], P[k], Sigma_for[k]
    scan2_inputs_combined = (
        jnp.stack(S_for_list_full_list, axis=0), # shape (p, m, m)
        P_list_jax,                              # shape (p, m, m) - original order
        jnp.stack(Sigma_for_full_list[:-1], axis=0) # shape (p, m, m) (Sigma_for[0]..Sigma_for[p-1])
    )

    # Scan body for s = 0..p-1 (index k in scan)
    def step2_combined_scan_body(carry, inputs):
        # k goes from 0 to p-1. This is Stan's `s` index.
        # carry: (phi_for_prev_padded, phi_rev_prev_padded, Gamma_trans_prev_padded, Sigma_rev_s_val)
        phi_for_prev_padded, phi_rev_prev_padded, Gamma_trans_prev_padded, Sigma_rev_s_val = carry

        # Inputs for step s=k: (S_for_list_k, P_k, Sigma_for_k)
        S_for_list_k, P_k, Sigma_for_k_val = inputs

        # Check inputs for NaNs/Infs before using
        S_for_list_k = jnp.where(jnp.all(jnp.isfinite(S_for_list_k)), S_for_list_k, jnp.full_like(S_for_list_k, jnp.nan))
        P_k = jnp.where(jnp.all(jnp.isfinite(P_k)), P_k, jnp.full_like(P_k, jnp.nan))
        Sigma_for_k_val = jnp.where(jnp.all(jnp.isfinite(Sigma_for_k_val)), Sigma_for_k_val, jnp.full_like(Sigma_for_k_val, jnp.nan))
        Sigma_rev_s_val = jnp.where(jnp.all(jnp.isfinite(Sigma_rev_s_val)), Sigma_rev_s_val, jnp.full_like(Sigma_rev_s_val, jnp.nan))


        # 1. Compute phi_for[k][j], phi_rev[k][j] for j=0..k-1 (if k > 0)
        # Use phi_for[k-1][j], phi_rev[k-1][j] from phi_for_prev_padded, phi_rev_prev_padded

        # Compute phi_for[k][k] and phi_rev[k][k] first (using Sigma_rev[k] and S_for_list[k])
        S_rev_k_val = sqrtm_jax(Sigma_rev_s_val) # Sigma_rev[k] is Sigma_rev_s_val

        # Check inputs for phi_for/rev[k][k] solve for NaNs/Infs
        S_rev_k_val = jnp.where(jnp.all(jnp.isfinite(S_rev_k_val)), S_rev_k_val, jnp.full_like(S_rev_k_val, jnp.nan))


        # Compute phi_for[k][k] = solve(S_rev_k_val.T, (S_for_list_k @ P_k.T).T, T).T
        S_rev_k_val_T_reg = S_rev_k_val.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
           phi_for_kk = jsl.solve(S_rev_k_val_T_reg, (S_for_list_k @ P_k.T).T, assume_a='pos').T
        except Exception:
           phi_for_kk = jnp.full_like(phi_for_kk, jnp.nan)


        # Compute phi_rev[k][k] = solve(S_for_list_k.T, (S_rev_k_val @ P_k.T).T, T).T
        S_for_list_k_T_reg = S_for_list_k.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
           phi_rev_kk = jsl.solve(S_for_list_k_T_reg, (S_rev_k_val @ P_k.T).T, assume_a='pos').T
        except Exception:
           phi_rev_kk = jnp.full_like(phi_rev_kk, jnp.nan)


        phi_for_k_padded = jnp.zeros((p, p, m, m), dtype=_DEFAULT_DTYPE)
        phi_rev_k_padded = jnp.zeros((p, p, m, m), dtype=_DEFAULT_DTYPE)

        # Check phi_for_kk and phi_rev_kk for NaNs/Infs before setting/using
        phi_for_kk = jnp.where(jnp.all(jnp.isfinite(phi_for_kk)), phi_for_kk, jnp.full_like(phi_for_kk, jnp.nan))
        phi_rev_kk = jnp.where(jnp.all(jnp.isfinite(phi_rev_kk)), phi_rev_kk, jnp.full_like(phi_rev_kk, jnp.nan))


        if k > 0:
            # Map over j=0..k-1
            def compute_phi_k_j_scoped(j_idx):
                 # uses k, phi_for_prev_padded, phi_rev_prev_padded, phi_for_kk, phi_rev_kk
                 # Check inputs for NaNs/Infs before use within map
                 pf_km1_j = jnp.where(jnp.all(jnp.isfinite(phi_for_prev_padded[k-1, j_idx, :, :])), phi_for_prev_padded[k-1, j_idx, :, :], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                 pr_km1_km1_j = jnp.where(jnp.all(jnp.isfinite(phi_rev_prev_padded[k-1, k-1-j_idx, :, :])), phi_rev_prev_padded[k-1, k-1-j_idx, :, :], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                 pr_km1_j = jnp.where(jnp.all(jnp.isfinite(phi_rev_prev_padded[k-1, j_idx, :, :])), phi_rev_prev_padded[k-1, j_idx, :, :], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                 pf_km1_km1_j = jnp.where(jnp.all(jnp.isfinite(phi_for_prev_padded[k-1, k-1-j_idx, :, :])), phi_for_prev_padded[k-1, k-1-j_idx, :, :], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                 pkk = jnp.where(jnp.all(jnp.isfinite(phi_for_kk)), phi_for_kk, jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                 prkk = jnp.where(jnp.all(jnp.isfinite(phi_rev_kk)), phi_rev_kk, jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))


                 phi_for_k_j = pf_km1_j - pkk @ pr_km1_km1_j
                 phi_rev_k_j = pr_km1_j - prkk @ pf_km1_km1_j

                 # Check outputs for NaNs/Infs
                 phi_for_k_j = jnp.where(jnp.all(jnp.isfinite(phi_for_k_j)), phi_for_k_j, jnp.full_like(phi_for_k_j, jnp.nan))
                 phi_rev_k_j = jnp.where(jnp.all(jnp.isfinite(phi_rev_k_j)), phi_rev_k_j, jnp.full_like(phi_rev_k_j, jnp.nan))

                 return phi_for_k_j, phi_rev_k_j

            js_phi_k = jnp.arange(k)
            phi_for_k_first_k, phi_rev_k_first_k = lax.map(
                compute_phi_k_j_scoped,
                js_phi_k
            ) # Shapes (k, m, m)

            # Check outputs of map for NaNs/Infs before setting
            phi_for_k_first_k = jnp.where(jnp.all(jnp.isfinite(phi_for_k_first_k)), phi_for_k_first_k, jnp.full_like(phi_for_k_first_k, jnp.nan))
            phi_rev_k_first_k = jnp.where(jnp.all(jnp.isfinite(phi_rev_k_first_k)), phi_rev_k_first_k, jnp.full_like(phi_rev_k_first_k, jnp.nan))

            # Only set if map output is finite
            phi_for_k_padded = phi_for_k_padded.at[k, :k, :, :].set(jnp.where(jnp.all(jnp.isfinite(phi_for_k_first_k)), phi_for_k_first_k, jnp.zeros_like(phi_for_k_first_k)))
            phi_rev_k_padded = phi_rev_k_padded.at[k, :k, :, :].set(jnp.where(jnp.all(jnp.isfinite(phi_rev_k_first_k)), phi_rev_k_first_k, jnp.zeros_like(phi_rev_k_first_k)))


        # Set reflection coefficients at index [k, k]
        phi_for_k_padded = phi_for_k_padded.at[k, k, :, :].set(phi_for_kk)
        phi_rev_k_padded = phi_rev_k_padded.at[k, k, :, :].set(phi_rev_kk)

        # Copy previous layers (0..k-1) into the new padded arrays
        if k > 0:
             phi_for_k_padded = phi_for_k_padded.at[:k, :, :].set(jnp.where(jnp.all(jnp.isfinite(phi_for_prev_padded[:k, :, :])), phi_for_prev_padded[:k, :, :], jnp.full_like(phi_for_prev_padded[:k, :, :], jnp.nan)))
             phi_rev_k_padded = phi_rev_k_padded.at[:k, :, :].set(jnp.where(jnp.all(jnp.isfinite(phi_rev_prev_padded[:k, :, :])), phi_rev_prev_padded[:k, :, :], jnp.full_like(phi_rev_prev_padded[:k, :, :], jnp.nan)))


        # 2. Compute Gamma_trans[k+1]
        # Initialize Gamma_trans[k+1] = phi_for[k][k] @ Sigma_rev[k]
        Gamma_trans_kp1_val = phi_for_kk @ Sigma_rev_s_val # Using potentially NaN phi_for_kk or Sigma_rev_s_val

        # Add sum_{j=0}^{k-1} phi_for[k-1][j] @ Gamma_trans[k-j]
        # Only if k > 0
        if k > 0:
             def compute_gamma_sum_step_k_scoped(j_idx):
                  # uses k, phi_for_prev_padded, Gamma_trans_prev_padded
                  # phi_for[k-1][j] is at phi_for_prev_padded[k-1, j]
                  # Gamma_trans[k-j] is at Gamma_trans_prev_padded[k-j]
                  # Check inputs for NaNs/Infs within map
                  pf_km1_j = jnp.where(jnp.all(jnp.isfinite(phi_for_prev_padded[k-1, j_idx, :, :])), phi_for_prev_padded[k-1, j_idx, :, :], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                  Gt_k_mj = jnp.where(jnp.all(jnp.isfinite(Gamma_trans_prev_padded[k-j_idx, :, :])), Gamma_trans_prev_padded[k-j_idx, :, :], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))

                  term = pf_km1_j @ Gt_k_mj

                  # Check output for NaNs/Infs
                  term = jnp.where(jnp.all(jnp.isfinite(term)), term, jnp.full_like(term, jnp.nan))
                  return term


             js_gamma_k = jnp.arange(k)
             gamma_sum_terms_k = lax.map(
                  compute_gamma_sum_step_k_scoped,
                  js_gamma_k
             ) # Shape (k, m, m)

             # Sum the terms (might result in NaN if any term is NaN)
             sum_terms = jnp.sum(gamma_sum_terms_k, axis=0)
             # Check sum for NaNs/Infs
             sum_terms = jnp.where(jnp.all(jnp.isfinite(sum_terms)), sum_terms, jnp.full_like(sum_terms, jnp.nan))

             # Add to Gamma_trans_kp1_val (might result in NaN if either is NaN)
             Gamma_trans_kp1_val = Gamma_trans_kp1_val + sum_terms


        # Check final Gamma_trans_kp1_val for NaNs/Infs before setting
        Gamma_trans_kp1_val = jnp.where(jnp.all(jnp.isfinite(Gamma_trans_kp1_val)), Gamma_trans_kp1_val, jnp.full_like(Gamma_trans_kp1_val, jnp.nan))

        # Construct next padded Gamma_trans history
        Gamma_trans_k_padded = Gamma_trans_prev_padded.at[k + 1, :, :].set(Gamma_trans_kp1_val)
        # Copy previous layers (0..k) into the new padded array (index 0..k were already set/carried)
        # This isn't strictly necessary if we copy the previous padded array and just update index k+1
        # Let's simplify and just update index k+1 on the previous padded array copy
        # The previous padded array is Gamma_trans_prev_padded.
        # Gamma_trans_k_padded = Gamma_trans_prev_padded.at[k + 1, :, :].set(Gamma_trans_kp1_val)

        # 3. Compute Sigma_rev[k+1]
        # Sigma_rev[k+1] = Sigma_rev[k] - quad_form_sym(Sigma_for[k], phi_rev[k][k].T)
        # Need Sigma_for[k]. Sigma_for_full_list[k].
        Sigma_for_k_val_for_Sigma_rev_update = Sigma_for_full_list[k] # This is Sigma_for[k]

        # Check inputs for Sigma_rev[k+1] update for NaNs/Infs
        Sigma_rev_s_val = jnp.where(jnp.all(jnp.isfinite(Sigma_rev_s_val)), Sigma_rev_s_val, jnp.full_like(Sigma_rev_s_val, jnp.nan))
        Sigma_for_k_val_for_Sigma_rev_update = jnp.where(jnp.all(jnp.isfinite(Sigma_for_k_val_for_Sigma_rev_update)), Sigma_for_k_val_for_Sigma_rev_update, jnp.full_like(Sigma_for_k_val_for_Sigma_rev_update, jnp.nan))
        phi_rev_kk_T = jnp.where(jnp.all(jnp.isfinite(phi_rev_kk.T)), phi_rev_kk.T, jnp.full_like(phi_rev_kk.T, jnp.nan)) # Use checked phi_rev_kk

        # quad_form_sym handles symmetry internally
        qf_term = quad_form_sym_jax(Sigma_for_k_val_for_Sigma_rev_update, phi_rev_kk_T)

        # Check qf_term for NaNs/Infs
        qf_term = jnp.where(jnp.all(jnp.isfinite(qf_term)), qf_term, jnp.full_like(qf_term, jnp.nan))

        # Subtract (might result in NaN)
        Sigma_rev_kp1_val = Sigma_rev_s_val - qf_term

        # Check final Sigma_rev_kp1_val for NaNs/Infs
        Sigma_rev_kp1_val = jnp.where(jnp.all(jnp.isfinite(Sigma_rev_kp1_val)), Sigma_rev_kp1_val, jnp.full_like(Sigma_rev_kp1_val, jnp.nan))


        next_carry = (phi_for_k_padded, phi_rev_k_padded, Gamma_trans_k_padded, Sigma_rev_kp1_val)

        # Store the state at step k (corresponding to Stan index s=k).
        # We need Gamma_trans[s] for s=1..p for the final output.
        # Gamma_trans_k_padded[:, :, :] at index k+1 is Gamma_trans[k+1]
        # Store Gamma_trans[k+1]
        gamma_trans_output_for_k = Gamma_trans_k_padded[k+1, :, :] # This is Gamma_trans[k+1]

        # Return the state and the Gamma_trans[k+1] for this step
        return next_carry, gamma_trans_output_for_k

    # Run the combined scan for s=0..p-1
    if p > 0:
        # Initial carry (k=0, s=0) values are computed before the scan
        # (initial_phi_for_padded, initial_phi_rev_padded, initial_Gamma_trans_padded, Sigma_rev_0)

        # Store Gamma_trans[1] in the scan output for k=0
        # The scan output will be stacked Gamma_trans[1], Gamma_trans[2], ..., Gamma_trans[p]
        final_carry_step2, gamma_trans_scan_outputs = lax.scan(
            step2_combined_scan_body,
            init_carry_step2_combined,
            scan2_inputs_combined, # Inputs (S_for_list, P, Sigma_for) for s=0..p-1
            length=p
        )

        # Final phi_list: phi_for[p-1][0..p-1] from the last padded matrix (final_carry_step2[0])
        # Ensure final_carry_step2[0] is finite before slicing
        final_phi_for_padded = jnp.where(jnp.all(jnp.isfinite(final_carry_step2[0])), final_carry_step2[0], jnp.full_like(final_carry_step2[0], jnp.nan))

        # Extract the final phi matrices [phi_{p-1}[0], ..., phi_{p-1}[p-1]]
        phi_list_final = [final_phi_for_padded[p-1, i, :, :] for i in range(p)]

        # Final gamma_list: Gamma_trans[1..p] from the scan outputs
        # gamma_trans_scan_outputs is already stacked Gamma_trans[1], ..., Gamma_trans[p]
        gamma_list_final_stacked = gamma_trans_scan_outputs # shape (p, m, m)

        # Ensure final gamma_list is finite before converting to list
        gamma_list_final_stacked = jnp.where(jnp.all(jnp.isfinite(gamma_list_final_stacked)), gamma_list_final_stacked, jnp.full_like(gamma_list_final_stacked, jnp.nan))

        gamma_list_final = [gamma_list_final_stacked[i, :, :] for i in range(p)] # Indices 0 to p-1

    else: # p == 0, handled above. Should not reach here.
         raise ValueError("Should not reach this part for p==0")


    return phi_list_final, gamma_list_final


# This should be the final function exposed
def make_stationary_var_transformation_jax(Sigma: ArrayLike, A_list: List[ArrayLike], m: int, p: int) -> Tuple[List[jax.Array], List[jax.Array]]:
    """
    JAX: Transform A matrices to stationary VAR parameters (phi) and autocovariances (gamma).
    Equivalent to Stan's make_stationary_var_transformation.

    Args:
        Sigma: m×m error covariance matrix (JAX array).
        A_list: List of p JAX arrays, [A_0, ..., A_{p-1}], each m×m.
        m: Dimension of the VAR.
        p: Order of the VAR.

    Returns:
        Tuple of (phi_list, gamma_list):
            - phi_list: List of p VAR coefficient matrices [phi_0, ..., phi_{p-1}].
                        These are the final VAR coefficients (phi_{p-1}[0..p-1] in Stan).
            - gamma_list: List of p autocovariance matrices [Gamma_trans[1], ..., Gamma_trans[p]].
                          Note: Gamma_trans[k] corresponds to autocovariance at lag k.
                          So this list is [Gamma_1, ..., Gamma_p].
    """
    # Handle VAR(0) case separately
    if p == 0:
        Sigma_jax = jnp.asarray(Sigma, dtype=_DEFAULT_DTYPE)
        return [], [Sigma_jax] # phi_list empty, gamma_list is [Gamma_0 = Sigma]

    # Convert A matrices to P matrices using JAX
    # Stack A_list for vmap if AtoP_jax was vmapped, but it's not. Call sequentially.
    P_list_jax = [AtoP_jax(A_list[i], m) for i in range(p)]

    # Check for NaNs/Infs in P_list after AtoP
    P_list_jax = [jnp.where(jnp.all(jnp.isfinite(P)), P, jnp.full_like(P, jnp.nan)) for P in P_list_jax]
    # Stack the checked P_list for rev_mapping
    P_list_jax_stacked = jnp.stack(P_list_jax, axis=0) # shape (p, m, m)

    # Apply reverse mapping (Levinson-Durbin backward)
    return rev_mapping_jax(P_list_jax_stacked, Sigma, m, p)


def create_companion_matrix_jax(Phi_list: List[ArrayLike], p: int, m: int) -> jax.Array:
    """
    JAX: Create the companion matrix from VAR parameters.

    Args:
        Phi_list: List of p JAX arrays [phi_0, ..., phi_{p-1}], each m×m.
        p: Order of the VAR.
        m: Dimension of the VAR.

    Returns:
        mp×mp companion matrix.
    """
    if p == 0:
        # VAR(0) has no companion matrix in the standard sense.
        # Return empty (0x0).
        return jnp.empty((0, 0), dtype=_DEFAULT_DTYPE)

    mp = m * p
    comp_matrix = jnp.zeros((mp, mp), dtype=_DEFAULT_DTYPE)

    # Add coefficient matrices to first block row
    # Stack phi matrices horizontally: [phi_0 | phi_1 | ... | phi_{p-1}] (m x mp)
    # Ensure inputs are finite before stacking
    Phi_list_finite = [jnp.where(jnp.all(jnp.isfinite(phi)), phi, jnp.full_like(phi, jnp.nan)) for phi in Phi_list]
    phi_stacked = jnp.concatenate(Phi_list_finite, axis=1)

    # Ensure stacked phi is finite before setting
    phi_stacked = jnp.where(jnp.all(jnp.isfinite(phi_stacked)), phi_stacked, jnp.full_like(phi_stacked, jnp.nan))
    comp_matrix = comp_matrix.at[:m, :].set(phi_stacked)

    # Add identity matrix below the first block row if p > 1
    if p > 1:
        identity_block = jnp.eye(m * (p - 1), dtype=_DEFAULT_DTYPE)
        # Ensure comp_matrix slice is finite before setting (it should be zeros)
        comp_matrix = comp_matrix.at[m:, :-m].set(identity_block)

    # Ensure the final matrix is finite
    comp_matrix = jnp.where(jnp.all(jnp.isfinite(comp_matrix)), comp_matrix, jnp.full_like(comp_matrix, jnp.nan))

    return comp_matrix

# Test function
def check_stationarity_jax(phi_list: List[ArrayLike], m: int, p: int) -> jax.Array:
    """
    JAX: Check if a VAR process is stationary based on its companion matrix.

    Args:
        phi_list: List of p JAX arrays [phi_0, ..., phi_{p-1}].
        m: Dimension of the VAR.
        p: Order of the VAR.

    Returns:
        Boolean JAX array: True if stationary, False otherwise.
    """
    if p == 0:
        return jnp.array(True, dtype=jnp.bool_) # VAR(0) is stationary

    # Create companion matrix. This function already includes isfinite checks.
    companion = create_companion_matrix_jax(phi_list, p, m)

    # Check if the companion matrix is finite
    if not jnp.all(jnp.isfinite(companion)):
        return jnp.array(False, dtype=jnp.bool_) # Not stationary if matrix is invalid

    # Compute eigenvalues
    try:
       # jsl.eigvals can return complex numbers. This is expected.
       eigenvalues = jsl.eigvals(companion)

       # Check for NaNs/Infs in eigenvalues. If any are NaN/Inf, the process is not stationary.
       if not jnp.all(jnp.isfinite(eigenvalues)):
            return jnp.array(False, dtype=jnp.bool_) # Not stationary if eigenvalues are invalid

       # Check if all eigenvalues' magnitudes are strictly less than 1.
       # Use a small tolerance for numerical stability, matching PyMC test file (1.0 - 1e-9).
       max_abs_eigenvalue = jnp.max(jnp.abs(eigenvalues))

       # Return True if max magnitude < 1.0 - tolerance
       return max_abs_eigenvalue < (1.0 - 1e-9)
    except Exception:
       # Handle potential numerical issues in eigenvalue computation (rare with isfinite check, but possible).
       # If eigvals raises an error, the matrix is likely numerically singular or ill-conditioned,
       # implying the process is not stationary in a practical sense.
       return jnp.array(False, dtype=jnp.bool_)