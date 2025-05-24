# stationary_prior_jax.py - Adding debug prints to stationarity check

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
# No lax.scan needed in this version
# from jax import lax
from jax.typing import ArrayLike
from typing import List, Tuple

# Use jax.debug.print for printing within JIT (although this version isn't JITted, it's good practice)
import jax.debug as jdebug

# Ensure JAX is configured for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# Numerical stability jitter
_JITTER_PRIOR = 1e-8

def symmetrize_jax(A: ArrayLike) -> jax.Array:
    """JAX: Symmetrize a matrix by computing (A + A.T)/2."""
    A_jax = jnp.asarray(A, dtype=_DEFAULT_DTYPE)
    return 0.5 * (A_jax + A_jax.T)

def sqrtm_jax(A: ArrayLike) -> jax.Array:
    """
    JAX: Matrix square root computation using eigendecomposition.
    Equivalent to Stan's sqrtm function (clips negative eigenvalues).
    Includes a tiny jitter before eigh for numerical stability.
    """
    A_sym = symmetrize_jax(A)

    A_sym_reg = A_sym + _JITTER_PRIOR * 1e-3 * jnp.eye(A_sym.shape[0], dtype=_DEFAULT_DTYPE)

    evals, evecs = jsl.eigh(A_sym_reg)

    evals_clipped = jnp.maximum(evals, _JITTER_PRIOR)

    root_root_evals = jnp.sqrt(jnp.sqrt(evals_clipped))

    eprod = evecs @ jnp.diag(root_root_evals)

    result = eprod @ eprod.T
    result = jnp.where(jnp.all(jnp.isfinite(result)), result, jnp.full_like(result, jnp.nan))
    return result


def tcrossprod_jax(A: ArrayLike) -> jax.Array:
    """JAX: Compute A @ A.T (equivalent to Stan's tcrossprod)."""
    A_jax = jnp.asarray(A, dtype=_DEFAULT_DTYPE)
    result = A_jax @ A_jax.T
    result = jnp.where(jnp.all(jnp.isfinite(result)), result, jnp.full_like(result, jnp.nan))
    return result

def quad_form_sym_jax(A: ArrayLike, B: ArrayLike) -> jax.Array:
    """
    JAX: Compute B.T @ A @ B (symmetric quadratic form).
    Ensures A is treated as symmetric.
    """
    A_sym = symmetrize_jax(A)
    B_jax = jnp.asarray(B, dtype=_DEFAULT_DTYPE)
    result = B_jax.T @ A_sym @ B_jax
    result = jnp.where(jnp.all(jnp.isfinite(result)), result, jnp.full_like(result, jnp.nan))
    return result

def AtoP_jax(A: ArrayLike, m: int) -> jax.Array:
    """
    JAX: Transform A to P. Includes robustness. Assumes m >= 1.
    """
    A_jax = jnp.asarray(A, dtype=_DEFAULT_DTYPE)

    B = tcrossprod_jax(A_jax)

    # Add 1.0 to diagonal elements of B
    B_checked = jnp.where(jnp.all(jnp.isfinite(B)), B, jnp.full_like(B, jnp.nan))
    B_plus_I = B_checked + jnp.eye(m, dtype=_DEFAULT_DTYPE)

    sqrtB = sqrtm_jax(B_plus_I)

    sqrtB_reg = sqrtB + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)

    sqrtB_reg_checked = jnp.where(jnp.all(jnp.isfinite(sqrtB_reg)), sqrtB_reg, jnp.full_like(sqrtB_reg, jnp.nan))
    A_jax_checked = jnp.where(jnp.all(jnp.isfinite(A_jax)), A_jax, jnp.full_like(A_jax, jnp.nan))

    try:
       P = jsl.solve(sqrtB_reg_checked, A_jax_checked, assume_a='pos')
       P = jnp.where(jnp.all(jnp.isfinite(P)), P, jnp.full_like(P, jnp.nan))
       return P
    except Exception:
       return jnp.full_like(A_jax, jnp.nan)


def rev_mapping_jax(P_list: List[ArrayLike], Sigma: ArrayLike, m: int, p: int) -> Tuple[List[jax.Array], List[jax.Array]]:
    """
    JAX: Equivalent to Stan's rev_mapping function using Python loops.
    Implements the backward Levinson-Durbin recursion.
    Includes robustness for sqrtm and solve.
    Assumes p >= 1 and m >= 1.
    """
    Sigma_jax = jnp.asarray(Sigma, dtype=_DEFAULT_DTYPE)
    P_list_jax = [jnp.asarray(P, dtype=_DEFAULT_DTYPE) for P in P_list]

    Sigma_jax = jnp.where(jnp.all(jnp.isfinite(Sigma_jax)), Sigma_jax, jnp.full_like(Sigma_jax, jnp.nan))
    P_list_jax = [jnp.where(jnp.all(jnp.isfinite(P)), P, jnp.full_like(P, jnp.nan)) for P in P_list_jax]


    # Initialize lists for Step 1
    Sigma_for = [jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)] * (p + 1)
    S_for_list = [jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)] * (p + 1)

    # Initialize lists for Step 2
    Sigma_rev = [jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)] * (p + 1)
    Gamma_trans = [jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)] * (p + 1)

    # Initialize phi matrices (store as list of lists for direct translation)
    phi_for = [[jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE) for _ in range(p)] for _ in range(p)]
    phi_rev = [[jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE) for _ in range(p)] for _ in range(p)]

    # Step 1 (Python loop)
    Sigma_for[p] = Sigma_jax
    S_for_list[p] = sqrtm_jax(Sigma_jax)

    # Check initial values and propagate NaNs if they appeared
    Sigma_for[p] = jnp.where(jnp.all(jnp.isfinite(Sigma_for[p])), Sigma_for[p], jnp.full_like(Sigma_for[p], jnp.nan))
    S_for_list[p] = jnp.where(jnp.all(jnp.isfinite(S_for_list[p])), S_for_list[p], jnp.full_like(S_for_list[p], jnp.nan))


    # Loop s from 1 to p (Stan indexing) => loop index from 0 to p-1
    for s_idx in range(p):
        s = s_idx + 1 # Stan's s is 1-indexed

        # Check if previous iteration introduced NaNs
        if jnp.any(~jnp.isfinite(Sigma_for[p-s+1])) or jnp.any(~jnp.isfinite(S_for_list[p-s+1])):
             # If NaNs are already present, propagate them to the next iteration's outputs
             Sigma_for[p-s] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
             S_for_list[p-s] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
             continue # Skip calculations for this s if inputs are NaN


        P_current = P_list_jax[p - s] # P_list_jax is 0-indexed, P[p-s+1] in Stan is P_list_jax[p-s]

        # Compute S_for
        S_for_val = -tcrossprod_jax(P_current) + jnp.eye(m, dtype=_DEFAULT_DTYPE)
        S_for_val_checked = jnp.where(jnp.all(jnp.isfinite(S_for_val)), S_for_val, jnp.full_like(S_for_val, jnp.nan))

        # Compute S_rev from S_for_val
        S_rev_curr = sqrtm_jax(S_for_val_checked)
        S_rev_curr_checked = jnp.where(jnp.all(jnp.isfinite(S_rev_curr)), S_rev_curr, jnp.full_like(S_rev_curr, jnp.nan))

        # Compute Q = S_rev_curr.T @ Sigma_for[s] @ S_rev_curr
        Q = quad_form_sym_jax(Sigma_for[p-s+1], S_rev_curr_checked)
        Q_checked = jnp.where(jnp.all(jnp.isfinite(Q)), Q, jnp.full_like(Q, jnp.nan))

        # Compute sqrtm(Q)
        sqrtQ = sqrtm_jax(Q_checked)
        sqrtQ_checked = jnp.where(jnp.all(jnp.isfinite(sqrtQ)), sqrtQ, jnp.full_like(sqrtQ, jnp.nan))

        # Compute MID = S_rev_curr^(-1) @ sqrtQ
        S_rev_curr_reg = S_rev_curr_checked + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
            MID = jsl.solve(S_rev_curr_reg, sqrtQ_checked, assume_a='pos')
        except Exception:
             MID = jnp.full_like(sqrtQ_checked, jnp.nan)

        MID_checked = jnp.where(jnp.all(jnp.isfinite(MID)), MID, jnp.full_like(MID, jnp.nan))

        # Compute S_for_list[p-s] = MID @ S_rev_curr^(-1)
        S_rev_curr_T_reg = S_rev_curr_checked.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
            S_for_list_s_minus_1 = jsl.solve(S_rev_curr_T_reg, MID_checked.T, assume_a='pos').T
        except Exception:
            S_for_list_s_minus_1 = jnp.full_like(MID_checked.T, jnp.nan).T

        S_for_list_s_minus_1_checked = jnp.where(jnp.all(jnp.isfinite(S_for_list_s_minus_1)), S_for_list_s_minus_1, jnp.full_like(S_for_list_s_minus_1, jnp.nan))
        S_for_list[p-s] = S_for_list_s_minus_1_checked


        # Compute Sigma_for[p-s] = tcrossprod(S_for_list[p-s])
        Sigma_for_s_minus_1 = tcrossprod_jax(S_for_list[p-s])
        Sigma_for_s_minus_1_checked = jnp.where(jnp.all(jnp.isfinite(Sigma_for_s_minus_1)), Sigma_for_s_minus_1, jnp.full_like(Sigma_for_s_minus_1, jnp.nan))
        Sigma_for[p-s] = Sigma_for_s_minus_1_checked


    # Step 2 (Python loop)
    Sigma_rev[0] = Sigma_for[0]
    Gamma_trans[0] = Sigma_for[0]

    # Check initial values for step 2 and propagate NaNs if they appeared
    Sigma_rev[0] = jnp.where(jnp.all(jnp.isfinite(Sigma_rev[0])), Sigma_rev[0], jnp.full_like(Sigma_rev[0], jnp.nan))
    Gamma_trans[0] = jnp.where(jnp.all(jnp.isfinite(Gamma_trans[0])), Gamma_trans[0], jnp.full_like(Gamma_trans[0], jnp.nan))


    # Loop s from 0 to p-1 (Stan indexing) => loop index from 0 to p-1
    for s in range(p): # s is 0-indexed loop index here, matches Stan index
        # Check if previous iteration introduced NaNs in inputs needed for this step
        if jnp.any(~jnp.isfinite(Sigma_rev[s])) or jnp.any(~jnp.isfinite(S_for_list[s])) or jnp.any(~jnp.isfinite(P_list_jax[s])):
             # Propagate NaNs to outputs of this step (phi[s][s], phi_rev[s][s], Gamma_trans[s+1], Sigma_rev[s+1])
             if s < p: # phi_for/rev[s][s] are computed
                 phi_for[s][s] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
                 phi_rev[s][s] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
                 if s + 1 <= p: # Gamma_trans[s+1] is computed
                    Gamma_trans[s+1] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
                 if s < p-1: # Sigma_rev[s+1] is computed
                    Sigma_rev[s+1] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)

             # Propagate NaNs to outputs of inner loops (phi[s][k], phi_rev[s][k], Gamma_trans[s+1] summation)
             if s >= 1:
                  for k in range(s):
                      phi_for[s][k] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
                      phi_rev[s][k] = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
                  # Gamma_trans[s+1] summation term also relies on inputs marked NaN above

             continue # Skip calculations for this s if inputs are NaN


        S_for = S_for_list[s]
        S_rev = sqrtm_jax(Sigma_rev[s]) # S_rev from Sigma_rev[s]

        # Check inputs for phi_for/rev[s][s] solve and propagate
        S_for_checked = jnp.where(jnp.all(jnp.isfinite(S_for)), S_for, jnp.full_like(S_for, jnp.nan))
        S_rev_checked = jnp.where(jnp.all(jnp.isfinite(S_rev)), S_rev, jnp.full_like(S_rev, jnp.nan))
        P_s = P_list_jax[s]
        P_s_checked = jnp.where(jnp.all(jnp.isfinite(P_s)), P_s, jnp.full_like(P_s, jnp.nan))


        # Compute phi_for[s][s]
        temp_phi_for = S_for_checked @ P_s_checked
        S_rev_T_reg = S_rev_checked.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
            phi_for_ss = jsl.solve(S_rev_T_reg, temp_phi_for.T, assume_a='pos').T
        except Exception:
            phi_for_ss = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
        phi_for[s][s] = jnp.where(jnp.all(jnp.isfinite(phi_for_ss)), phi_for_ss, jnp.full_like(phi_for_ss, jnp.nan))


        # Compute phi_rev[s][s]
        temp_phi_rev = S_rev_checked @ P_s_checked.T
        S_for_T_reg = S_for_checked.T + _JITTER_PRIOR * jnp.eye(m, dtype=_DEFAULT_DTYPE)
        try:
            phi_rev_ss = jsl.solve(S_for_T_reg, temp_phi_rev.T, assume_a='pos').T
        except Exception:
            phi_rev_ss = jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE)
        phi_rev[s][s] = jnp.where(jnp.all(jnp.isfinite(phi_rev_ss)), phi_rev_ss, jnp.full_like(phi_rev_ss, jnp.nan))


        # Compute Gamma_trans[s+1] (Initialization)
        if s + 1 <= p: # Ensure index is valid
             Gamma_trans_sp1_val = phi_for[s][s] @ Sigma_rev[s] # Use potentially NaN phi_for[s][s]

             # Summation for phi_for[s][k], phi_rev[s][k] for k=0..s-1
             if s >= 1:
                 gamma_sum_terms = []
                 for k in range(s):
                      # Check inputs and propagate NaNs
                      pf_sm1_k = jnp.where(jnp.all(jnp.isfinite(phi_for[s-1][k])), phi_for[s-1][k], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
                      Gt_sp1_km1 = jnp.where(jnp.all(jnp.isfinite(Gamma_trans[s+1-k-1])), Gamma_trans[s+1-k-1], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))

                      term = pf_sm1_k @ Gt_sp1_km1
                      term_checked = jnp.where(jnp.all(jnp.isfinite(term)), term, jnp.full_like(term, jnp.nan))
                      gamma_sum_terms.append(term_checked)

                 sum_terms = jnp.sum(jnp.stack(gamma_sum_terms, axis=0), axis=0) if gamma_sum_terms else jnp.zeros((m,m), dtype=_DEFAULT_DTYPE) # Handle empty sum
                 sum_terms_checked = jnp.where(jnp.all(jnp.isfinite(sum_terms)), sum_terms, jnp.full_like(sum_terms, jnp.nan))

                 Gamma_trans_sp1_val = Gamma_trans_sp1_val + sum_terms_checked # Add to initial Gamma_trans[s+1] value

             # Assign final Gamma_trans[s+1]
             Gamma_trans[s+1] = jnp.where(jnp.all(jnp.isfinite(Gamma_trans_sp1_val)), Gamma_trans_sp1_val, jnp.full_like(Gamma_trans_sp1_val, jnp.nan))


        # Compute Sigma_rev[s+1] if s < p-1
        if s < p - 1:
            # Check inputs and propagate NaNs
            Sr_s = jnp.where(jnp.all(jnp.isfinite(Sigma_rev[s])), Sigma_rev[s], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
            Sf_s = jnp.where(jnp.all(jnp.isfinite(Sigma_for[s])), Sigma_for[s], jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))
            Prs_T = jnp.where(jnp.all(jnp.isfinite(phi_rev[s][s].T)), phi_rev[s][s].T, jnp.full((m,m), jnp.nan, dtype=_DEFAULT_DTYPE))

            qf_term = quad_form_sym_jax(Sf_s, Prs_T)
            qf_term_checked = jnp.where(jnp.all(jnp.isfinite(qf_term)), qf_term, jnp.full_like(qf_term, jnp.nan))

            Sigma_rev_sp1_val = Sr_s - qf_term_checked
            Sigma_rev[s+1] = jnp.where(jnp.all(jnp.isfinite(Sigma_rev_sp1_val)), Sigma_rev_sp1_val, jnp.full_like(Sigma_rev_sp1_val, jnp.nan))


    # Prepare output
    # phi_list contains [phi_for[p-1][0], ..., phi_for[p-1][p-1]]
    phi_list_final = [phi_for[p-1][i] for i in range(p)]

    # gamma_list contains [Gamma_trans[1], ..., Gamma_trans[p]]
    gamma_list_final = [Gamma_trans[i] for i in range(1, p + 1)]


    return phi_list_final, gamma_list_final


# This should be the final function exposed
def make_stationary_var_transformation_jax(Sigma: ArrayLike, A_list: List[ArrayLike], m: int, p: int) -> Tuple[List[jax.Array], List[jax.Array]]:
    """
    JAX: Transform A matrices to stationary VAR parameters (phi) and autocovariances (gamma).
    Equivalent to Stan's make_stationary_var_transformation using Python loops.
    Assumes p >= 1 and m >= 1.
    """
    # Check input Sigma for NaNs and propagate
    Sigma_jax = jnp.where(jnp.all(jnp.isfinite(Sigma)), jnp.asarray(Sigma, dtype=_DEFAULT_DTYPE), jnp.full_like(Sigma, jnp.nan))

    # Convert A matrices to P matrices using JAX
    # Check input A_list for NaNs before AtoP and propagate
    A_list_jax = [jnp.where(jnp.all(jnp.isfinite(A)), jnp.asarray(A, dtype=_DEFAULT_DTYPE), jnp.full_like(A, jnp.nan)) for A in A_list]

    P_list_jax = [AtoP_jax(A_list_jax[i], m) for i in range(p)]

    # Check for NaNs/Infs in P_list after AtoP and propagate
    P_list_jax_checked = [jnp.where(jnp.all(jnp.isfinite(P)), P, jnp.full_like(P, jnp.nan)) for P in P_list_jax]

    # Apply reverse mapping (Levinson-Durbin backward)
    # This function handles NaNs internally and propagates them.
    return rev_mapping_jax(P_list_jax_checked, Sigma_jax, m, p)


def create_companion_matrix_jax(Phi_list: List[ArrayLike], p: int, m: int) -> jax.Array:
    """
    JAX: Create the companion matrix from VAR parameters.
    Assumes p >= 1 and m >= 1.
    """
    mp = m * p
    comp_matrix = jnp.zeros((mp, mp), dtype=_DEFAULT_DTYPE)

    # Add coefficient matrices to first block row
    # Stack phi matrices horizontally: [phi_0 | phi_1 | ... | phi_{p-1}] (m x mp)
    # Ensure inputs are finite before stacking and propagate
    Phi_list_finite = [jnp.where(jnp.all(jnp.isfinite(phi)), phi, jnp.full_like(phi, jnp.nan)) for phi in Phi_list]
    phi_stacked = jnp.concatenate(Phi_list_finite, axis=1)

    # Ensure stacked phi is finite before setting and propagate
    phi_stacked_checked = jnp.where(jnp.all(jnp.isfinite(phi_stacked)), phi_stacked, jnp.full_like(phi_stacked, jnp.nan))
    comp_matrix = comp_matrix.at[:m, :].set(phi_stacked_checked)

    # Add identity matrix below the first block row if p > 1
    if p > 1:
        identity_block = jnp.eye(m * (p - 1), dtype=_DEFAULT_DTYPE)
        comp_matrix = comp_matrix.at[m:, :-m].set(identity_block)

    # Ensure the final matrix is finite and propagate
    comp_matrix_final = jnp.where(jnp.all(jnp.isfinite(comp_matrix)), comp_matrix, jnp.full_like(comp_matrix, jnp.nan))

    return comp_matrix_final

def check_stationarity_jax(phi_list: List[ArrayLike], m: int, p: int) -> jax.Array:
    """
    JAX: Check if a VAR process is stationary based on its companion matrix.
    Assumes p >= 1 and m >= 1.
    Checks if eigenvalues are strictly less than 1.0.
    """
    companion = create_companion_matrix_jax(phi_list, p, m)

    if not jnp.all(jnp.isfinite(companion)):
        return jnp.array(False, dtype=jnp.bool_)

    try:
       eigenvalues = jsl.eigvals(companion)

       if not jnp.all(jnp.isfinite(eigenvalues)):
            return jnp.array(False, dtype=jnp.bool_)

       max_abs_eigenvalue = jnp.max(jnp.abs(eigenvalues))

       # Check if max absolute eigenvalue is strictly less than 1.0
       result = max_abs_eigenvalue < 1.0 # <-- Changed from (1.0 - 1e-9)
       return result

    except Exception:
       return jnp.array(False, dtype=jnp.bool_)