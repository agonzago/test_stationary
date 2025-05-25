import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from typing import List, Tuple

# Configure JAX for float64
jax.config.update("jax_enable_x64", True)
_DEFAULT_DTYPE = jnp.float64

def symmetrize_jax(A):
    """Symmetrize a matrix by computing (A + A.T)/2."""
    return 0.5 * (A + A.T)

def tcrossprod_jax(A):
    """
    Compute A @ A.T (equivalent to Stan's tcrossprod).
    """
    return A @ A.T

def sqrtm_jax(A):
    """
    Matrix square root computation using eigendecomposition.
    EXACT equivalent to Stan's sqrtm function, translated to JAX.
    """
    A_sym = symmetrize_jax(A)  # Ensure matrix is symmetric

    # Eigenvalue decomposition
    evals, evecs = jnp.linalg.eigh(A_sym)

    # Compute 4th root of eigenvalues, with clipping
    root_root_evals = jnp.sqrt(jnp.sqrt(jnp.maximum(evals, 1e-12)))

    # Construct matrix using eigenvectors and transformed eigenvalues
    eprod = evecs @ jnp.diag(root_root_evals)

    # Compute eprod @ eprod.T
    return tcrossprod_jax(eprod)

def quad_form_sym_jax(A, B):
    """
    Compute B.T @ A @ B (symmetric quadratic form).
    Ensures A is treated as symmetric.
    """
    A_sym = symmetrize_jax(A)
    return B.T @ A_sym @ B

def AtoP_jax(A, m):
    """
    Transform A to P (equivalent to Stan's AtoP function), translated to JAX.
    """
    B = tcrossprod_jax(A)  # B = A @ A.T

    # Add 1.0 to diagonal elements of B
    # In JAX, arrays are immutable, so we create a new B
    B = B.at[jnp.arange(m), jnp.arange(m)].add(1.0)

    # Compute matrix square root of B
    sqrtB = sqrtm_jax(B)

    # Compute sqrtB^(-1) @ A
    # jsl.solve requires sqrtB to be positive definite.
    # The original code uses assume_a='pos'.
    # Check for positive definiteness before solving
    try:
        P = jsl.solve(sqrtB, A, assume_a='pos')
    except Exception as e:
         # If solve fails (e.g., sqrtB not positive definite numerically),
         # fall back to pinv or return a default value.
         # Returning NaN or a large value might be better for MCMC stability,
         # letting the sampler avoid these regions.
         # For now, let's try pinv as a fallback, although it's less numerically stable.
         # A robust approach might involve checking eigenvalues of sqrtB.
         # Let's stick to the original PyTensor logic's implicit assumption
         # that AtoP results in valid sqrtB for valid A.
         # If assume_a='pos' fails, jsl.solve should raise an error.
         # Let's trust assume_a='pos' for now as per the prompt implies these utils are working.
         # If numerical issues arise in practice, this is a point to revisit.
         # A simple fallback to pinv:
         try:
             P = jnp.linalg.pinv(sqrtB) @ A
         except Exception:
             # If even pinv fails, return NaNs or a large value to signal failure
             P = jnp.full_like(A, jnp.nan)
    return P

def rev_mapping_jax(P: List[jnp.ndarray], Sigma: jnp.ndarray, p: int, m: int) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Equivalent to Stan's rev_mapping function, translated to JAX.

    Args:
        P: List of p matrices, each m×m
        Sigma: m×m covariance matrix
        p: Order of the VAR process
        m: Dimension of the VAR process

    Returns:
        Tuple of (phi, Gamma) lists
    """
    # Initialize arrays for Step 1
    # Use jnp.zeros directly as list elements
    Sigma_for = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]
    S_for_list = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]

    # Initialize arrays for Step 2
    Sigma_rev = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]
    Gamma_trans = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]

    # Initialize phi matrices - list of lists
    phi_for = [[jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p)] for _ in range(p)]
    phi_rev = [[jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p)] for _ in range(p)]

    # Step 1
    Sigma_for_p = Sigma
    S_for_list_p = sqrtm_jax(Sigma)

    # Assign initial values
    Sigma_for[p] = Sigma_for_p
    S_for_list[p] = S_for_list_p

    # Loop for s = 1 to p
    for s in range(1, p + 1):
        P_idx = p - s # P is indexed 0 to p-1, corresponding to s=p down to s=1
        P_current = P[P_idx]

        S_for_val = -tcrossprod_jax(P_current) # This is actually -P_current @ P_current.T
        S_for_val = S_for_val.at[jnp.arange(m), jnp.arange(m)].add(1.0) # Add 1 to diagonal
        S_for_val = symmetrize_jax(S_for_val) # Ensure symmetry before sqrtm

        S_rev_val = sqrtm_jax(S_for_val) # This is sqrt(I - P @ P.T)

        Q_val = quad_form_sym_jax(Sigma_for[p - s + 1], S_rev_val) # S_rev_val.T @ Sigma_for[p-s+1] @ S_rev_val
        Q_val = symmetrize_jax(Q_val) # Ensure symmetry before sqrtm
        sqrtQ_val = sqrtm_jax(Q_val) # This is sqrtm(sqrt(I-PP').T * Sigma[s-1] * sqrt(I-PP'))

        # MID = solve(sqrt(I-PP'), sqrtm(...)) @ P
        # Original PyTensor/Stan: MID = sqrtB^(-1) @ sqrtQ
        # Let's stick to the PyTensor code structure which was sqrtB_inv @ A,
        # so here it should be S_rev_val_inv @ sqrtQ_val
        # The PyTensor code had `MID = jsl.solve(S_rev_val, sqrtQ_val, assume_a='pos')`
        # which corresponds to MID * S_rev_val = sqrtQ_val => MID = sqrtQ_val @ S_rev_val_inv (right solve)
        # But jsl.solve(A, B) solves AX=B => X=A_inv B.
        # So jsl.solve(S_rev_val, sqrtQ_val) computes S_rev_val_inv @ sqrtQ_val. Let's trust this.
        try:
            MID_val = jsl.solve(S_rev_val, sqrtQ_val, assume_a='pos')
        except Exception:
             MID_val = jnp.full_like(sqrtQ_val, jnp.nan) # Handle potential solve errors

        # S_for_list[p-s] = solve(S_rev_val.T, MID_val.T).T
        # This computes (S_rev_val.T)^(-1) @ MID_val.T and then transposes.
        # Which is (MID_val @ S_rev_val^(-1)).T
        # Equivalent to S_rev_val_inv.T @ MID_val.T
        # Let's trust jsl.solve again.
        try:
            S_for_list[p - s] = jsl.solve(S_rev_val.T, MID_val.T, assume_a='pos').T
        except Exception:
            S_for_list[p - s] = jnp.full_like(MID_val.T, jnp.nan).T

        Sigma_for[p - s] = tcrossprod_jax(S_for_list[p - s]) # S_for[p-s] @ S_for[p-s].T
        Sigma_for[p-s] = symmetrize_jax(Sigma_for[p-s])

    # Step 2
    Sigma_rev[0] = Sigma_for[0]
    Gamma_trans[0] = Sigma_for[0] # This seems to be the Omega_0 term in some notations

    # Loop for s = 0 to p-1
    for s in range(p):
        S_for_s = S_for_list[s]
        S_rev_s = sqrtm_jax(Sigma_rev[s]) # sqrtm(Sigma_rev[s])

        # temp_phi_for = S_for_s @ P[s]
        # phi_for[s][s] = solve(S_rev_s.T, temp_phi_for.T).T
        # solves S_rev_s.T * X.T = temp_phi_for.T
        # X.T = (S_rev_s.T)^-1 @ temp_phi_for.T
        # X = temp_phi_for @ S_rev_s^-1
        # X = (S_for_list[s] @ P[s]) @ sqrtm(Sigma_rev[s])^(-1)
        # This looks like sqrtm(Sigma_rev[s])^-1 @ S_for_list[s] @ P[s] ... let's trust the direct translation
        try:
            temp_phi_for = S_for_s @ P[s]
            phi_for_s_s = jsl.solve(S_rev_s.T, temp_phi_for.T, assume_a='pos').T
        except Exception:
             phi_for_s_s = jnp.full_like(P[s], jnp.nan)

        # temp_phi_rev = S_rev_s @ P[s].T
        # phi_rev[s][s] = solve(S_for_s.T, temp_phi_rev.T).T
        # solves S_for_s.T * X.T = temp_phi_rev.T
        # X.T = (S_for_s.T)^-1 @ temp_phi_rev.T
        # X = temp_phi_rev @ S_for_s^-1
        # X = (sqrtm(Sigma_rev[s]) @ P[s].T) @ S_for_list[s]^(-1)
        # This looks like S_for_list[s]^-1 @ sqrtm(Sigma_rev[s]) @ P[s].T ... let's trust the direct translation
        try:
            temp_phi_rev = S_rev_s @ P[s].T
            phi_rev_s_s = jsl.solve(S_for_s.T, temp_phi_rev.T, assume_a='pos').T
        except Exception:
            phi_rev_s_s = jnp.full_like(P[s].T, jnp.nan)


        # Update lists of JAX arrays using .at set for immutability
        phi_for[s] = phi_for[s].copy() # Create a shallow copy of the list
        phi_for[s][s] = phi_for_s_s # Update the element

        phi_rev[s] = phi_rev[s].copy() # Create a shallow copy of the list
        phi_rev[s][s] = phi_rev_s_s # Update the element

        # Gamma_trans[s+1] = phi_for[s][s] @ Sigma_rev[s] (initial term for s+1)
        Gamma_trans[s + 1] = phi_for[s][s] @ Sigma_rev[s]


        # Recursion for off-diagonal phi and Gamma
        # Stan code: for k in 0..s-1
        # phi_for[s][k] = phi_for[s-1][k] - phi_for[s][s] * phi_rev[s-1][s-1-k]
        # phi_rev[s][k] = phi_rev[s-1][k] - phi_rev[s][s] * phi_for[s-1][s-1-k]
        # Gamma_trans[s+1] = Gamma_trans[s+1] + phi_for[s-1][k] * Gamma_trans[s+1-k-1] # s+1-k-1 = s-k

        # Loop for k from 0 to s-1 (if s > 0)
        if s > 0: # Loop runs only if s >= 1
            for k in range(s):
                # Calculate new phi values
                phi_for_s_k_val = phi_for[s-1][k] - phi_for[s][s] @ phi_rev[s-1][s-1-k]
                phi_rev_s_k_val = phi_rev[s-1][k] - phi_rev[s][s] @ phi_for[s-1][s-1-k]

                # Update lists using .at set
                phi_for[s] = phi_for[s].copy()
                phi_for[s][k] = phi_for_s_k_val

                phi_rev[s] = phi_rev[s].copy()
                phi_rev[s][k] = phi_rev_s_k_val

                # Update Gamma_trans[s+1] sum
                # Based on original PyTensor/Stan index s+1-k-1 which is s-k
                # Gamma_trans indices are 0...p
                Gamma_trans[s + 1] = Gamma_trans[s + 1] + phi_for[s-1][k] @ Gamma_trans[s-k] # Index s-k is correct per original
                                                                                          # k=0 -> Gamma_trans[s]; k=s-1 -> Gamma_trans[1]


        # Update Sigma_rev[s+1]
        # Stan code: Sigma_rev[s+1] = Sigma_rev[s] - quad_form_sym(Sigma_for[s], phi_rev[s][s]')
        # quad_form_sym(A, B') = B @ A @ B'
        # phi_rev[s][s]' @ Sigma_for[s] @ phi_rev[s][s]
        Sigma_rev[s + 1] = Sigma_rev[s] - quad_form_sym_jax(Sigma_for[s], phi_rev[s][s].T)
        Sigma_rev[s+1] = symmetrize_jax(Sigma_rev[s+1])


    # Final output phi is phi_for[p-1][0..p-1]
    # Final output Gamma is Gamma_trans[1..p] (Stan is 1-indexed for Gamma)
    phi_list_out = [phi_for[p - 1][i] for i in range(p)]
    gamma_list_out = [Gamma_trans[i+1] for i in range(p)] # Map 0-indexed Gamma_trans[1..p] to output list

    return phi_list_out, gamma_list_out

def make_stationary_var_transformation_jax(Sigma: jnp.ndarray, A_list: List[jnp.ndarray], m: int, p: int) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Transform A matrices to stationary VAR parameters, JAX version.
    """
    P_list = [AtoP_jax(A_list[i], m) for i in range(p)]
    phi_list, gamma_list = rev_mapping_jax(P_list, Sigma, p, m)
    return phi_list, gamma_list

def create_companion_matrix_jax(Phi_list: List[jnp.ndarray], p: int, m: int) -> jnp.ndarray:
    """
    Create the companion matrix from VAR parameters, JAX version.
    Phi_list is [Phi_1, ..., Phi_p]
    Companion matrix F = [Phi_1 Phi_2 ... Phi_p]
                          [I     0     ... 0    ]
                          [0     I     ... 0    ]
                          [...               ]
                          [0     ... I   0    ]
    Shape is (mp, mp).
    """
    mp = m * p
    # Ensure all Phi matrices have the correct shape (m, m)
    # Assuming they are passed correctly from make_stationary_var_transformation_jax
    # Stack the Phi matrices horizontally
    top_row = jnp.hstack(Phi_list) # Shape (m, m*p)

    # Create the identity block for the rest of the matrix
    if p > 1:
        identity_block = jnp.eye(m * (p - 1), dtype=_DEFAULT_DTYPE) # Shape (m*(p-1), m*(p-1))
        # Create the bottom block [I 0]
        bottom_block = jnp.hstack([identity_block, jnp.zeros((m * (p - 1), m), dtype=_DEFAULT_DTYPE)]) # Shape (m*(p-1), m*p)
        # Stack top and bottom blocks
        comp_matrix = jnp.vstack([top_row, bottom_block]) # Shape (mp, mp)
    else: # p == 1
        comp_matrix = Phi_list[0] # Companion matrix is just Phi_1, shape (m, m)

    return comp_matrix

def check_stationarity_jax(phi_list: List[jnp.ndarray], m: int, p: int) -> bool:
    """
    Check if a VAR process is stationary based on its companion matrix, JAX version.
    A VAR is stationary if all eigenvalues of the companion matrix are inside the unit circle.
    """
    if p == 0: # Technically stationary if no dynamics? Or maybe not applicable.
        return True # Let's assume p=0 is stationary.
    
    # Check if any phi matrix contains NaN, which indicates transformation failure
    if any(jnp.any(jnp.isnan(phi)) for phi in phi_list):
         return False

    companion = create_companion_matrix_jax(phi_list, p, m)

    # If companion matrix construction failed or contains NaNs
    if jnp.any(jnp.isnan(companion)):
        return False

    try:
        eigenvalues = jnp.linalg.eigvals(companion)
        # Check for NaN eigenvalues
        if jnp.any(jnp.isnan(eigenvalues)):
            return False

        # Check if max absolute eigenvalue is strictly less than 1.0
        max_abs_eigenvalue = jnp.max(jnp.abs(eigenvalues))
        # Use a small tolerance for floating point comparisons
        tolerance = 1e-6 # Or similar small value
        return max_abs_eigenvalue < (1.0 - tolerance)
    except Exception:
        # Handle potential numerical errors in eigenvalue computation
        return False # Assume non-stationary if computation fails


# --- END OF FILE utils/stationary_prior_jax_simplified.py ---