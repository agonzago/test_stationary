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
    P = jsl.solve(sqrtB, A, assume_a='pos')
    
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
    Sigma_for = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]
    S_for_list = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]
    
    # Initialize arrays for Step 2
    Sigma_rev = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]
    Gamma_trans = [jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p + 1)]
    
    # Initialize phi matrices
    # These lists will store JAX arrays
    phi_for = [[jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p)] for _ in range(p)]
    phi_rev = [[jnp.zeros((m, m), dtype=_DEFAULT_DTYPE) for _ in range(p)] for _ in range(p)]

    # Step 1
    Sigma_for_p = Sigma
    S_for_list_p = sqrtm_jax(Sigma)

    # Store initial values in the lists using .at[index].set(value) if they were JAX arrays,
    # but since they are lists of JAX arrays, direct assignment is fine.
    Sigma_for[p] = Sigma_for_p
    S_for_list[p] = S_for_list_p
    
    for s in range(1, p + 1):
        P_idx = p - s
        P_current = P[P_idx]
        
        S_for_val = -tcrossprod_jax(P_current)
        S_for_val = S_for_val.at[jnp.arange(m), jnp.arange(m)].add(1.0)
        
        S_rev_val = sqrtm_jax(S_for_val)
        
        Q_val = quad_form_sym_jax(Sigma_for[p - s + 1], S_rev_val)
        sqrtQ_val = sqrtm_jax(Q_val)
        
        MID_val = jsl.solve(S_rev_val, sqrtQ_val, assume_a='pos')
        
        S_for_list[p - s] = jsl.solve(S_rev_val.T, MID_val.T, assume_a='pos').T
        Sigma_for[p - s] = tcrossprod_jax(S_for_list[p - s])

    # Step 2
    Sigma_rev[0] = Sigma_for[0]
    Gamma_trans[0] = Sigma_for[0]
    
    for s in range(p):
        S_for_s = S_for_list[s]
        S_rev_s = sqrtm_jax(Sigma_rev[s])
        
        temp_phi_for = S_for_s @ P[s]
        phi_for_s_s = jsl.solve(S_rev_s.T, temp_phi_for.T, assume_a='pos').T
        
        temp_phi_rev = S_rev_s @ P[s].T
        phi_rev_s_s = jsl.solve(S_for_s.T, temp_phi_rev.T, assume_a='pos').T

        # Update lists of JAX arrays
        phi_for_s_list = phi_for[s].copy() # Get current list for row s
        phi_for_s_list[s] = phi_for_s_s
        phi_for[s] = phi_for_s_list

        phi_rev_s_list = phi_rev[s].copy() # Get current list for row s
        phi_rev_s_list[s] = phi_rev_s_s
        phi_rev[s] = phi_rev_s_list
        
        Gamma_trans[s + 1] = phi_for[s][s] @ Sigma_rev[s]
        
        if s >= 1:
            for k in range(s):
                phi_for_s_k_val = phi_for[s-1][k] - phi_for[s][s] @ phi_rev[s-1][s-1-k]
                phi_rev_s_k_val = phi_rev[s-1][k] - phi_rev[s][s] @ phi_for[s-1][s-1-k]

                phi_for_s_list = phi_for[s].copy()
                phi_for_s_list[k] = phi_for_s_k_val
                phi_for[s] = phi_for_s_list

                phi_rev_s_list = phi_rev[s].copy()
                phi_rev_s_list[k] = phi_rev_s_k_val
                phi_rev[s] = phi_rev_s_list
            
            for k in range(s): # Original Stan code has 'k' from 0 to s-1
                # Gamma_trans[s+1] = Gamma_trans[s+1] + phi_for[s-1][k] @ Gamma_trans[s+1-k-1]
                # Corrected indexing based on common VAR literature: Gamma_trans[s+1] += phi_for[s][k] @ Gamma_trans[s-k]
                # The original PyTensor code uses: phi_for[s-1][k] @ Gamma_trans[s+1-k-1]
                # Let's stick to the PyTensor code's direct translation for now.
                Gamma_trans[s + 1] = Gamma_trans[s + 1] + phi_for[s-1][k] @ Gamma_trans[s-k] # Re-check original Stan: s+1-k-1 for Gamma_trans index.
                                                                                          # If Gamma_trans is indexed 0..p, then s+1-k-1 => s-k
                                                                                          # The loop is for k in 0..s-1.
                                                                                          # When k = 0, index is s. When k = s-1, index is 1.

        #if s < p - 1:
        Sigma_rev[s + 1] = Sigma_rev[s] - quad_form_sym_jax(Sigma_for[s], phi_rev[s][s].T)
            
    phi_list_out = [phi_for[p - 1][i] for i in range(p)]
    gamma_list_out = [Gamma_trans[i+1] for i in range(p)] # Stan's Gamma is 1-indexed, so Gamma_trans[1]...Gamma_trans[p]

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
    """
    mp = m * p
    comp_matrix = jnp.zeros((mp, mp), dtype=_DEFAULT_DTYPE)
    
    for i in range(p):
        comp_matrix = comp_matrix.at[:m, i*m:(i+1)*m].set(Phi_list[i])
    
    if p > 1:
        comp_matrix = comp_matrix.at[m:, :-m].set(jnp.eye(m * (p - 1), dtype=_DEFAULT_DTYPE))
        
    return comp_matrix

def check_stationarity_jax(phi_list: List[jnp.ndarray], m: int, p: int) -> bool:
    """
    Check if a VAR process is stationary based on its companion matrix, JAX version.
    """
    companion = create_companion_matrix_jax(phi_list, p, m)
    eigenvalues = jnp.linalg.eigvals(companion)
    max_abs_eigenvalue = jnp.max(jnp.abs(eigenvalues))
    return max_abs_eigenvalue < 1.0
