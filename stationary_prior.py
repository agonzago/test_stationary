# stationary_prior.py - Complete re-implementation based on Stan code

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.tensor import nlinalg, slinalg
from typing import List, Tuple
import pymc as pm 
# Ensure floatX is consistent
if not hasattr(pytensor.config, 'floatX') or pytensor.config.floatX != 'float64':
    pytensor.config.floatX = 'float64'

def symmetrize(A):
    """Symmetrize a matrix by computing (A + A.T)/2."""
    return 0.5 * (A + A.T)

def sqrtm(A):
    """
    Matrix square root computation using eigendecomposition.
    EXACT equivalent to Stan's sqrtm function.
    """
    A_sym = symmetrize(A)  # Ensure matrix is symmetric
    
    # Stan: eigenvalues_sym(A)
    evals, evecs = nlinalg.eigh(A_sym)
    
    # Stan: root_root_evals = sqrt(sqrt(eigenvalues_sym(A)))
    # This computes the 4th root of eigenvalues
    root_root_evals = pt.sqrt(pt.sqrt(pt.maximum(evals, 1e-12)))
    
    # Stan: eprod = diag_post_multiply(evecs, root_root_evals)
    # This is equivalent to evecs @ diag(root_root_evals)
    eprod = evecs @ pt.diag(root_root_evals)
    
    # Stan: return tcrossprod(eprod)
    # This is equivalent to eprod @ eprod.T
    return tcrossprod(eprod)

def tcrossprod(A):
    """
    Compute A @ A.T (equivalent to Stan's tcrossprod).
    """
    return A @ A.T

def quad_form_sym(A, B):
    """
    Compute B.T @ A @ B (symmetric quadratic form).
    Ensures A is treated as symmetric.
    """
    A_sym = symmetrize(A)
    return B.T @ A_sym @ B

def AtoP(A, m):
    """
    Transform A to P (equivalent to Stan's AtoP function).
    """
    B = tcrossprod(A)  # B = A @ A.T
    
    # Add 1.0 to diagonal elements of B
    B_diag = pt.diagonal(B)
    B = pt.set_subtensor(B[pt.arange(m), pt.arange(m)], B_diag + 1.0)
    
    # Compute matrix square root of B
    sqrtB = sqrtm(B)
    
    # Compute sqrtB^(-1) @ A
    P = slinalg.solve(sqrtB, A, assume_a='pos')
    
    return P

def rev_mapping(P, Sigma, p, m):
    """
    Equivalent to Stan's rev_mapping function.
    
    Args:
        P: List of p matrices, each m×m
        Sigma: m×m covariance matrix
        p: Order of the VAR process
        m: Dimension of the VAR process
        
    Returns:
        Tuple of (phi, Gamma) lists
    """
    # Initialize arrays for Step 1
    Sigma_for = [pt.zeros((m, m)) for _ in range(p+1)]
    S_for_list = [pt.zeros((m, m)) for _ in range(p+1)]
    
    # Initialize arrays for Step 2
    Sigma_rev = [pt.zeros((m, m)) for _ in range(p+1)]
    Gamma_trans = [pt.zeros((m, m)) for _ in range(p+1)]
    
    # Initialize phi matrices
    phi_for = [[pt.zeros((m, m)) for _ in range(p)] for _ in range(p)]
    phi_rev = [[pt.zeros((m, m)) for _ in range(p)] for _ in range(p)]
    
    # Step 1
    Sigma_for[p] = Sigma
    S_for_list[p] = sqrtm(Sigma)
    
    for s in range(1, p+1):
        # In Stan: P[p-s+1]
        P_idx = p - s
        P_current = P[P_idx]
        
        # Compute S_for
        S_for = -tcrossprod(P_current)
        
        # Add 1.0 to diagonal
        S_for_diag = pt.diagonal(S_for)
        S_for = pt.set_subtensor(S_for[pt.arange(m), pt.arange(m)], S_for_diag + 1.0)
        
        # Compute S_rev
        S_rev = sqrtm(S_for)
        
        # Compute quadratic form
        Q = quad_form_sym(Sigma_for[p-s+1], S_rev)
        
        # Compute sqrtm(Q)
        sqrtQ = sqrtm(Q)
        
        # Compute MID
        MID = slinalg.solve(S_rev, sqrtQ, assume_a='pos')
        
        # Compute S_for_list[p-s+1] = MID @ S_rev^(-1)
        S_for_list[p-s] = slinalg.solve(S_rev.T, MID.T, assume_a='pos').T
        
        # Compute Sigma_for[p-s+1]
        Sigma_for[p-s] = tcrossprod(S_for_list[p-s])
    
    # Step 2
    Sigma_rev[0] = Sigma_for[0]
    Gamma_trans[0] = Sigma_for[0]
    
    for s in range(p):
        S_for = S_for_list[s]
        S_rev = sqrtm(Sigma_rev[s])
        
        # Compute phi_for[s,s] and phi_rev[s,s]
        temp_phi_for = S_for @ P[s]
        phi_for[s][s] = slinalg.solve(S_rev.T, temp_phi_for.T, assume_a='pos').T
        
        temp_phi_rev = S_rev @ P[s].T
        phi_rev[s][s] = slinalg.solve(S_for.T, temp_phi_rev.T, assume_a='pos').T
        
        # Compute Gamma_trans[s+1]
        Gamma_trans[s+1] = phi_for[s][s] @ Sigma_rev[s]
        
        if s >= 1:
            for k in range(s):
                phi_for[s][k] = phi_for[s-1][k] - phi_for[s][s] @ phi_rev[s-1][s-1-k]
                phi_rev[s][k] = phi_rev[s-1][k] - phi_rev[s][s] @ phi_for[s-1][s-1-k]
            
            for k in range(s):
                Gamma_trans[s+1] = Gamma_trans[s+1] + phi_for[s-1][k] @ Gamma_trans[s+1-k-1]
        
        if s < p-1:
            Sigma_rev[s+1] = Sigma_rev[s] - quad_form_sym(Sigma_for[s], phi_rev[s][s].T)
    
    # Prepare output
    phi_list = [phi_for[p-1][i] for i in range(p)]
    gamma_list = [Gamma_trans[i] for i in range(p)]
    
    return phi_list, gamma_list

def make_stationary_var_transformation(Sigma, A_list, m, p):
    """
    Transform A matrices to stationary VAR parameters.
    
    Args:
        Sigma: m×m error covariance matrix
        A_list: List of p matrices [A_0, ..., A_{p-1}], each m×m
        m: Dimension of the VAR
        p: Order of the VAR
        
    Returns:
        Tuple of (phi_list, gamma_list):
            - phi_list: List of p VAR coefficient matrices [phi_0, ..., phi_{p-1}]
            - gamma_list: List of p autocovariance matrices [Gamma_0, ..., Gamma_{p-1}]
    """
    # Convert A matrices to P matrices
    P_list = [AtoP(A_list[i], m) for i in range(p)]
    
    # Apply reverse mapping to get phi and Gamma matrices
    phi_list, gamma_list = rev_mapping(P_list, Sigma, p, m)
    
    return phi_list, gamma_list

def create_companion_matrix(Phi_list, p, m):
    """
    Create the companion matrix from VAR parameters.
    
    Args:
        Phi_list: List of p VAR coefficient matrices [phi_0, ..., phi_{p-1}]
        p: Order of the VAR
        m: Dimension of the VAR
        
    Returns:
        mp×mp companion matrix
    """
    mp = m * p
    comp_matrix = pt.zeros((mp, mp))
    
    # Add coefficient matrices to first block row
    for i in range(p):
        comp_matrix = pt.set_subtensor(comp_matrix[:m, i*m:(i+1)*m], Phi_list[i])
    
    # Add identity matrices below diagonal if p > 1
    if p > 1:
        comp_matrix = pt.set_subtensor(comp_matrix[m:, :-m], pt.eye(m * (p - 1)))
    
    return comp_matrix

# Test function
# def check_stationarity(phi_list, m, p):
#     """
#     Check if a VAR process is stationary based on its companion matrix.
    
#     Args:
#         phi_list: List of p VAR coefficient matrices [phi_0, ..., phi_{p-1}]
#         m: Dimension of the VAR
#         p: Order of the VAR
        
#     Returns:
#         True if the process is stationary, False otherwise
#     """
#     # Create companion matrix
#     companion = create_companion_matrix(phi_list, p, m)
    
#     # Compute eigenvalues
#     eigenvalues = pt.linalg.eigvals(companion)
    
#     # Check if all eigenvalues are inside the unit circle
#     max_abs_eigenvalue = pt.max(pt.abs_(eigenvalues))
    
#     return max_abs_eigenvalue < 1.0

def check_stationarity(phi_list, m, p):
    """
    Check if a VAR process is stationary based on its companion matrix.

    Args:
        phi_list: List of p VAR coefficient matrices [phi_0, ..., phi_{p-1}]
        m: Dimension of the VAR
        p: Order of the VAR

    Returns:
        True if the process is stationary, False otherwise
    """
    # Create companion matrix
    companion = create_companion_matrix(phi_list, p, m)

    # Compute eigenvalues using the correct path: pt.linalg.eigvals
    eigenvalues = pt.linalg.eig(companion) # Corrected line

    # Check if all eigenvalues are inside the unit circle
    max_abs_eigenvalue = pm.math.maximum(pm.math.abs(eigenvalues))

    # Compile the PyTensor graph to get a numerical result
    get_max_abs_eigenvalue = pytensor.function([], max_abs_eigenvalue)
    
    return get_max_abs_eigenvalue() < 1.0