# Test script for the clean stationary prior implementation

import numpy as np
import pytensor
import pytensor.tensor as pt
import arviz as az
from typing import List, Dict
import matplotlib.pyplot as plt

import os
# # Force NumPy to use OpenBLAS instead of MKL
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['MKL_DYNAMIC'] = 'FALSE'
# # Completely disable MKL
# os.environ['NPY_MKL_FORCE_INTEL'] = '0'
# os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
#os.environ['omp_get_num_procs'] = '1'
# ##Try to use MKL if available
# try:
#     pytensor.config.blas__ldflags = '-Wl,--start-group ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_sequential.a ${MKLROOT}/lib/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl'
# #     #pytensor.config.blas__ldflags="-L\"C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\lib\" -lmkl_core -lmkl_intel_lp64 -lmkl_sequential"
# #     #pytensor.config.blas__ldflags="-L\"C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest\\lib\" -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -lopenmp"
    
# except Exception:
#     print("Could not set MKL as BLAS backend for PyTensor")

import jax
import jax.numpy as jnp
import pymc as pm
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)
#pm.config.update("compute_backend", "jax")


# # # --- JAX/Numpyro Setup ---
# # # ... (identical to your last working version)
# print("Attempting to force JAX to use CPU...")
# try:
#     jax.config.update("jax_platforms", "cpu")
#     print(f"JAX targeting CPU.")
# except Exception as e_cpu:
#     print(f"Warning: Could not force CPU platform: {e_cpu}")
# print(f"JAX default platform: {jax.default_backend()}")
# jax.config.update("jax_enable_x64", True)
# _DEFAULT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
# print(f"Using JAX with dtype: {_DEFAULT_DTYPE}")

# Import the clean implementation
from stationary_prior import (
    make_stationary_var_transformation,
    check_stationarity
)

# def test_stationary_prior(m=3, p=1, n_samples=10, seed=123):
#     """Run with minimal parameters for testing"""
#     print(f"--- [DEBUG test_stationary_prior function start] ---")
#     print(f"Parameter m = {m}")
#     print(f"Parameter p = {p}")

#     with pm.Model() as model:
#         # 1. Prior for Sigma (error covariance)
#         L_Sigma, _, _ = pm.LKJCholeskyCov(
#             "packed_chol_cov",
#             n=m,
#             eta=2.0,
#             sd_dist=pm.Exponential.dist(1.0),
#             compute_corr=True
#         )
    
#         # Construct Sigma = L @ L.T
#         Sigma = pt.dot(L_Sigma, L_Sigma.T)
        
#         print(f"--- [DEBUG test_stationary_prior, after Sigma definition] ---")
#         print(f"Symbolic Sigma.ndim = {Sigma.ndim}")
#         print(f"Symbolic Sigma.type = {Sigma.type}")
    
#         # 2. Prior for A matrices
#         A_list = []
#         for i in range(p):
#             A_i = pm.Normal(f"A_{i}", mu=0.0, sigma=0.5, shape=(m, m))
#             A_list.append(A_i)
        
#         # 3. Transform A matrices to stationary parameters
#         phi_list, gamma_list = make_stationary_var_transformation(
#             Sigma, A_list, m, p
#         )
        
#         # Save transformed parameters
#         for i in range(p):
#             pm.Deterministic(f"phi_{i}", phi_list[i])
        
#         # Sample from prior
#         print(f"Sampling {n_samples} draws from prior...")
        
#         # Use only a few samples for testing
#         trace = pm.sample_prior_predictive(samples=n_samples, random_seed=seed)
        
#         print("Prior sampling completed successfully!")
        
#     return True

def test_stationary_prior(m=3, p=2, n_samples=100, seed=123):
    print(f"--- [DEBUG test_stationary_prior function start] ---")
    print(f"Parameter m = {m}")
    print(f"Parameter p = {p}")

    with pm.Model() as model:
        # Note that we access the distribution for the standard
        # deviations, and do not create a new random variable.
        sd_dist = pm.Exponential.dist(1.0, size=m)
        chol, corr, sigmas = pm.LKJCholeskyCov(
            'chol_cov', eta=1.1, n=m, sd_dist=sd_dist
        )

       
        # Or compute the covariance matrix
        Sigma = pt.dot(chol, chol.T)
        pm.Deterministic('covariance', Sigma)
       
        # 2. Prior for A matrices
        A_list = []
        for i in range(p):
            A_i = pm.Normal(f"A_{i}", mu=0.0, sigma=0.5, shape=(m, m))
            A_list.append(A_i)
        
        # 3. Transform A matrices to stationary parameters
        phi_list, gamma_list = make_stationary_var_transformation(
            Sigma, A_list, m, p
        )
        
        # # Save transformed parameters
        for i in range(p):
            pm.Deterministic(f"phi_{i}", phi_list[i])
        
        # Sample from prior
        print(f"Sampling {n_samples} draws from prior...")
        trace = pm.sample_prior_predictive(
                samples=n_samples, 
                random_seed=seed)                                       
                
    
    # Check stationarity of samples
    results = {
        "m": m,
        "p": p,
        "samples_checked": 0,
        "A_stationary_count": 0,
        "phi_stationary_count": 0
    }
    
    print("Checking stationarity of samples...")
    for i in range(min(n_samples, 100)):  # Check up to 100 samples
        try:
            # Extract A matrices
            A_matrices = []
            for j in range(p):
                A_j = trace.prior[f"A_{j}"][0, i].values
                A_matrices.append(A_j)
            
            # Extract phi matrices
            phi_matrices = []
            for j in range(p):
                phi_j = trace.prior[f"phi_{j}"][0, i].values
                phi_matrices.append(phi_j)
            
            # Convert to numpy for easier checking
            A_np = [np.array(A) for A in A_matrices]
            phi_np = [np.array(phi) for phi in phi_matrices]
            
            # Check stationarity
            A_stationary = np_check_stationarity(A_np, m, p)
            phi_stationary = np_check_stationarity(phi_np, m, p)
            
            # Update results
            results["samples_checked"] += 1
            results["A_stationary_count"] += int(A_stationary)
            results["phi_stationary_count"] += int(phi_stationary)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Processed {i+1} samples")
                
        except Exception as e:
            print(f"Error checking sample {i}: {str(e)}")
    
    # Calculate percentages
    if results["samples_checked"] > 0:
        results["A_stationary_percent"] = results["A_stationary_count"] / results["samples_checked"] * 100
        results["phi_stationary_percent"] = results["phi_stationary_count"] / results["samples_checked"] * 100
    else:
        results["A_stationary_percent"] = 0
        results["phi_stationary_percent"] = 0
    
    return results

def np_check_stationarity(phi_matrices, m, p):
    """
    NumPy implementation of stationarity check
    """
    if not phi_matrices:
        return True  # Empty list (VAR(0)) is stationary
    
    # Create companion matrix
    companion = np.zeros((m*p, m*p))
    
    # Add coefficient matrices to first block row
    for i in range(p):
        companion[:m, i*m:(i+1)*m] = phi_matrices[i]
    
    # Add identity matrices on subdiagonal blocks
    if p > 1:
        for i in range(1, p):
            companion[m*i:m*(i+1), m*(i-1):m*i] = np.eye(m)
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(companion)
    
    # Process is stationary if all eigenvalues are inside unit circle
    max_abs_eigenvalue = np.max(np.abs(eigenvalues))
    
    return max_abs_eigenvalue < 1.0


# res = test_stationary_prior(
#     m=2,
#     p=2,
#     n_samples=50
# )
    
if __name__ == "__main__":
    print("Testing Clean Stationary Prior Implementation")
    print("============================================")
    
    # Test with different m and p values
    test_cases = [
      #  {"m": 1, "p": 1, "label": "VAR(1) with m=1"},
        {"m": 3, "p": 1, "label": "VAR(1) with m=3"},
        {"m": 3, "p": 2, "label": "VAR(2) with m=3"}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{case['label']}")
        res = test_stationary_prior(
            m=case["m"],
            p=case["p"],
            n_samples=50
        )
        res["label"] = case["label"]
        results.append(res)
        
        print(f"\nResults for {case['label']}:")
        print(f"Samples checked: {res['samples_checked']}")
        print(f"Original matrices stationary: {res['A_stationary_count']} ({res['A_stationary_percent']:.1f}%)")
        print(f"Transformed matrices stationary: {res['phi_stationary_count']} ({res['phi_stationary_percent']:.1f}%)")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    labels = [res["label"] for res in results]
    a_pcts = [res["A_stationary_percent"] for res in results]
    phi_pcts = [res["phi_stationary_percent"] for res in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, a_pcts, width, label='Original A')
    bars2 = ax.bar(x + width/2, phi_pcts, width, label='Transformed Phi')
    
    ax.set_ylabel('Stationarity Rate (%)')
    ax.set_title('Stationarity Comparison: Original vs Transformed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('stationarity_comparison.png')
    plt.close()
    
    print("\nResults visualization saved to 'stationarity_comparison.png'")