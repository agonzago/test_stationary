import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.tensor import nlinalg # For symmetrize
import numpy as np
import arviz as az
from typing import List, Dict, Tuple

# Assuming stationary_prior.py is in the same directory or accessible via PYTHONPATH
from stationary_prior import (
    make_stationary_var_transformation,
    symmetrize as symmetrize_pt 
    # The create_companion_matrix from stationary_prior.py is symbolic (PyTensor)
    # We need a NumPy version for the numerical check.
)

# Ensure floatX is consistent for the test
if not hasattr(pytensor.config, 'floatX') or pytensor.config.floatX != 'float64':
    pytensor.config.floatX = 'float64'


def check_stationarity_np(phi_matrices_list_np: List[np.ndarray], m: int, p_order: int) -> bool:
    """
    Numerically checks if the VAR process defined by phi_matrices is stationary using NumPy.
    
    Args:
        phi_matrices_list_np: A list of p_order numpy arrays, each of shape (m,m),
                              representing [phi_0, phi_1, ..., phi_{p_order-1}].
        m: Dimension of the VAR.
        p_order: Order of the VAR.

    Returns:
        True if stationary, False otherwise, or None if eigenvalues can't be computed.
    """
    if p_order == 0: # VAR(0) is stationary
        return True
    if not phi_matrices_list_np and p_order > 0:
        # This case implies an issue if p_order > 0 but no phi matrices were generated
        print(f"Warning: No phi matrices provided for VAR({p_order}), m={m}. Treating as non-stationary.")
        return False

    # NumPy version of create_companion_matrix for the check
    def create_companion_matrix_np(Phi_list_np: List[np.ndarray], p_c: int, m_dim_c: int) -> np.ndarray:
        mp_c = m_dim_c * p_c
        if mp_c == 0: # Handles p_c=0 or m_dim_c=0
            return np.array([[]]) if m_dim_c > 0 else np.array([]) # Consistent empty shapes

        comp_matrix_np = np.zeros((mp_c, mp_c))
        
        if p_c > 0:
            # Assumes Phi_list_np is [phi_0, ..., phi_{p_c-1}]
            try:
                Phi_concat_np = np.concatenate(Phi_list_np, axis=1)
            except ValueError as e:
                # Handle cases where concatenation might fail due to empty list or shape mismatch
                print(f"Error concatenating Phi_list_np for companion matrix: {e}")
                print(f"Phi_list_np: {Phi_list_np}, p_c: {p_c}, m_dim_c: {m_dim_c}")
                return None # Indicate failure to create companion matrix
            comp_matrix_np[:m_dim_c, :mp_c] = Phi_concat_np
        
        if p_c > 1:
            comp_matrix_np[m_dim_c:, :-m_dim_c] = np.eye(m_dim_c * (p_c - 1))
        return comp_matrix_np

    companion_matrix = create_companion_matrix_np(phi_matrices_list_np, p_order, m)
    
    if companion_matrix is None: # Failed to create companion matrix
        return None
    if companion_matrix.size == 0 and p_order > 0 : # e.g. m=0 but p>0
         print(f"Warning: Companion matrix is empty for VAR({p_order}), m={m}, but p_order > 0.")
         return False # Not stationary if ill-defined for p > 0
    if companion_matrix.size == 0 and p_order == 0: # VAR(0)
        return True

    try:
        eigenvalues = np.linalg.eigvals(companion_matrix)
    except np.linalg.LinAlgError:
        # Failed to compute eigenvalues, likely due to numerical issues from extreme phi values
        print(f"Warning: Could not compute eigenvalues for companion matrix (m={m}, p={p_order}).")
        return None # Indicate failure
        
    max_abs_eigenvalue = np.max(np.abs(eigenvalues))
    
    is_stationary = max_abs_eigenvalue < (1.0 - 1e-9) # Check strictly less than 1
    return is_stationary


def run_stationary_prior_transformation_test(
    m_dim: int,
    p_order: int,
    n_samples: int = 500,
    seed: int = None
) -> Dict:
    """
    Tests the make_stationary_var_transformation within a PyMC model.

    Args:
        m_dim: Dimension of the VAR (number of variables).
        p_order: Order of the VAR.
        n_samples: Number of samples to draw from the prior predictive.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary containing test results.
    """
    print(f"\n--- Testing stationary_prior.py for VAR({p_order}), m={m_dim} ---")
    
    results = {
        "m": m_dim,
        "p": p_order,
        "n_samples": n_samples,
        "prior_trace": None,
        "compilation_successful": False,
        "phi_generation_count": 0,
        "stationary_samples_count": 0,
        "non_stationary_samples_count": 0,
        "check_failed_count": 0, # Count of samples where stationarity check itself failed
        "stationarity_ratio_of_checked": 0.0,
        "stationarity_ratio_of_total": 0.0,
    }

    with pm.Model() as test_model:
        # 1. Priors for Sigma (Error Covariance Matrix for the VAR process)
        #    pm.LKJCholeskyCov directly models L, the Cholesky factor of Sigma.
        #    So, chol_L_Sigma_rv *IS* L.
        chol_L_Sigma_rv = pm.LKJCholeskyCov(
            "chol_L_Sigma_rv", # Changed name to indicate it's the RV itself
            n=m_dim,
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=False # This ensures the RV is Cholesky of COVARIANCE
        )
        # chol_L_Sigma_rv is already the symbolic L (Cholesky of Sigma)

        # ---- DEBUG PRINT ----
        print(f"DEBUG: chol_L_Sigma_rv type: {type(chol_L_Sigma_rv)}")
        if hasattr(chol_L_Sigma_rv, 'ndim'):
            print(f"DEBUG: chol_L_Sigma_rv.ndim symbolic: {chol_L_Sigma_rv.ndim}")
        else:
            print(f"DEBUG: chol_L_Sigma_rv does not have ndim attribute directly.")
        # ---- END DEBUG PRINT ----

        # Now construct Sigma = L @ L.T
        Sigma_input_pt_calc = pt.dot(chol_L_Sigma_rv, chol_L_Sigma_rv.T)
        Sigma_input_pt = pm.Deterministic("Sigma_input_for_transform", symmetrize_pt(Sigma_input_pt_calc))

        # ---- DEBUG PRINT ----
        print(f"DEBUG: Sigma_input_pt type: {type(Sigma_input_pt)}")
        if hasattr(Sigma_input_pt, 'ndim'):
            print(f"DEBUG: Sigma_input_pt.ndim symbolic: {Sigma_input_pt.ndim}")
        else:
            print(f"DEBUG: Sigma_input_pt does not have ndim attribute directly (it's a pm.Deterministic). Check its .owner.inputs[0]")
            if Sigma_input_pt.owner and Sigma_input_pt.owner.inputs:
                 print(f"DEBUG: Underlying Sigma_input_pt tensor ndim: {Sigma_input_pt.owner.inputs[0].ndim}")
        # 2. Priors for A_list (raw coefficient matrices [A_0, ..., A_{p_order-1}])
        #    A_list is also an input to make_stationary_var_transformation.
        A_list_pt_input = []
        if p_order > 0:
            for i in range(p_order):
                # Using relatively tight Normal priors for A_i elements to avoid extreme inputs,
                # though the transformation should ideally be robust.
                A_i_pt = pm.Normal(f"A_{i}_input", mu=0.0, sigma=0.5, shape=(m_dim, m_dim))
                A_list_pt_input.append(A_i_pt)
        # If p_order=0, A_list_pt_input remains empty.

        # 3. Perform Transformation using functions from stationary_prior.py
        #    The output phi_list_transformed_pt represents [phi_0, ..., phi_{p_order-1}]
        phi_list_transformed_pt = [] # Default to empty list
        if p_order > 0:
            # make_stationary_var_transformation expects Sigma (error cov) and A_list (untransformed params)
            # It returns [phi_list, gamma_list]
            phi_list_from_transform, _ = make_stationary_var_transformation(
                Sigma_input_pt,
                A_list_pt_input,
                m_dim,
                p_order
            )
            phi_list_transformed_pt = phi_list_from_transform # Keep as list

            # Save each transformed phi matrix and the stacked version for easier access in trace
            for i_phi in range(p_order):
                 pm.Deterministic(f"phi_transformed_{i_phi}", phi_list_transformed_pt[i_phi])
            pm.Deterministic("phi_stacked_transformed", pt.stack(phi_list_transformed_pt, axis=0))
        else: # VAR(0) case, no phi matrices to transform
            empty_phi_shape = (0, m_dim, m_dim) if m_dim > 0 else (0,0,0)
            pm.Deterministic("phi_stacked_transformed",
                             pt.zeros(empty_phi_shape, dtype=pytensor.config.floatX))

        # 4. Sample from the prior predictive distribution
        try:
            prior_trace = pm.sample_prior_predictive(
                samples=n_samples,
                model=test_model,
                random_seed=seed
            )
            results["prior_trace"] = prior_trace
            results["compilation_successful"] = True
            print("Prior predictive sampling completed.")
        except Exception as e:
            print(f"ERROR: Prior predictive sampling failed: {e}")
            import traceback
            traceback.print_exc()
            return results # Return early

    # 5. Numerically check stationarity of the sampled phi_list
    if results["compilation_successful"] and hasattr(prior_trace, 'prior'):
        # phi_stacked_transformed has shape (chain, draw, p_order, m_dim, m_dim)
        # or (chain, draw, 0, m_dim, m_dim) if p_order=0
        phi_samples_xr = prior_trace.prior["phi_stacked_transformed"]
        
        # Stack chain and draw dimensions: results in shape (n_samples_total, p_order, m_dim, m_dim)
        phi_samples_np_all = phi_samples_xr.stack(sample=("chain", "draw")).data # .data or .values
        
        results["phi_generation_count"] = phi_samples_np_all.shape[0]
        
        if p_order > 0:
            for i_sample in range(results["phi_generation_count"]):
                # current_phi_sample_stack has shape (p_order, m_dim, m_dim)
                current_phi_sample_stack = phi_samples_np_all[i_sample, ...]
                # Convert to list of m_dim x m_dim matrices: [phi_0, phi_1, ..., phi_{p_order-1}]
                phi_list_np_for_this_draw = [current_phi_sample_stack[lag_idx, ...] for lag_idx in range(p_order)]
                
                stationarity_result = check_stationarity_np(phi_list_np_for_this_draw, m_dim, p_order)
                if stationarity_result is True:
                    results["stationary_samples_count"] += 1
                elif stationarity_result is False:
                    results["non_stationary_samples_count"] += 1
                else: # stationarity_result is None (check failed)
                    results["check_failed_count"] += 1
        else: # p_order == 0 (VAR(0) model is always stationary)
            results["stationary_samples_count"] = results["phi_generation_count"]

    # Calculate ratios
    total_checked_successfully = results["stationary_samples_count"] + results["non_stationary_samples_count"]
    if total_checked_successfully > 0:
        results["stationarity_ratio_of_checked"] = results["stationary_samples_count"] / total_checked_successfully
    if results["phi_generation_count"] > 0:
         results["stationarity_ratio_of_total"] = results["stationary_samples_count"] / results["phi_generation_count"]

    print(f"Total phi sets generated: {results['phi_generation_count']}")
    print(f"  Stationary: {results['stationary_samples_count']}")
    print(f"  Non-stationary: {results['non_stationary_samples_count']}")
    print(f"  Stationarity check failed: {results['check_failed_count']}")
    print(f"Stationarity Ratio (of successfully checked): {results['stationarity_ratio_of_checked']:.4f}")
    print(f"Stationarity Ratio (of total generated): {results['stationarity_ratio_of_total']:.4f}")
    print(f"--- Test for VAR({p_order}), m={m_dim} finished ---")

    return results

if __name__ == '__main__':
    # Test Suite
    test_cases = [
        {"m": 1, "p": 1, "n_samples": 100, "seed": 10}, # Univariate VAR(1)
        {"m": 2, "p": 1, "n_samples": 200, "seed": 123},
        {"m": 2, "p": 2, "n_samples": 200, "seed": 456},
        {"m": 2, "p": 0, "n_samples": 50, "seed": 789},  # VAR(0)
        {"m": 3, "p": 3, "n_samples": 100, "seed": 101}, # Larger model
        # Potentially add a case with m=1, p=0
        {"m": 1, "p": 0, "n_samples": 20, "seed": 20}
    ]

    all_tests_passed = True
    detailed_results = []

    for i, case_params in enumerate(test_cases):
        print(f"\nRunning Test Case {i+1}: m={case_params['m']}, p={case_params['p']}")
        res = run_stationary_prior_transformation_test(
            m_dim=case_params["m"],
            p_order=case_params["p"],
            n_samples=case_params["n_samples"],
            seed=case_params["seed"]
        )
        detailed_results.append(res)

        if not res["compilation_successful"]:
            print(f"FAIL: Test Case {i+1} (m={case_params['m']}, p={case_params['p']}) failed compilation/sampling.")
            all_tests_passed = False
            continue

        if res["phi_generation_count"] == 0 and case_params["n_samples"] > 0 :
            print(f"FAIL: Test Case {i+1} (m={case_params['m']}, p={case_params['p']}) generated no phi samples.")
            all_tests_passed = False
            continue
        
        # For p > 0, expect a high stationarity ratio. For p = 0, expect 1.0.
        expected_min_ratio = 1.0 if case_params["p"] == 0 else 0.95 # Allow some leeway for numerical edge cases

        # Check ratio of successfully checked samples. If many checks fail, that's also a problem.
        if res["check_failed_count"] > 0.1 * res["phi_generation_count"]: # If >10% checks failed
            print(f"WARNING: Test Case {i+1} (m={case_params['m']}, p={case_params['p']}) had many stationarity check failures: {res['check_failed_count']}.")
            # This might not be a strict fail of the transformation itself, but of the test stability.

        if res["stationarity_ratio_of_checked"] < expected_min_ratio :
             if total_checked_successfully > 0 : # Avoid division by zero if all checks failed
                print(f"FAIL: Test Case {i+1} (m={case_params['m']}, p={case_params['p']}) stationarity ratio too low: {res['stationarity_ratio_of_checked']:.4f} (expected >= {expected_min_ratio}).")
                all_tests_passed = False
        else:
             if total_checked_successfully > 0 : # Only print pass if samples were checked
                print(f"PASS: Test Case {i+1} (m={case_params['m']}, p={case_params['p']}) stationarity ratio: {res['stationarity_ratio_of_checked']:.4f}")


    print("\n--- Test Suite Summary ---")
    if all_tests_passed:
        print("All standalone tests for stationary_prior.py passed successfully!")
    else:
        print("Some standalone tests for stationary_prior.py FAILED.")

    # Optional: Print detailed results
    # import pandas as pd
    # print("\nDetailed Results:")
    # df_results = pd.DataFrame(detailed_results)
    # print(df_results)