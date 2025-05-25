# stationary_prior_jax_test.py - Test for the JAX implementation with Python loops

import jax
import jax.numpy as jnp
import jax.random as jrandom
# No lax.scan needed in test function
# from jax import lax
from jax.typing import ArrayLike
from typing import List, Tuple, Dict # <-- Added Dict import
import time
import matplotlib.pyplot as plt
import numpy as np # For plotting

# Import the JAX stationary prior functions (MAKE SURE THIS IS THE PYTHON LOOP VERSION)
from stationary_prior_jax_simplified import (
    make_stationary_var_transformation_jax,
    check_stationarity_jax,
    create_companion_matrix_jax,
    _DEFAULT_DTYPE
)

# Ensure JAX settings match the library
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu") # Match the library's potential CPU preference
print(f"JAX backend: {jax.default_backend()}, float64 enabled: {jax.config.jax_enable_x64}")


# --- Helper function to generate random SPD matrix ---
def random_spd_matrix(key, size, scale=1.0):
    """Generates a random symmetric positive definite matrix using JAX."""
    A = jrandom.normal(key, (size, size), dtype=_DEFAULT_DTYPE)
    ATA = A @ A.T
    ATA += scale * jnp.eye(size, dtype=_DEFAULT_DTYPE) # Add scale identity for robustness
    return ATA

# --- Test function (Runs in eager mode with Python loops) ---
# Removed JIT and VMAP - runs N samples sequentially in Python
def run_stationary_prior_jax_test(
    m_dim: int,
    p_order: int,
    n_samples: int = 1000,
    key_seed: int = 42
) -> Dict:
    """
    Tests the JAX make_stationary_var_transformation_jax (Python loops version)
    by generating random input matrices and checking the stationarity of the output.
    Assumes p_order >= 1.
    """
    if p_order < 1:
        raise ValueError("This test is for stationary priors with p >= 1.")
    if m_dim < 1:
         raise ValueError("Dimension m must be >= 1.")


    print(f"--- Running JAX Stationary Prior Test (Python Loops) for VAR({p_order}), m={m_dim}, {n_samples} samples ---")
    key = jrandom.PRNGKey(key_seed)

    results = {
        "m": m_dim,
        "p": p_order,
        "n_samples": n_samples,
        "A_stationary_count": 0, # Count of inputs that were stationary (should be low)
        "phi_stationary_count": 0, # Count of outputs that were stationary
        "transform_failed_count": 0, # Count of samples where transformation resulted in NaN/Inf
        "stationarity_check_failed_count": 0 # Count where the check itself failed on finite phi
    }

    start_time = time.time()

    # Loop for N samples (Python loop)
    for i in range(n_samples):
        # Generate inputs for one sample
        key, key_sample = jrandom.split(key) # Split key for each sample
        key_sigma, key_A = jrandom.split(key_sample)

        sigma_input = random_spd_matrix(key_sigma, m_dim, scale=jnp.sqrt(m_dim))

        A_list_input = []
        key_A_matrices = jrandom.split(key_A, p_order)
        A_list_input = [jrandom.normal(key_A_matrices[i], (m_dim, m_dim), dtype=_DEFAULT_DTYPE) for i in range(p_order)]


        # --- Check stationarity of the input A matrices (optional) ---
        is_A_stationary = jnp.array(False)
        try:
            is_A_stationary = check_stationarity_jax(A_list_input, m_dim, p_order)
        except Exception:
            is_A_stationary = jnp.array(False)
        if is_A_stationary:
            results["A_stationary_count"] += 1

        # --- Perform the transformation ---
        phi_list_transformed = []
        gamma_list_transformed = []
        transform_failed_current_sample = False

        try:
            # Call the transformation function
            phi_list_transformed, gamma_list_transformed = make_stationary_var_transformation_jax(
                sigma_input,
                A_list_input,
                m_dim,
                p_order
            )

            # Explicitly check if the output lists contain any NaNs/Infs
            if not phi_list_transformed or not gamma_list_transformed :
                transform_failed_current_sample = True
            else:
               if len(phi_list_transformed) != p_order or any(p.shape != (m_dim, m_dim) for p in phi_list_transformed) or \
                  len(gamma_list_transformed) != p_order or any(g.shape != (m_dim, m_dim) for g in gamma_list_transformed):
                   transform_failed_current_sample = True
               else:
                   phi_stacked = jnp.stack(phi_list_transformed, axis=0)
                   gamma_stacked = jnp.stack(gamma_list_transformed, axis=0)
                   if jnp.any(~jnp.isfinite(phi_stacked)) or jnp.any(~jnp.isfinite(gamma_stacked)):
                       transform_failed_current_sample = True

        except Exception:
            transform_failed_current_sample = True
            # No need to create NaNs here, will be counted as failed anyway


        if transform_failed_current_sample:
             results["transform_failed_count"] += 1
        else:
             # --- Check stationarity if transform was successful ---
             is_phi_stationary = jnp.array(False, dtype=jnp.bool_)
             check_failed_current_sample = False

             try:
                 is_phi_stationary = check_stationarity_jax(phi_list_transformed, m_dim, p_order)
                 if not is_phi_stationary: # Count non-stationary among successful transforms
                      pass # This is expected sometimes for random inputs to the Stan algorithm
                 # If check_stationarity_jax returns True or False without error, check is not failed
                 check_failed_current_sample = False
             except Exception:
                  # An unexpected error during the check itself
                  check_failed_current_sample = True
                  is_phi_stationary = jnp.array(False, dtype=jnp.bool_) # Mark as non-stationary if check failed

             if check_failed_current_sample:
                 results["stationarity_check_failed_count"] += 1
             elif is_phi_stationary:
                 results["phi_stationary_count"] += 1 # Only count stationary if check did not fail


        # Optional: Print progress
        if (i + 1) % (n_samples // 10) == 0 or (i + 1) == n_samples:
             print(f"  Processed {i + 1}/{n_samples} samples...")


    end_time = time.time()

    results["test_duration"] = end_time - start_time

    # Calculate ratios
    results["A_stationarity_ratio"] = results["A_stationary_count"] / n_samples if n_samples > 0 else 0.0

    num_successful_transforms = n_samples - results["transform_failed_count"]
    num_successfully_checked = num_successful_transforms - results["stationarity_check_failed_count"]


    results["phi_stationarity_ratio_of_successful_checked"] = results["phi_stationary_count"] / num_successfully_checked if num_successfully_checked > 0 else 0.0
    results["phi_stationarity_ratio"] = results["phi_stationary_count"] / n_samples if n_samples > 0 else 0.0 # Overall ratio


    print(f"\nTest Summary for VAR({p_order}), m={m_dim}:")
    print(f"  Generated and processed {n_samples} samples.")
    print(f"  Input A matrices stationary: {results['A_stationary_count']}/{n_samples} ({results['A_stationarity_ratio']:.2%})")
    transform_fail_ratio = (results['transform_failed_count'] / n_samples) if n_samples > 0 else 0.0
    print(f"  Transformation failed (NaN/Inf output): {results['transform_failed_count']}/{n_samples} ({transform_fail_ratio:.1%})")
    check_fail_ratio_of_attempted = (results['stationarity_check_failed_count'] / num_successful_transforms) if num_successful_transforms > 0 else 0.0
    print(f"  Stationarity check failed (on successful transforms): {results['stationarity_check_failed_count']}/{num_successful_transforms if num_successful_transforms > 0 else 0} ({check_fail_ratio_of_attempted:.1%})")
    phi_ratio_successful_checked = (results['phi_stationary_count'] / num_successfully_checked) if num_successfully_checked > 0 else 0.0
    print(f"  Transformed Phi matrices stationary (of successfully checked): {results['phi_stationary_count']}/{num_successfully_checked if num_successfully_checked > 0 else 0} ({phi_ratio_successful_checked:.4f})")
    print(f"Test duration: {results['test_duration']:.2f} seconds")

    # Assertions (adjusted for Python loop version expectations)
    test_passed = True
    fail_messages = []

    # Condition 1: Transformation failure should be low, but might not be zero for random inputs
    # Stan's algorithm isn't guaranteed to never produce NaNs for *any* random input
    if results["transform_failed_count"] > 0.01 * n_samples: # Allow up to 1% transform failures
        fail_messages.append(f"Transformation produced high rate of NaN/Inf output: {results['transform_failed_count']}/{n_samples} ({results['transform_failed_count'] / n_samples:.1%}). Expected <= 1%.")
        test_passed = False
    elif results["transform_failed_count"] > 0:
         print(f"Info: Transformation failed for {results['transform_failed_count']} samples ({results['transform_failed_count'] / n_samples:.1%}), within acceptable tolerance.")

    # Condition 2: Stationarity check failure (relative to successful transforms) should be very low
    if num_successful_transforms > 0 and results["stationarity_check_failed_count"] > 0:
         check_fail_ratio_of_successful = results["stationarity_check_failed_count"] / num_successful_transforms
         if check_fail_ratio_of_successful > 0.001: # Allow up to 0.1% check failures on successful transforms
             fail_messages.append(f"Stationarity check failed for high rate of successful transforms: {results['stationarity_check_failed_count']}/{num_successful_transforms} ({check_fail_ratio_of_successful:.1%}). Expected <= 0.1%.")
             test_passed = False
         elif results["stationarity_check_failed_count"] > 0:
             print(f"Info: {results['stationarity_check_failed_count']} samples failed the stationarity check itself ({check_fail_ratio_of_successful:.1%}) among successful transforms, within tolerance.")


    # Condition 3: Stationarity ratio of successfully checked transforms should be very high (near 1.0)
    if num_successfully_checked > 0:
        actual_phi_ratio = results["phi_stationarity_ratio_of_successful_checked"]
        if actual_phi_ratio < 0.99: # Allow up to 1% non-stationary after successful check
            fail_messages.append(f"Stationarity ratio of successfully checked transforms ({results['phi_stationary_count']}/{num_successfully_checked}) is lower than expected (expected >= 0.99, got {actual_phi_ratio:.4f}).")
            test_passed = False
        elif actual_phi_ratio < 0.999 and actual_phi_ratio >= 0.99:
             print(f"Info: Stationarity ratio is {actual_phi_ratio:.4f}, above 99% but below 99.9%. Acceptable for random inputs.")
    # This 'else' corresponds to 'if num_successfully_checked > 0'
    # It means no checks resulted in a "stationary" outcome (either the check itself failed or all valid checks were non-stationary)
    else:
        # num_attempted_checks is equivalent to num_successful_transforms here.
        # num_successful_transforms is defined earlier in the function.
        # If transforms were successful, then checks were attempted.
        if num_successful_transforms > 0:
            fail_messages.append(f"No samples were successfully verified as stationary out of {num_successful_transforms} that were eligible for checking (either the check itself failed for all, or all valid checks resulted in non-stationary).")
            # Ensure test_passed is set if this condition implies a failure according to test logic
            if results["phi_stationarity_ratio_of_successful_checked"] < 0.99 : # This will be true if count is 0 and checks were attempted
                 test_passed = False


    # Final check on test_passed flag based on fail_messages
    if len(fail_messages) > 0:
        test_passed = False

    if test_passed:
        print("Test Case Passed.")
    else:
        print("Test Case FAILED:")
        for msg in fail_messages:
            print(f"  - {msg}")

    print(f"--- Test for VAR({p_order}), m={m_dim} Finished ---")
    return results


# --- Main execution block ---
if __name__ == "__main__":
    print("Starting JAX Stationary Prior Transformation Test Suite (Python Loops)")
    print("======================================================================")

    # Test cases focusing on p >= 1
    test_cases = [
        # Small models - higher samples for reliability
        {"m": 1, "p": 1, "n_samples": 500, "seed": 101},
        {"m": 2, "p": 1, "n_samples": 500, "seed": 201},
        {"m": 2, "p": 2, "n_samples": 500, "seed": 202},
        {"m": 3, "p": 1, "n_samples": 500, "seed": 301},
        {"m": 3, "p": 2, "n_samples": 500, "seed": 302},
        {"m": 3, "p": 3, "n_samples": 200, "seed": 303},

        # Larger dimensions/orders - fewer samples for speed
        {"m": 4, "p": 1, "n_samples": 200, "seed": 401},
        {"m": 4, "p": 2, "n_samples": 200, "seed": 402},
        {"m": 5, "p": 1, "n_samples": 100, "seed": 501},
        {"m": 5, "p": 2, "n_samples": 100, "seed": 502},
        {"m": 5, "p": 3, "n_samples": 500, "seed": 503},
        {"m": 10, "p": 1, "n_samples": 500, "seed": 1001},
        {"m": 10, "p": 2, "n_samples": 300, "seed": 1002},
        {"m": 10, "p": 3, "n_samples": 200, "seed": 1003},
        {"m": 15, "p": 1, "n_samples": 100, "seed": 1501}

    ]


    all_results = []
    all_test_cases_passed = True

    for i, case_params in enumerate(test_cases):
        print(f"\n--- Running Test Case {i+1}/{len(test_cases)} ---")
        # Run without debug printing for the full suite
        results = run_stationary_prior_jax_test(
            m_dim=case_params["m"],
            p_order=case_params["p"],
            n_samples=case_params["n_samples"],
            key_seed=case_params["seed"]
            # enable_debug_print=False is default
        )
        all_results.append(results)
        # Update overall pass/fail status based on the 'Test Case FAILED' message
        summary_str = f"Test Summary for VAR({results['p']}), m={results['m']} FAILED:"
        if summary_str in (summary_str,) + tuple(results.get("fail_messages", [])): # Look for the specific failure message
             all_test_cases_passed = False


    print("\n======================================================")
    print("         JAX Stationary Prior Test Suite Summary      ")
    print("======================================================")
    print(f"{'Case':<8} | {'m':<3} | {'p':<3} | {'Samples':<7} | {'A Stat (%)':<10} | {'Phi Stat (Succ Checked)':<24} | {'Transform Fail (%)':<18} | {'Check Fail (%) (of attempted)':<28}")
    print("-" * 130)
    for i, res in enumerate(all_results):
        num_successful_transforms = res['n_samples'] - res['transform_failed_count']
        num_successfully_checked = num_successful_transforms - res['stationarity_check_failed_count']

        phi_ratio_successful_checked = res['phi_stationary_count'] / num_successfully_checked if num_successfully_checked > 0 else 0.0
        transform_fail_ratio = res['transform_failed_count'] / res['n_samples'] if res['n_samples'] > 0 else 0.0
        check_fail_ratio_of_attempted = res['stationarity_check_failed_count'] / num_successful_transforms if num_successful_transforms > 0 else 0.0 # Ratio vs successful transforms, not attempted checks

        # Pre-format strings to avoid complex nested f-strings in print
        a_stat_ratio_str = f"{res['A_stationarity_ratio']:.1%}"
        phi_ratio_str = f"{phi_ratio_successful_checked:.4f}" # Retained for consistency, was not the error source
        transform_fail_str = f"{transform_fail_ratio:.1%}"
        check_fail_str = f"{check_fail_ratio_of_attempted:.1%}"
        
        # Old commented line (for reference, will be removed by replace if it was part of search)
        #print(f"{f'Case {i+1}':<8} | {res['m']:<3} | {res['p']:<3} | {res['n_samples']:<7} | {f\"{res['A_stationarity_ratio']:.1%}\":<10} | {f'{phi_ratio_successful_checked:.4f}'<24} | {f'{transform_fail_ratio:.1%}'<18} | {f'{check_fail_ratio_of_attempted:.1%}'<28}")
        print(f"{f'Case {i+1}':<8} | {res['m']:<3} | {res['p']:<3} | {res['n_samples']:<7} | {a_stat_ratio_str:<10} | {phi_ratio_str:<24} | {transform_fail_str:<18} | {check_fail_str:<28}")

    # Plotting summary
    labels = [f"m={r['m']}, p={r['p']}" for r in all_results]

    phi_success_ratios = [(r["phi_stationary_count"] / (r['n_samples'] - r['transform_failed_count'] - r['stationarity_check_failed_count'])) if (r['n_samples'] - r['transform_failed_count'] - r['stationarity_check_failed_count']) > 0 else 0.0 for r in all_results]

    transform_fail_pct = [r["transform_failed_count"] / r['n_samples'] * 100 if r['n_samples'] > 0 else 0.0 for r in all_results]

    check_fail_pct = [r["stationarity_check_failed_count"] / (r['n_samples'] - r['transform_failed_count']) * 100 if (r['n_samples'] - r['transform_failed_count']) > 0 else 0.0 for r in all_results]


    x = jnp.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 9))

    rects1 = ax.bar(x - width, phi_success_ratios, width, label='Phi Stat Ratio (Successfully Checked)', color='tab:blue')
    rects2 = ax.bar(x, transform_fail_pct, width, label='Transform Failure (%)', color='tab:red')
    rects3 = ax.bar(x + width, check_fail_pct, width, label='Stationarity Check Failure (%) (of Successful Transforms)', color='tab:orange')


    ax.set_ylabel('Ratio / Percentage')
    ax.set_title('JAX Stationary Prior Transformation Test Results (Python Loops)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left')


    def autolabel_ratio(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    def autolabel_pct(rects):
         for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)


    autolabel_ratio(rects1)
    autolabel_pct(rects2)
    autolabel_pct(rects3)


    plt.tight_layout()
    plt.savefig('jax_stationary_prior_test_summary_python_loops.png') # Save with different name
    plt.close()

    print("\nResults plot saved to 'jax_stationary_prior_test_summary_python_loops.png'")

    if all_test_cases_passed:
        print("\n======================================================")
        print(" ALL JAX Stationary Prior Tests (Python Loops) PASSED ")
        print("======================================================")
    else:
        print("\n======================================================")
        print(" SOME JAX Stationary Prior Tests (Python Loops) FAILED ")
        print("======================================================")