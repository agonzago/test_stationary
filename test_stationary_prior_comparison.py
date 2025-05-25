import unittest
import numpy as np
import jax
import jax.numpy as jnp

# JAX functions to test
from stationary_prior_jax_simplified import (
    make_stationary_var_transformation_jax,
    check_stationarity_jax,
    _DEFAULT_DTYPE as JAX_DEFAULT_DTYPE
)

# PyTensor runners
from pytensor_to_numpy_utils import (
    get_numpy_runner_for_make_stationary_var_transformation,
    get_numpy_runner_for_check_stationarity
)

# Ensure PyTensor side also uses float64 if not set by its utils
import pytensor
if not hasattr(pytensor.config, 'floatX') or pytensor.config.floatX != 'float64':
    pytensor.config.floatX = 'float64'
NUMPY_DEFAULT_DTYPE = np.float64

# Ensure JAX settings match the library
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu") # Match the library's potential CPU preference
print(f"JAX backend: {jax.default_backend()}, float64 enabled: {jax.config.jax_enable_x64}")


class TestStationaryPriorComparison(unittest.TestCase):

    def _generate_sigma_np(self, m: int, key_idx: int = 0) -> np.ndarray:
        # Using numpy for this, as it's input to both
        # More control than jax.random for symmetric positive definite
        seed = 42 + key_idx # Ensure different Sigma for different tests if needed
        rng = np.random.RandomState(seed)
        X = rng.rand(m, m)
        Sigma_np = np.dot(X, X.T) + np.eye(m) * (0.01 * m) # Ensure positive definiteness
        return Sigma_np.astype(NUMPY_DEFAULT_DTYPE)

    def _generate_A_list_np(self, m: int, p: int, key_idx: int = 0) -> list[np.ndarray]:
        if p == 0:
            return []
        seed = 123 + key_idx
        rng = np.random.RandomState(seed)
        A_list_np = [rng.rand(m, m).astype(NUMPY_DEFAULT_DTYPE) for _ in range(p)]
        return A_list_np

    def _run_comparison_test(self, m: int, p: int, key_idx: int = 0, atol: float = 1e-7):
        print(f"\nRunning comparison for m={m}, p={p}")

        # 1. Generate Input Data (NumPy)
        Sigma_np = self._generate_sigma_np(m, key_idx)
        A_list_np = self._generate_A_list_np(m, p, key_idx)

        # Handle p=0 case specifically for check_stationarity
        if p == 0:
            # PyTensor side
            run_make_transform_np = get_numpy_runner_for_make_stationary_var_transformation(m, p)
            phi_list_np, gamma_list_np = run_make_transform_np(Sigma_np, A_list_np)
            is_stationary_np = True # VAR(0) is stationary by definition

            # JAX side
            Sigma_jax = jnp.array(Sigma_np, dtype=JAX_DEFAULT_DTYPE)
            A_list_jax = [jnp.array(a, dtype=JAX_DEFAULT_DTYPE) for a in A_list_np] # empty list
            phi_list_jax, gamma_list_jax = make_stationary_var_transformation_jax(Sigma_jax, A_list_jax, m, p)
            is_stationary_jax = True # VAR(0) is stationary by definition

            self.assertEqual(len(phi_list_np), 0, "phi_list_np should be empty for p=0")
            self.assertEqual(len(gamma_list_np), 0, "gamma_list_np should be empty for p=0")
            self.assertEqual(len(phi_list_jax), 0, "phi_list_jax should be empty for p=0")
            self.assertEqual(len(gamma_list_jax), 0, "gamma_list_jax should be empty for p=0")
            self.assertTrue(is_stationary_np)
            self.assertTrue(is_stationary_jax)
            print(f"Comparison for m={m}, p={p} (VAR(0)) PASSED.")
            return

        # 2. Get PyTensor Runners (p > 0)
        run_make_transform_np = get_numpy_runner_for_make_stationary_var_transformation(m, p)
        run_check_stationarity_np = get_numpy_runner_for_check_stationarity(m, p)

        # 3. Execute PyTensor Version
        print("Executing PyTensor version...")
        phi_list_np, gamma_list_np = run_make_transform_np(Sigma_np, A_list_np)
        is_stationary_np = run_check_stationarity_np(phi_list_np)
        print("PyTensor version executed.")

        # 4. Execute JAX Version
        print("Executing JAX version...")
        Sigma_jax = jnp.array(Sigma_np, dtype=JAX_DEFAULT_DTYPE)
        A_list_jax = [jnp.array(a, dtype=JAX_DEFAULT_DTYPE) for a in A_list_np]
        
        # JIT compile the JAX function for potential speed up in repeated calls (though not strictly necessary here)
        # jit_make_stationary_var_transformation_jax = jax.jit(make_stationary_var_transformation_jax, static_argnums=(2,3))
        # jit_check_stationarity_jax = jax.jit(check_stationarity_jax, static_argnums=(1,2))

        # phi_list_jax, gamma_list_jax = jit_make_stationary_var_transformation_jax(Sigma_jax, A_list_jax, m, p)
        # is_stationary_jax = jit_check_stationarity_jax(phi_list_jax, m, p)
        
        phi_list_jax, gamma_list_jax = make_stationary_var_transformation_jax(Sigma_jax, A_list_jax, m, p)
        is_stationary_jax = check_stationarity_jax(phi_list_jax, m, p)
        print("JAX version executed.")

        # 5. Compare Outputs
        self.assertEqual(len(phi_list_np), p, f"phi_list_np length mismatch for m={m}, p={p}")
        self.assertEqual(len(phi_list_jax), p, f"phi_list_jax length mismatch for m={m}, p={p}")
        self.assertEqual(len(gamma_list_np), p, f"gamma_list_np length mismatch for m={m}, p={p}")
        self.assertEqual(len(gamma_list_jax), p, f"gamma_list_jax length mismatch for m={m}, p={p}")

        for i in range(p):
            phi_np_i = phi_list_np[i]
            phi_jax_i = np.array(phi_list_jax[i]) # Convert JAX array to NumPy array for comparison
            self.assertTrue(
                np.allclose(phi_np_i, phi_jax_i, atol=atol),
                f"Phi matrix {i} mismatch for m={m}, p={p}.\nNumPy:\n{phi_np_i}\nJAX:\n{phi_jax_i}"
            )

            gamma_np_i = gamma_list_np[i]
            gamma_jax_i = np.array(gamma_list_jax[i]) # Convert JAX array to NumPy array
            self.assertTrue(
                np.allclose(gamma_np_i, gamma_jax_i, atol=atol),
                f"Gamma matrix {i} mismatch for m={m}, p={p}.\nNumPy:\n{gamma_np_i}\nJAX:\n{gamma_jax_i}"
            )
        
        # Ensure comparison is between Python bools or equivalent NumPy bools
        np_stat_bool = is_stationary_np.item() if hasattr(is_stationary_np, 'item') else bool(is_stationary_np)
        jax_stat_bool = is_stationary_jax.item() if hasattr(is_stationary_jax, 'item') else bool(is_stationary_jax)
        
        self.assertEqual(
            np_stat_bool, 
            jax_stat_bool,
            f"Stationarity check mismatch for m={m}, p={p}. NumPy: {np_stat_bool}, JAX: {jax_stat_bool}"
        )
        print(f"Comparison for m={m}, p={p} PASSED (Stationary: {np_stat_bool}).")

    # Test cases
    # def test_m1_p0(self): # VAR(0)
    #     self._run_comparison_test(m=1, p=0, key_idx=0)

    # def test_m2_p0(self): # VAR(0)
    #     self._run_comparison_test(m=2, p=0, key_idx=1)

    def test_m1_p1(self):
        self._run_comparison_test(m=1, p=1, key_idx=10)

    def test_m2_p1(self):
        self._run_comparison_test(m=2, p=1, key_idx=20, atol=1e-6) # Might need slightly higher atol for more complex

    def test_m2_p2(self):
        self._run_comparison_test(m=2, p=2, key_idx=30, atol=1e-6)

    def test_m3_p2(self):
        self._run_comparison_test(m=3, p=2, key_idx=40, atol=1e-6)
        
    def test_m1_p3(self): # Higher order p
        self._run_comparison_test(m=5, p=3, key_idx=50, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
