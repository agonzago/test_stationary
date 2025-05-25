import pytensor
import pytensor.tensor as pt
import numpy as np

# Import specific functions from stationary_prior
# These are the main entry points we want to wrap.
# PyTensor will automatically pull in their dependencies (AtoP, sqrtm, etc.)
# when compiling the graph.
from stationary_prior import make_stationary_var_transformation, check_stationarity

# Ensure floatX is consistent and set to float64 for precision
if not hasattr(pytensor.config, 'floatX') or pytensor.config.floatX != 'float64':
    pytensor.config.floatX = 'float64'

def get_numpy_runner_for_make_stationary_var_transformation(m: int, p: int):
    """
    Returns a callable function that executes make_stationary_var_transformation 
    from stationary_prior.py with NumPy inputs and returns NumPy outputs.
    
    Args:
        m: Dimension of the VAR process.
        p: Order of the VAR process.
    """
    # Define symbolic PyTensor input variables
    Sigma_pt_input = pt.matrix('Sigma_pt_input')
    A_list_pt_inputs = [pt.matrix(f'A_pt_input_{i}') for i in range(p)]

    # Call the original PyTensor function with these symbolic variables
    # The actual m and p are Python integers, used by the PyTensor function internally
    # for loops, shaping, etc. They are not symbolic inputs in the graph itself.
    phi_list_pt_out, gamma_list_pt_out = make_stationary_var_transformation(
        Sigma_pt_input, A_list_pt_inputs, m, p
    )

    # Prepare inputs and outputs for pytensor.function
    # Inputs must be a flat list of the PyTensor variables
    inputs_for_compile = [Sigma_pt_input] + A_list_pt_inputs
    # Outputs must be a flat list of the PyTensor variables
    # phi_list_pt_out and gamma_list_pt_out are already lists of pt.TensorVariable
    outputs_for_compile = phi_list_pt_out + gamma_list_pt_out

    # Compile the PyTensor graph into a callable Python function
    compiled_func = pytensor.function(
        inputs=inputs_for_compile, 
        outputs=outputs_for_compile,
        on_unused_input='warn' # Useful for debugging graph construction
    )

    def numpy_runner(Sigma_np: np.ndarray, A_list_np: list[np.ndarray]):
        """
        Takes NumPy arrays as input, passes them to the compiled PyTensor function,
        and returns the NumPy array outputs.
        """
        if not isinstance(Sigma_np, np.ndarray):
            raise TypeError(f"Sigma_np must be a NumPy array, got {type(Sigma_np)}")
        if not all(isinstance(A_i, np.ndarray) for A_i in A_list_np):
            raise TypeError("All elements in A_list_np must be NumPy arrays.")
        if Sigma_np.shape != (m, m):
            raise ValueError(f"Sigma_np expected shape ({m},{m}), got {Sigma_np.shape}")
        if len(A_list_np) != p:
            raise ValueError(f"Expected A_list_np to have {p} elements, got {len(A_list_np)}")
        for i, A_i_np in enumerate(A_list_np):
            if A_i_np.shape != (m, m):
                raise ValueError(f"A_list_np[{i}] expected shape ({m},{m}), got {A_i_np.shape}")

        # Prepare inputs for the compiled function call
        call_inputs = [Sigma_np.astype(pytensor.config.floatX)] + \
                      [A_i.astype(pytensor.config.floatX) for A_i in A_list_np]
        
        results_pt_values = compiled_func(*call_inputs)
        
        # Results are returned as a flat list, reconstruct phi_list and gamma_list
        # phi_list has p elements, gamma_list has p elements
        phi_list_np_out = results_pt_values[:p]
        gamma_list_np_out = results_pt_values[p:]
        
        return phi_list_np_out, gamma_list_np_out
            
    return numpy_runner

def get_numpy_runner_for_check_stationarity(m: int, p: int):
    """
    Returns a callable function that executes check_stationarity 
    from stationary_prior.py with NumPy inputs and returns a NumPy output.
    
    Args:
        m: Dimension of the VAR process.
        p: Order of the VAR process.
    """
    # Define symbolic PyTensor input variables
    phi_list_pt_inputs = [pt.matrix(f'phi_pt_input_{i}') for i in range(p)]
    
    # Call the original PyTensor function with these symbolic variables
    is_stationary_pt_out = check_stationarity(phi_list_pt_inputs, m, p)

    # Compile the PyTensor graph
    compiled_func = pytensor.function(
        inputs=phi_list_pt_inputs, 
        outputs=is_stationary_pt_out,
        on_unused_input='warn'
    )

    def numpy_runner(phi_list_np: list[np.ndarray]):
        """
        Takes a list of NumPy arrays as input, passes them to the compiled PyTensor function,
        and returns the NumPy boolean/scalar output.
        """
        if not all(isinstance(phi_i, np.ndarray) for phi_i in phi_list_np):
            raise TypeError("All elements in phi_list_np must be NumPy arrays.")
        if len(phi_list_np) != p:
            raise ValueError(f"Expected phi_list_np to have {p} elements, got {len(phi_list_np)}")
        for i, phi_i_np in enumerate(phi_list_np):
            if phi_i_np.shape != (m, m):
                raise ValueError(f"phi_list_np[{i}] expected shape ({m},{m}), got {phi_i_np.shape}")

        call_inputs = [phi_i.astype(pytensor.config.floatX) for phi_i in phi_list_np]
        result_np = compiled_func(*call_inputs)
        
        # The result should be a NumPy scalar (e.g., array(True))
        return np.bool_(result_np) 
            
    return numpy_runner

if __name__ == "__main__":
    print("Running basic tests for pytensor_to_numpy_utils...")
    
    # Test parameters
    m_test = 2
    p_test = 2
    
    # Create sample NumPy data
    Sigma_np_test = np.eye(m_test, dtype=pytensor.config.floatX)
    A_list_np_test = [np.random.rand(m_test, m_test).astype(pytensor.config.floatX) for _ in range(p_test)]
    
    # --- Test make_stationary_var_transformation ---
    print(f"\nTesting make_stationary_var_transformation (m={m_test}, p={p_test})...")
    try:
        runner_transform = get_numpy_runner_for_make_stationary_var_transformation(m_test, p_test)
        print("Runner created.")
        
        phi_list_out, gamma_list_out = runner_transform(Sigma_np_test, A_list_np_test)
        print("NumPy runner executed.")
        
        assert len(phi_list_out) == p_test, f"Expected {p_test} phi matrices, got {len(phi_list_out)}"
        assert len(gamma_list_out) == p_test, f"Expected {p_test} gamma matrices, got {len(gamma_list_out)}"
        
        print(f"Phi matrices ({len(phi_list_out)}):")
        for i, phi in enumerate(phi_list_out):
            assert isinstance(phi, np.ndarray), f"phi[{i}] is not a NumPy array"
            assert phi.shape == (m_test, m_test), f"phi[{i}] has incorrect shape {phi.shape}"
            print(f"phi_{i} shape: {phi.shape}, dtype: {phi.dtype}")
            # print(phi)
            
        print(f"\nGamma matrices ({len(gamma_list_out)}):")
        for i, gamma in enumerate(gamma_list_out):
            assert isinstance(gamma, np.ndarray), f"gamma[{i}] is not a NumPy array"
            assert gamma.shape == (m_test, m_test), f"gamma[{i}] has incorrect shape {gamma.shape}"
            print(f"gamma_{i} shape: {gamma.shape}, dtype: {gamma.dtype}")
            # print(gamma)
        
        print("\nmake_stationary_var_transformation test PASSED.")
        
        # --- Test check_stationarity ---
        # Use the phi_list_out from the previous test
        print(f"\nTesting check_stationarity (m={m_test}, p={p_test})...")
        runner_stationarity = get_numpy_runner_for_check_stationarity(m_test, p_test)
        print("Runner created.")
        
        is_stationary = runner_stationarity(phi_list_out) # phi_list_out is already a list of np arrays
        print("NumPy runner executed.")
        
        assert isinstance(is_stationary, np.bool_), \
            f"Expected bool output, got {type(is_stationary)}"
        print(f"Is stationary: {is_stationary} (type: {type(is_stationary)})")
        
        print("\ncheck_stationarity test PASSED.")
        
        # Example of non-stationary VAR (p=1)
        phi_non_stationary_p1 = [np.array([[1.1, 0.0], [0.0, 0.5]], dtype=pytensor.config.floatX)]
        m_p1, p_p1 = 2, 1
        runner_stat_p1 = get_numpy_runner_for_check_stationarity(m_p1, p_p1)
        is_stat_p1 = runner_stat_p1(phi_non_stationary_p1)
        print(f"\nCheck non-stationary (p=1): {is_stat_p1}")
        assert not is_stat_p1, "Non-stationary check failed."

        # Example of stationary VAR (p=1)
        phi_stationary_p1 = [np.array([[0.5, 0.0], [0.0, 0.3]], dtype=pytensor.config.floatX)]
        is_stat_p1_s = runner_stat_p1(phi_stationary_p1)
        print(f"Check stationary (p=1): {is_stat_p1_s}")
        assert is_stat_p1_s, "Stationary check failed."


    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    print("\nBasic tests completed.")
