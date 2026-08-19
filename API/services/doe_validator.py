import numpy as np
from Utils.get_con_db import get_connection

def validate_doe(doe, lb, ub, conn):
    
    with conn.cursor() as cur:
        cur.execute("SELECT num_points, problem_dim FROM sampling_sets")
        result = cur.fetchone()

        if result is None:
            raise alueError("Sampling-set configuration was not found.")

        num_sample_points, prob_dim = result
        
    if not isinstance(doe, np.ndarray):
        raise TypeError("DOE must be a NumPy array.")
    
    if not isinstance(lb, np.ndarray):
        raise TypeError("Lower bound vector must be a NumPy array.")

    if not isinstance(ub, np.ndarray):
        raise TypeError("Upper bound vector must be a NumPy array.")

    if doe.shape[0] != num_sample_points:
        raise ValueError("The number of points in the DOE does not match the number of required sampling points.")
    
    if doe.shape[1] != prob_dim+1:
        raise ValueError("The number of dimensions in the DOE does not match the required problem dimension.")
    
    if lb.shape[0] != prob_dim:
        raise ValueError("The lower bound vector does not match the required problem dimension.")

    if ub.shape[0] != prob_dim:
        raise ValueError("The upper bound vector does not match the required problem dimension.")

    for i in range(prob_dim):
        if not np.isfinite(doe[:, i]).all():
            raise ValueError(f"Variable {i+1} in the DOE contains non-finite values.")
        if np.any(doe[:, i] < lb[i]) or np.any(doe[:, i] > ub[i]):
            raise ValueError(f"Variable {i+1} in the DOE is out of bounds.")
        
    if not np.isfinite(doe[:, -1]).all():
        raise ValueError("The objective values column in the DOE contains non-finite values.")
                
    return