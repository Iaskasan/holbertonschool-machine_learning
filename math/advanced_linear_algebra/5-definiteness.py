import numpy as np

def definiteness(matrix):
    """Determines the definiteness of a matrix."""
    
    # Check if numpy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    # Check if square matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    
    # Check if symmetric
    if not np.allclose(matrix, matrix.T):
        return None
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Tolerance for floating point errors
    tol = 1e-8
    
    # Classify eigenvalues
    positive = np.all(eigenvalues > tol)
    non_negative = np.all(eigenvalues >= -tol)
    negative = np.all(eigenvalues < -tol)
    non_positive = np.all(eigenvalues <= tol)
    
    if positive:
        return "Positive definite"
    elif non_negative:
        return "Positive semi-definite"
    elif negative:
        return "Negative definite"
    elif non_positive:
        return "Negative semi-definite"
    elif np.any(eigenvalues > tol) and np.any(eigenvalues < -tol):
        return "Indefinite"
    
    return None
