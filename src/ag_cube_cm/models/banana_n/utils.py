import numpy as np
from scipy.stats import lognorm

def generate_lognorm_pool(mu: float, sigma: float, size: int = 1000) -> np.ndarray:
    """
    Generates the log-normal pools exactly as R's qlnorm does.

    Parameters
    ----------
    mu : float
        Mean of the log-normal distribution (in log space).
    sigma : float
        Standard deviation of the log-normal distribution (in log space).
    size : int, optional
        Number of probabilities to evaluate, by default 1000.

    Returns
    -------
    np.ndarray
        A numpy array containing the values of the log-normal percent point function.
    """
    probs = np.arange(1, size) / size
    return lognorm.ppf(probs, s=sigma, scale=np.exp(mu))

