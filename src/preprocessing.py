import numpy as np
from scipy.signal import savgol_filter

def detrend_light_curve(signal: np.ndarray, window_length: int = 101, polyorder: int = 3) -> np.ndarray:
    """
    Removes a low-frequency stellar trend from the time series (Detrending).

    Args:
        signal (np.ndarray): A raw array of star brightness.
        window_length (int): The window size for the Savitzky-Golay filter.
        polyorder (int): The order of the smoothing polynomial.
        
    Returns:
        np.ndarray: A cleared signal with a baseline of 1.0.
    """
    trend = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
    clean_signal = (signal - trend) + 1.0
    return clean_signal