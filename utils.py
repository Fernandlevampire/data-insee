import psutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from typing import Callable, Dict, Any

"""system"""
def print_memory(message:str):
    process = psutil.Process(os.getpid())
    print(f"{message}: {process.memory_info().rss / (1024**2):.2f} Mb")

"""dataframes"""
def apply_masks(df: pd.DataFrame, *mask_funcs) -> pd.DataFrame:
    """
    Apply multiple boolean masks to a dataframe with AND condition.
    """
    if not mask_funcs:
        return df
    
    combined_mask = mask_funcs[0](df)
    for mask in mask_funcs[1:]:
        combined_mask = combined_mask & mask(df)
    
    return df.loc[combined_mask]

"""formating"""
def reformat_number(number:float|np.float64|int|np.int64)->str:
    if type(number) == int or type(number) == np.int64:
        return f"{number:,}".replace(",", " ")
    if type(number) == float or type(number) == np.float64:
        return f"{number:,.2f}".replace(",", " ").replace(".", ",")
    print("neither int nor float", type(number))
    return str(number)

"""statistics""" 
# compute model error
def model_estimate(y: np.ndarray, func: Callable, x: np.ndarray, popt: tuple) -> Dict:
    y_pred = func(x, *popt)
    residus = y - y_pred
    ss_res = np.sum(residus**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    se = np.sqrt(ss_res / (len(y) - len(popt)))
    print(f"R²: {r_squared:2f}, std error: {se:2f}")
    return {
        "r_squared": r_squared, 
        "std_error": se, 
    }

# asymetric gaussian distribution
def gauss(x, a, x0, sigma) -> float:
    return a*np.exp(-(x-x0)**2 / (2*sigma**2))

def gauss_opt_params(data:pd.Series, bins: int) -> dict:
    n_bins = min(int(len(data)/10), bins)
    y, _ = np.histogram(data, bins=n_bins, density=False, weights=np.ones_like(data)/len(data))

    x = np.linspace(data.min(), data.max(), n_bins)
    (a, x0, sigma), _ = optimize.curve_fit(gauss, x, y, p0=[max(y), data.mean(), np.std(data)])

    return {"a": a, "x0": x0, "sigma": sigma}

def gauss_mode(data:pd.Series, bins: int) -> float:
    return gauss_opt_params(data, bins)["x0"]