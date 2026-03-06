import psutil
import os
import pandas as pd
import numpy as np

from scipy import optimize
from typing import Callable, Dict, Any

"""system"""
def check_ram(message:str="check RAM: "):
    process = psutil.Process(os.getpid())
    print(f"{message}: {process.memory_info().rss / (1024**2):.2f} Mb")

"""formating"""
def reformat_number(number:float|np.float64|int|np.int64)->str:
    """
    Return a number formatted for french norm
    - " " for thousands separator
    - "," for decimal separator
    """
    if type(number) == int or type(number) == np.int64:
        return f"{number:,}".replace(",", " ")
    if type(number) == float or type(number) == np.float64:
        return f"{number:,.2f}".replace(",", " ").replace(".", ",")
    print("neither int nor float", type(number))
    return str(number)

"""dataframes"""
def apply_masks(df: pd.DataFrame, *mask_funcs) -> pd.DataFrame:
    """
    Apply multiple masks to a dataframe
    mask inputs are expected as functions of type df: pd.DataFrame -> mask
    """
    if not mask_funcs:
        return df
    
    combined_mask = mask_funcs[0](df)
    for mask in mask_funcs[1:]:
        combined_mask = combined_mask & mask(df)
    
    return df.loc[combined_mask]

"""statistics""" 
def outliers_limits(data:pd.Series, min_val:float=float("-inf"), max_val:float=float("inf"), coeff_IQR:float=1.5) -> tuple[float, float]:
    quartiles = data.quantile(np.r_[0:1.01:0.25]) 
    interq = quartiles[0.75] - quartiles[0.25]
    x_min, x_max = max(min_val, quartiles[0.25]-coeff_IQR*interq), min(max_val, quartiles[0.75]+coeff_IQR*interq)

    return x_min, x_max

def remove_outliers_df(df:pd.DataFrame, col: str, min_val:float=float("-inf"), max_val:float=float("inf"), coeff_IQR:float=1.5) -> pd.DataFrame:
    """
    Remove mild outliers according to Tukey method
    Keep values in range: [Q1-coeff*IQR:Q3+coeff*IQR] 
    where IQR = interquartile range
    """ 
    df = df.copy()

    x_min, x_max = outliers_limits(data=df[col], min_val=min_val, max_val=max_val, coeff_IQR=coeff_IQR)

    return df.loc[(df[col] > x_min) & (df[col] < x_max)]

def remove_outliers(data:pd.Series, min_val:float=float("-inf"), max_val:float=float("inf"), coeff_IQR:float=1.5) -> pd.Series:
    """
    Remove mild outliers according to Tukey method
    Keep values in range: [Q1-coeff*IQR:Q3+coeff*IQR] (IQR = interquartile range)
    """ 
    x_min, x_max = outliers_limits(data=data, min_val=min_val, max_val=max_val, coeff_IQR=coeff_IQR)

    return data[(data > x_min) & (data < x_max)]

def model_estimate(y: np.ndarray, func: Callable, x: np.ndarray, popt: tuple) -> Dict:
    """compute model error (R² and standard error)""" 
    y_pred = func(x, *popt)
    residus = y - y_pred
    ss_res = np.sum(residus**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    se = np.sqrt(ss_res / (len(y) - len(popt)))
    return {
        "r_squared": r_squared, 
        "std_error": se, 
    }

def estimate_bins(data:pd.Series) -> int:
    return int(len(data)/3)

def gauss(x, a, x0, sigma) -> float:
    """asymetric gaussian distribution"""
    return a*np.exp(-(x-x0)**2 / (2*sigma**2))

def gauss_opt_params(data:pd.Series, bins:int=50) -> dict:
    """
    Return optimal a, x0 and sigma to approximate data as an asymetric gaussian distribution:
    gauss = a*np.exp(-(x-x0)**2 / (2*sigma**2))
    """
    n_bins = min(estimate_bins(data), bins)
    y, _ = np.histogram(data, bins=n_bins, density=False, weights=np.ones_like(data)/len(data))
    x = np.linspace(data.min(), data.max(), n_bins)

    (a, x0, sigma), _ = optimize.curve_fit(gauss, x, y, p0=[max(y), data.mean(), np.std(data)], bounds=([0, 500, 0], 20000))
    estimate = model_estimate(y, gauss, x, (a, x0, sigma))

    return {"a": a, "x0": x0, "sigma": sigma} | estimate

def gauss_mode(data:pd.Series, bins:int=50) -> float:
    """return mode (highest frequency value) for asymetric gaussian approximation of data"""
    if len(data)<100:
        return data.median()
    try:
        params = gauss_opt_params(data, bins)
        x0, r_squared  = params["x0"], params["r_squared"]
        return x0 if r_squared > 0.9 else data.median()
    except ValueError:
        return data.median()