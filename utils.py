import psutil
import os
import pandas as pd
import numpy as np

from typing import Callable, Dict, Any

def print_memory(message:str):
    process = psutil.Process(os.getpid())
    print(f"{message}: {process.memory_info().rss / (1024**2):.2f} Mb")

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

def reformat_number(number:float|np.float64|int|np.int64)->str:
    if type(number) == int or type(number) == np.int64:
        return f"{number:,}".replace(",", " ")
    if type(number) == float or type(number) == np.float64:
        return f"{number:,.2f}".replace(",", " ").replace(".", ",")
    print("neither int nor float", type(number))
    return str(number)


def model_estimate(y: np.ndarray, func: Callable, x: np.ndarray, popt: tuple) -> Dict:
    # Calcul des résidus
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