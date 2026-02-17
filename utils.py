import psutil
import os
import pandas as pd

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