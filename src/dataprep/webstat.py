import os
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

import utils

load_dotenv()
WEBSTAT_API_KEY = Path(os.getenv("WEBSTAT_API_KEY")) # type: ignore

def extract_mortgage_rates(min_year:int=2000)->pd.DataFrame:
    """
    Get mortgage rates evolution from Banque de France Webstat API
    Output is a pandas DataFrame with two columns (year and rate)
    """
    url = "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets/observations/exports/json"
    headers={"Authorization": f"Apikey {WEBSTAT_API_KEY}"}
    params = {
        "order_by": "time_period_start",
        "refine": "series_key:'MIR1.Q.FR.R.A22FRX.Z.R.A.2254FR.EUR.N'"
    }

    json_resp = utils.get_json(url, headers=headers, params=params)

    rates_by_year = {}
    for item in json_resp:
        if (year:=int(item["time_period_start"][:4])) >= min_year:
            rates_by_year.setdefault(year, []).append(item["obs_value"])

    avg_rates = [
        {
            "Annee": year,
            "Taux_immo": np.mean(values)
        } 
        for year, values in rates_by_year.items()
    ]

    return pd.DataFrame(avg_rates)

def get_mortgage_rates_df(min_year:int=2000, save_file:str="mortgage_rates.csv", is_save:bool=False)->pd.DataFrame:
    rates_df = utils.read_or_extract(
        extract_mortgage_rates, 
        save_file=save_file, 
        is_save=is_save, 
        min_year=min_year
    )
    
    return rates_df

if __name__ == "__main__":
    rates_df = get_mortgage_rates_df()
    rates_df.head()