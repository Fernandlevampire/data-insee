import pandas as pd

import utils

def extract_ipc(min_year:int=2000)->pd.DataFrame:
    """
    Get price evolution indices by year from INSEE API
    """
    json_resp = utils.get_json("https://api.insee.fr/melodi/data/DS_IPCH?FREQ=A&IND_TYPE=IW&COICOP_2018=04")

    ipc = [
        {
            "Annee": year,
            "Ind_prix": item["measures"]["OBS_VALUE_NIVEAU"]["value"] 
        }
        for item in json_resp.get("observations")
        if (year:=int(item["dimensions"]["TIME_PERIOD"])) >= min_year   
    ]

    return pd.DataFrame(ipc).sort_values(by="Annee")

def get_ipc_df(min_year:int=2000, save_file:str="ipc.csv", is_save:bool=False)->pd.DataFrame:
    ipc_df = utils.read_or_extract(
        extract_ipc,
        save_file=save_file,
        is_save=is_save,
        min_year=min_year
    )
    
    return ipc_df

if __name__ == "__main__":
    ipc_df = get_ipc_df()
    ipc_df.head()