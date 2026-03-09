import pandas as pd

import utils

"""
filosifi is an INSEE dataset that provides information on socio-economic characteristics of french territories at different scales
it can be requested through INSEE's API called Melodi
In this project, we extract and try to improve data readability
"""
def extract_filosofi() -> pd.DataFrame:
    data_dict = {}
    page = 0

    while True:
        page += 1
        params = {"page": page}
        json_response = utils.get_json("https://api.insee.fr/melodi/data/DS_FILOSOFI_CC", params=params)

        # loop will stop as soon as response does not include data (observations is empty)
        if observations:=json_response.get("observations"):
            for item in observations:
                geo, measure_id = item["dimensions"]["GEO"], item["dimensions"]["FILOSOFI_MEASURE"]
                try:
                    # if value is defined, store it in data_dict
                    value = item["measures"]["OBS_VALUE_NIVEAU"]["value"]
                    
                    if geo not in data_dict:
                        year, region_type, code = geo.split("-")
                        data_dict[geo] = {
                            "Echelle": region_type,
                            "Code_INSEE": code,
                            "Annee": year
                        }

                    data_dict[geo][measure_id] = value
                
                except KeyError:
                    # if value is not defined, pass to next observation item
                    pass
        else:
            break
    

    filosofi_df = pd.DataFrame(data_dict.values())
    
    return reindex_filosofi(filosofi_df)

def reindex_filosofi(df: pd.DataFrame) -> pd.DataFrame:
    columns_order = [
        "Echelle",
        "Code_INSEE",
        "Annee",
        "NUM_HH",
        "NUM_PER",
        "NUM_CU",
        "PR_MD60",
        "D1_SL",
        "MED_SL",
        "D9_SL",
        "IR_D9_D1_SL",
        "S_HH_TAX",
        "S_EI_DI",
        "S_EI_DI_SAL",
        "S_EI_DI_N_SAL",
        "S_EI_DI_UNE",
        "S_SOC_BEN_DI",
        "S_SOC_BEN_DI_FAM_BEN",
        "S_SOC_BEN_DI_HOU_BEN",
        "S_SOC_BEN_DI_MIN_SOC",
        "S_RET_PEN_DI",
        "S_INC_ASS_DI",
        "S_DIR_TAX_DI"
    ]
    column_names_dict = {
        "NUM_HH": "Menages_fiscaux",
        "NUM_PER": "Nb_personnes",
        "NUM_CU": "Unites_conso",
        "PR_MD60": "Taux_pauvrete",
        "D1_SL": "Salaire_D1",
        "MED_SL": "Salaire_median",
        "D9_SL": "Salaire_D9",
        "IR_D9_D1_SL": "Salaire_D9_D1",
        "S_HH_TAX": "Part_menages_imposes",
        "S_EI_DI": "Part_revenus_acti",
        "S_EI_DI_SAL": "Part_salaires",
        "S_EI_DI_N_SAL": "Part_acti_non_salar",
        "S_EI_DI_UNE": "Part_chomage",
        "S_SOC_BEN_DI": "Part_prest_soc",
        "S_SOC_BEN_DI_FAM_BEN": "Part_prest_famil",
        "S_SOC_BEN_DI_HOU_BEN": "Part_prest_logt",
        "S_SOC_BEN_DI_MIN_SOC": "Part_min_sociaux",
        "S_RET_PEN_DI": "Part_pensions_retraites",
        "S_INC_ASS_DI": "Part_patrimoine",
        "S_DIR_TAX_DI": "Part_impots",
    }

    # for better readability, we need to rename and reorder columns
    df = df.copy()
    df = df.reindex(columns=columns_order)
    df = df.rename(columns=column_names_dict)
    
    return df

def get_filosofi_df(save_file:str="filosofi.csv", is_save:bool=False) -> pd.DataFrame:
    filosofi_df = utils.read_or_extract(
        extract_filosofi,
        save_file=save_file,
        is_save=is_save
    )
    
    return filosofi_df

def filter_on_communes(filosofi_df:pd.DataFrame)->pd.DataFrame:
    fil_communes_df = filosofi_df.loc[(filosofi_df["Echelle"] == "COM")]
    fil_communes_df = fil_communes_df.drop(columns=["Echelle", "Annee"])
    return fil_communes_df

if __name__ == "__main__":
    filosofi_df = get_filosofi_df()
    filosofi_df.head()