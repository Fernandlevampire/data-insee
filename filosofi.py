import pandas as pd
import requests
import time

"""
filosifi is an INSEE dataset that provides information on socio-economic characteristics of french territories at different scales
it can be requested through INSEE's API called Melodi
In this project, we extract and try to improve data readability
"""

def request_melodi_filosifi(page:int=1):
    response = requests.get(
            "https://api.insee.fr/melodi/data/DS_FILOSOFI_CC" \
            f"?page={page}"
        )

    if status:=response.status_code != 200:
        print(f"Request did not go through with page {page}. Error: {status}")
        if status == 429:
            # melodi request number is 30 per minute. Wait and start requesting again
            print("Reached request number limit. Waiting 1 minute before trying again...")
            time.sleep(60)
            print("Proceed")
            response = requests.get(
                "https://api.insee.fr/melodi/data/DS_FILOSOFI_CC" \
                f"?page={page}"
            )
            print(f"Response status: {response.status_code}")
        
    return response.json()

def get_melodi_filosifi_data() -> pd.DataFrame:
    data_dict = {}
    page = 1

    while True:
        json_response = request_melodi_filosifi(page)
        page += 1
        # loop will stop as soon as response does not include data (observations is empty)
        if observations:=json_response.get("observations"):
            for item in observations:
                geo, measure_id = item["dimensions"]["GEO"], item["dimensions"]["FILOSOFI_MEASURE"]
                # if value is defined, store it in data_dict
                try:
                    value = item["measures"]["OBS_VALUE_NIVEAU"]["value"]
                    # if geo is already a line, complete it
                    try:
                        data_dict[geo][measure_id] = value
                    # else create the line and complete it
                    except KeyError:
                        year, region_type, code = geo.split("-")
                        data_dict[geo] = {
                            "echelle": region_type, 
                            "code_INSEE": code, 
                            "annee": year,
                            measure_id: value
                        }
                # if value is not defined, pass to next observation item
                except KeyError:
                    pass
        else:
            break
    
    # here we only want to keep values on communal scale
    return pd.DataFrame(data_dict.values())

def reindex_filosofi(df: pd.DataFrame) -> pd.DataFrame:
    columns_order = [
        "echelle",
        "code_INSEE",
        "annee",
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

def main() -> pd.DataFrame:
    return reindex_filosofi(get_melodi_filosifi_data())

if __name__ == "__main__":
    df = main()
    df.head()