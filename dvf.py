import numpy as np
import pandas as pd

from utils import print_memory, apply_masks

"""
DVF is a french public dataset that provides information on land sales in
it cannot yet (as of year 2026) be requested through API
In this project, we extract and try to improve data readability from dvf dataset

>>> data source : https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres

input data come as .txt files formatted as csv with | separator
input format is based on current version of dvf database as of year 2026
"""

def dvf_extract(file_path:str, mask_funks:list=[]):
    print(file_path)
    print_memory(message="Memory before reading")
    # drop useless columns : 0->7 are empty, "1er lot">"5eme lot" are unexploitable
    cols_base, cols_ex = np.r_[8:40, 42], np.r_[24:33:2, 37]
    cols = np.setdiff1d(cols_base, cols_ex)

    dtype_dict = {col: str for col in np.r_[11:20, 24:33:2]}
    chunks = pd.read_csv(file_path, sep="|", chunksize=1000000, usecols=cols, dtype=dtype_dict, low_memory=False)
    
    # apply masks to each chunk
    if mask_funks:
        chunks = [apply_masks(chunk, *mask_funks) for chunk in chunks]
    
    return chunks

def dvf_reformat(df:pd.DataFrame, local_type:int=0)->pd.DataFrame:
    df = df.copy()

    # sum surface_carrez of all lots and drop detailed data 
    surface_carrez_cols = df.filter(regex=("Surface Carrez du.*")).columns
    for col in surface_carrez_cols:
        df[col] = df[col].str.replace(",", ".").astype(float)
    df["Surface Carrez"] = df.filter(regex=("Surface Carrez du.*")).sum(axis=1)
    df = df.drop(columns=surface_carrez_cols)

    # reformat and group address and get rid of detailed data ; for further "group by" operations, "Code voie" can be used
    address_cols = ["No voie", "B/T/Q", "Type de voie", "Voie"]
    for col in address_cols:
        df[col] = df[col].fillna("")
    df["Code postal"] = df["Code postal"].fillna("")
    df["Adresse"] = (
        df["No voie"] + df["B/T/Q"] + np.where(df["No voie"] + df["B/T/Q"] != "", " ", "")
        + df["Type de voie"] + np.where(df["Type de voie"] != "", " ", "")
        + df["Voie"] + np.where(df["Voie"] != "", " ", "")
        + df["Code postal"] + np.where(df["Code postal"] != "", " ", "")
        + df["Commune"]
    )
    df = df.drop(columns=address_cols)

    # reformat numerical data
    df["Valeur fonciere"] = df["Valeur fonciere"].str.replace(",", ".").astype(float)
    df["Date mutation"] = pd.to_datetime(df["Date mutation"], format="%d/%m/%Y")
    df["Annee"] = df["Date mutation"].astype("datetime64[ns]").dt.year
    df["Code type local"] = df["Code type local"].fillna(0).astype("int32")
    df["Nombre pieces principales"] = df["Nombre pieces principales"].fillna(0).astype("int32")
    df["Surface reelle bati"] = df["Surface reelle bati"].fillna(0)

    # compute price per sq meter for each sale    
    df["Prix au m2"] = np.nan
    df.loc[df["Surface reelle bati"]>0, "Prix au m2"] = df["Valeur fonciere"]/df["Surface reelle bati"]

    col_order = [
        "Date mutation",
        "Annee",
        "Nature mutation",
        "Adresse",
        "Code voie",
        "Code postal",
        "Commune",
        "Code commune",
        "Code departement",
        "Prefixe de section",
        "Section",
        "No plan",
        "No Volume",
        "Nombre de lots",
        "Code type local",
        "Type local",
        "Nombre pieces principales",
        "Valeur fonciere",
        "Surface reelle bati",
        "Surface terrain",
        "Surface Carrez",
        "Prix au m2"
        ]
    df = df.reindex(columns=col_order)

    return df

def remove_outliers(df:pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    quartiles = df[col].quantile(np.r_[0:1.01:0.25]) 
    interq = quartiles[0.75] - quartiles[0.25]
    x_min, x_max = max(100,quartiles[0.25]-1.5*interq), quartiles[0.75]+1.5*interq

    df = df.loc[(df[col] > x_min) & (df[col] < x_max)]
    return df

if __name__ == "__main__":

    file_path = "data_test/ValeursFoncieres-2021-2025-sample.txt"

    masks = [
        lambda df: df["Code type local"].isin([1, 2]),
        lambda df: df["Code departement"] == "69"
    ]
    chunks = dvf_extract(file_path, masks)
    # Exemple : Calculer la moyenne d'une colonne pour chaque bloc
    test = 1
    for i, chunk in enumerate(chunks):
        print(f"Computing block {i}")
        chunk = dvf_reformat(chunk)
        price_avg = chunk["Prix au m2"].mean()
        price_med = chunk["Prix au m2"].median()
        print(f"Chunk shape: {chunk.shape}, Mean price per m²: {price_avg:.2f}, Median price per m²: {price_med:.2f}")
        if test:
            print(chunk.columns)
            test = 0