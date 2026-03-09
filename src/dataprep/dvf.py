import numpy as np
import pandas as pd
import os

from pathlib import Path
from dotenv import load_dotenv

import utils

"""
DVF is a french public dataset that provides information on land sales in
it cannot yet (as of year 2026) be requested through API
In this project, we extract and try to improve data readability from dvf dataset

>>> data source : https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres

input data come as .txt files formatted as csv with | separator
input format is based on current version of dvf database as of year 2026
"""

load_dotenv()
DVF_SRC_FOLDER = Path(os.getenv("DVF_SRC_FOLDER")) # type: ignore

def read_dvf(file_path:str, mask_funks:list=[], check_memory:bool=False):
    print(file_path)
    if check_memory:
        utils.check_ram(message="Memory before reading")
    # drop useless columns : 0->7 are empty, "1er lot" to "5eme lot" are unexploitable
    cols_base, cols_ex = np.r_[8:40, 42], np.r_[24:33:2, 37]
    cols = np.setdiff1d(cols_base, cols_ex)

    dtype_dict = {col: "string" for col in np.r_[11:20, 24:33:2]}
    chunks = pd.read_csv(file_path, sep="|", chunksize=1000000, usecols=cols, dtype=dtype_dict, low_memory=False)
    
    # apply masks to each chunk
    if mask_funks:
        chunks = [utils.apply_masks(chunk, *mask_funks) for chunk in chunks]
    
    return chunks

def reformat_dvf(df:pd.DataFrame, min_price:float=0, max_price:float=0)->pd.DataFrame:
    df = df.copy()

    # sum surface_carrez of all lots and drop detailed data 
    surface_carrez_cols = df.filter(regex=("Surface Carrez du.*")).columns
    for col in surface_carrez_cols:
        df[col] = df[col].str.replace(",", ".").astype(float)
    df["Surface Carrez"] = df.filter(regex=("Surface Carrez du.*")).sum(axis=1)
    df = df.drop(columns=surface_carrez_cols)

    # reformat and group address and get rid of useless adress data
    address_cols = ["No voie", "B/T/Q", "Type de voie", "Voie"]
    for col in address_cols:
        df[col] = df[col].fillna("")
    df["Code postal"] = df["Code postal"].fillna("")
    df["Code departement"] = df["Code departement"].fillna("").astype(str).str.zfill(2)
    df["Code commune"] = df["Code commune"].fillna("").astype(str).str.zfill(3)
    df["Code INSEE"] = df["Code departement"] + df["Code commune"]
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
    if min_price:
        df = df.loc[df["Prix au m2"]>min_price]
    if max_price:
        df = df.loc[df["Prix au m2"]<max_price]
    
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
        "Code INSEE",
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

def extract_dvf()->pd.DataFrame:
    # last try: ~5 minutes processing
    files_list = [os.path.join(DVF_SRC_FOLDER, file) for file in os.listdir(DVF_SRC_FOLDER)]

    mask_funcs = [
        # only houses and appartments
        lambda df: df["Code type local"].isin([1, 2])
    ]

    all_chunks = [
        reformat_dvf(chunk) 
        for file_path in files_list 
        for chunk in read_dvf(file_path, mask_funcs, check_memory=True)]

    dvf_tot = pd.concat(all_chunks)
    return dvf_tot

def get_dvf_df(save_file:str="dvf_tot.csv", is_save:bool=False)->pd.DataFrame:
    dtype_dict={
        "Code departement": "string",
        "Code commune": "string",
        "Code INSEE": "string"
    }
    
    dvf_tot = utils.read_or_extract(
        extract_dvf,
        save_file=save_file,
        is_save=is_save,
        dtype_dict=dtype_dict
    )
    
    return dvf_tot

def clean_dvf()->pd.DataFrame:
    # last try: ~135 minutes processing
    print("extract files from source folder. This process could take a couple hours")
    dvf_tot = get_dvf_df()

    dvf_clean = dvf_tot.copy()

    # to clean the dvf database, we want to get rid, at communal scales, of outliers in terms of price per sq. meter
    codes_INSEE = dvf_clean["Code INSEE"].unique()
    chunks_communes = []
    for code in codes_INSEE:
        df_commune = dvf_clean.loc[(dvf_clean["Code INSEE"] == code) & (dvf_clean["Prix au m2"] < 30000)]
        
        chunks_communes.append(
            utils.remove_outliers_df(df_commune, col="Prix au m2") if df_commune.shape[0] > 5 else df_commune
        )

    return pd.concat(chunks_communes)

def get_dvf_clean(save_file:str="dvf_clean.csv", is_save:bool=False)->pd.DataFrame:
    dtype_dict={
        "Code departement": "string",
        "Code commune": "string",
        "Code INSEE": "string"
    }
    
    dvf_clean = utils.read_or_extract(
        clean_dvf,
        save_file=save_file,
        is_save=is_save,
        dtype_dict=dtype_dict
    )
    
    return dvf_clean

def group_by_commune()->pd.DataFrame:
    dvf_clean = get_dvf_clean()
    # last try: ~1 min 30 sec processing
    excluded_dep = {"971", "972", "973", "974"}
    dvf_communes = (
        dvf_clean.loc[~dvf_clean["Code departement"].isin(excluded_dep)]
        .groupby(["Code INSEE", "Annee"], as_index=False)
        .agg(
            Commune=("Commune", "max"),
            Code_departement=("Code departement", "min"),
            Nb_ventes=("Date mutation", "count"),
            Valeur_fonciere=("Valeur fonciere", "sum"),
            Surface_reelle_bati=("Surface reelle bati", "sum"),
            Surface_terrain=("Surface terrain", "sum"),
            Surface_Carrez=("Surface Carrez", "sum"),
            Prix_au_m2_mode=("Prix au m2", utils.gauss_mode),
        )
    )

    # compute mean price in addition to mode price
    dvf_communes["Prix_au_m2_moyen"] = (
        dvf_communes["Valeur_fonciere"]
        .div(dvf_communes["Surface_reelle_bati"])
        .where(dvf_communes["Surface_reelle_bati"] > 0)
    )

    # compute reference price (full period mode per commune)
    dvf_communes_ref = (
        dvf_clean.groupby(["Code INSEE"], as_index=False)
        .agg(
            Date_mutation=("Date mutation", "count"), 
            Prix_ref=("Prix au m2", utils.gauss_mode)
        )
    )
    dvf_communes = dvf_communes.join(dvf_communes_ref[["Code INSEE", "Prix_ref"]].set_index("Code INSEE"), on="Code INSEE", how="left")

    # Replace extreme values with ref price
    mask_outlier = dvf_communes["Prix_au_m2_mode"] > 2 * dvf_communes["Prix_ref"]
    dvf_communes.loc[mask_outlier, "Prix_au_m2_mode"] = dvf_communes["Prix_ref"]
    dvf_communes = dvf_communes.rename(columns={"Code INSEE": "Code_INSEE"})

    return dvf_communes

def get_dvf_communes(save_file:str="dvf_communes.csv", is_save:bool=False)->pd.DataFrame:
    dtype_dict={
        "Code_departement": "string",
        "Code_commune": "string",
        "Code_INSEE": "string"
    }

    dvf_communes = utils.read_or_extract(
        group_by_commune,
        save_file=save_file,
        is_save=is_save,
        dtype_dict=dtype_dict
    )
    
    return dvf_communes

def clean_communes(dvf_communes:pd.DataFrame)-> pd.DataFrame:
    nb_years = len(dvf_communes["Annee"].unique())

    # Keep only communes satisfying both conditions : there have been sales every year, and at least 20 of them
    df_copy = dvf_communes.copy() 
    grouped = df_copy.groupby("Code_INSEE")
    nb_years_gr = grouped["Prix_au_m2_mode"].transform("count")
    min_sales_gr = grouped["Nb_ventes"].transform("min")

    dvf_communes_clean = df_copy[(nb_years_gr == nb_years) & (min_sales_gr > 20)]

    # Replace low mode values
    dvf_communes_clean.loc[(dvf_communes_clean["Prix_au_m2_mode"] < 1000), "Prix_au_m2_mode"] = dvf_communes_clean["Prix_au_m2_moyen"]

    # Compute lags
    dvf_communes_clean = dvf_communes_clean.sort_values(["Code_INSEE", "Annee"])
    dvf_communes_clean["Prix_au_m2_mode_n-1"] = dvf_communes_clean.groupby("Code_INSEE")["Prix_au_m2_mode"].shift(1)
    dvf_communes_clean["Prix_au_m2_moyen_n-1"] = dvf_communes_clean.groupby("Code_INSEE")["Prix_au_m2_moyen"].shift(1)

    return dvf_communes_clean

if __name__ == "__main__":

    file_path = "data_test/ValeursFoncieres-2021-2025-sample.txt"

    masks = [
        lambda df: df["Code type local"].isin([1, 2]),
        lambda df: df["Code departement"] == "69"
    ]
    chunks = read_dvf(file_path, masks)
    # Exemple : Calculer la moyenne d'une colonne pour chaque bloc
    test = 1
    for i, chunk in enumerate(chunks):
        print(f"Computing block {i}")
        chunk = reformat_dvf(chunk)
        price_avg = chunk["Prix au m2"].mean()
        price_med = chunk["Prix au m2"].median()
        print(f"Chunk shape: {chunk.shape}, Mean price per m²: {price_avg:.2f}, Median price per m²: {price_med:.2f}")
        if test:
            print(chunk.columns)
            test = 0