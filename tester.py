import json
import sqlite3
import numpy as np
import pandas as pd

connection=sqlite3.connect('MANUDBrev.db')

def get_names()->np.array:
    """
    List the names that can be used to query the DB with this method.
    """
    names=(
            pd
            .read_sql_query("SELECT id FROM location",con=connection)
            ["id"]
            .str.split("_")
            .str[:2]
            .str.join("_")
            .drop_duplicates()
            .sort_values()
        )
    name_conversion=pd.read_csv("name_conversion.txt")
    overlap=name_conversion[name_conversion["scientific_name"].isin(names.str.replace('_',' ').values)]
    correct_names=overlap["scientific_name"]+' '+'('+overlap["common_name"]+')'
    return correct_names.values

def get_names():
    with open("assemblies.json")as infile:
        assemblies=json.load(infile)
    names=pd.Series(assemblies.keys()).sort_values()
    name_conversion=pd.read_csv("name_conversion.txt")
    overlap=name_conversion[name_conversion["scientific_name"].isin(names.str.replace('_',' ').values)]
    correct_names=overlap["scientific_name"]+' '+'('+overlap["common_name"]+')'
    return correct_names.values

def get_names()->np.array:
    names=(
            pd
            .read_csv("MtSizes.csv")["orgname"]
            .sort_values()
        )
    name_conversion=pd.read_csv("name_conversion.txt")
    overlap=name_conversion[name_conversion["scientific_name"].isin(names.str.replace('_',' ').values)]
    correct_names=overlap["scientific_name"]+' '+'('+overlap["common_name"]+')'
    return correct_names.values

print(get_names())