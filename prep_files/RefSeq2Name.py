#importations
import json
import sqlite3
import numpy as np
import pandas as pd

con=sqlite3.connect("MANUDB.db")

#get names in a form that would be useful for BioMaRt eg.: mmusculus, hsapiens etc.
locations=pd.read_sql_query("SELECT * FROM location",con=con)
chunks=locations["id"].str.lower().str.split("_")
locations["id"]=chunks.str[0].str[0]+""+chunks.str[1]
names=pd.read_sql_query("SELECT * FROM statistic",con=con)["id"].str.split("_").str[:2].str.join(" ").drop_duplicates()
names=(names.str.split(" ").str[0].str[0].str.lower()+""+names.str.split(" ").str[1])

with open('assemblies.json', 'r') as file:
    data = json.load(file)

#load RefSeq and chr names into individual dfs
dfs=[]
for k,v in data.items():
    orgname=k[0].lower()+k.split("_")[1]
    subdf=pd.DataFrame(data=[v['refseq_accn'],v['assigned_molecule']]).T.dropna()
    subdf.columns=["refseq","name"]
    subdf["orgname"]=orgname
    dfs.append(subdf)

mapper=pd.concat(dfs)

groupedLoc,groupedMap=locations.groupby(by="id"),mapper.groupby(by="orgname")

#function for the actual mapping
def RefSeq2Chr(orgname:str)->pd.DataFrame:
    try:
        Loc,Map=groupedLoc.get_group(orgname),groupedMap.get_group(orgname)
        localMapper=Map.dropna(subset="refseq")
        localMapper=localMapper[localMapper["refseq"]!="na"].set_index("refseq")["name"]
        Loc["chr/name"]=Loc["genomic_id"].apply(lambda gid: localMapper.get(gid,np.nan)).values
        return Loc
    except KeyError:
        return np.nan

chr_ids=names.apply(RefSeq2Chr)

#write output
regions=pd.concat(chr_ids.dropna().tolist())
regions["genomic_end"]=regions["genomic_start"]+regions["genomic_length"]
regions[["id","genomic_start","genomic_end"]].to_csv("temp_files/regions.csv")