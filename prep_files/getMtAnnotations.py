#importations
import sqlite3
import numpy as np
import pandas as pd
from Bio import SeqIO

MtIDs=pd.read_csv("../scratch/MANUDB_MtIDs.csv")
con=sqlite3.connect("../scratch/MANUDB.db")

orgnames=MtIDs["orgname"].tolist()

#gbff file is from https://ftp.ncbi.nlm.nih.gov/genomes/refseq/mitochondrion/
df=[]
for record in SeqIO.parse("../scratch/mitochondrion.1.genomic.gbff","genbank"):
	orgname=" ".join(record.description.split()[:2])
	if (record.id in MtIDs["mtid"].unique()):
		for feature in record.features:
			if feature.type!="misc_feature":
				if feature.qualifiers.get("gene")!=None:
					if (type(feature.qualifiers.get("gene"))==list):
						df.append([orgname,feature.type,feature.location.start,feature.location.end,feature.qualifiers.get("gene",np.nan)[0]])
					else:
						df.append([orgname,feature.type,feature.location.start,feature.location.end,feature.qualifiers.get("gene",np.nan)])

MtAnnotations=pd.DataFrame(data=df,columns=["orgname","type","start","end","gene"])
MtAnnotations.to_csv("../scratch/MtAnnotations.csv",index=False)

NUMTs=pd.read_sql_query("SELECT * FROM location",con=con)
NUMTs["orgname"]=NUMTs["id"].str.split("_").str[:2].str.join(" ")
MtAnnotations=MtAnnotations[MtAnnotations["type"].isin(["gene","tRNA"])]

def MtIntragenic(row):
	orgname=row["orgname"]
	subdf=MtAnnotations[MtAnnotations["orgname"]==orgname]
	Nstart,Nend=row["mitochondrial_start"],row["mitochondrial_start"]+row["mitochondrial_length"]
	subdf=subdf[
		((Nstart<subdf["start"])&(Nend>subdf["start"]))
		|((Nstart>subdf["start"])&(Nend<subdf["end"]))
		|((Nstart<subdf["end"])&(Nend>subdf["end"]))
		|((Nstart<subdf["start"])&(Nend>subdf["end"]))
	]
	if subdf.shape[0]==0:
		return np.nan
	elif subdf.shape[0]==1:
		return subdf["gene"].values[0]
	else:
		return ",".join(subdf["gene"].tolist())

NUMTs["mt_gene"]=NUMTs.apply(MtIntragenic,axis=1)

#merge it with intragenic NUMTs
NUMTs[["id","mt_gene"]].join(pd.read_csv("../scratch/IntragenicNUMTs.tsv",sep="\t").set_index("id"),on="id").dropna(subset=["mt_gene","gene"],how="all").to_csv("../scratch/allGenes.tsv",sep="\t",index=False)