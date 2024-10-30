#import dependencies
import os
import sqlite3
import numpy as np
import pandas as pd
from ftplib import FTP
from subprocess import call, run

#class definition for processing DNA
class Annotation():
	def __init__(self, organism_name:str):
		self.organism_name=organism_name
	def GetGBFF(self):
		try:
			#connect to the NCBI FTP site
			ftp=FTP("ftp.ncbi.nlm.nih.gov")
			ftp.login()
			#go to the latest assembly version of the given organism
			ftp.cwd(f"/genomes/refseq/vertebrate_mammalian/{self.organism_name}/latest_assembly_versions/")
			LatestAssembly=ftp.nlst()[0]
			URL=f"https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/{self.organism_name}/latest_assembly_versions/{LatestAssembly}/"
			filename=f"{LatestAssembly}_feature_table.txt.gz"
			#download latest genome version
			call(f"wget --output-document=../scratch/annotations/{self.organism_name}_{filename} {URL+filename}",shell=True)
			#decompress genome
			call(f"gzip -d ../scratch/annotations/{self.organism_name}_{filename}",shell=True)
		except:
			print(f"A problem occured during {self.organism_name} annotation acquisition!\nPossibilities are:\n\t-broken FTP connection\n\t-non existing files")
	def HandleGBFF(self,filename:str)->pd.DataFrame:
		cols=['# feature','genomic_accession','chromosome','start','end','symbol']
		df=(
				pd
				.read_csv(filename,usecols=cols,sep="\t")
				.dropna(subset="chromosome")
			)
		return df[
			(df["# feature"]=="gene")
			&(df["symbol"].str.contains("LOC")==False)
		]
	def GetIntragenicNUMTs(self,NUMTs:pd.DataFrame,GBFFdf:pd.DataFrame)->pd.DataFrame:
		commonIDs=list(set(NUMTs["genomic_id"])&set(GBFFdf["genomic_accession"]))
		GBFFdf=GBFFdf[GBFFdf["genomic_accession"].isin(commonIDs)]
		grouped=GBFFdf.groupby(by="genomic_accession")
		def IntragenicNUMTs(row):
			ID=row["genomic_id"]
			if ID in commonIDs:
				subdf=grouped.get_group(ID)
				Nstart,Nend=row["genomic_start"],row["genomic_end"]
				subdf=subdf[
					((Nstart<subdf["start"])&(Nend>subdf["start"]))
					|((Nstart>subdf["start"])&(Nend<subdf["end"]))
					|((Nstart<subdf["end"])&(Nend>subdf["end"]))
					|((Nstart<subdf["start"])&(Nend>subdf["end"]))
				]
				if subdf.shape[0]>0:
					return ",".join(subdf["symbol"].tolist())
				else:
					return np.nan
			else:
				return np.nan
		return NUMTs.apply(IntragenicNUMTs,axis=1)


#connect to SQL	
con=sqlite3.connect("../scratch/MANUDB.db")
MANUDBOrgnames=pd.read_sql_query("SELECT * FROM location",con=con)["id"].str.split("_").str[:2].str.join("_").drop_duplicates()

#connect to NCBI FTP site
ftp=FTP("ftp.ncbi.nlm.nih.gov")
ftp.login()
#change directory to mammalian genomes
ftp.cwd("/genomes/refseq/vertebrate_mammalian/")
#get organism names
organisms=ftp.nlst()
#create folder for genomes
if os.path.isdir("../scratch/annotations/")==False:
	os.mkdir("../scratch/annotations/")
#create classes and apply methods
for organism in organisms:
	organism_class=Annotation(organism)#instantiation
	organism_class.GetGBFF()#download and decompress annotation_file

#get annotations
AnnFileNames=pd.Series(os.listdir("../scratch/annotations/"))
AnnFilePaths= "../scratch/annotations/"+AnnFileNames
AnnOrgNames=AnnFileNames.str.split("_").str[:2].str.join("_")
CommonOrgnames=list(set(MANUDBOrgnames)&set(AnnOrgNames))
#class instantiation
annot=Annotation(organism_name="Sample")

IntragenicNUMTs=[]

for orgname in CommonOrgnames:
	filename=AnnFilePaths[AnnFilePaths.str.contains(orgname)].values[0]
	GBFFdf=annot.HandleGBFF(filename=filename)
	NUMTs=pd.read_sql_query(f"SELECT id,genomic_id,genomic_start,genomic_length FROM location WHERE id LIKE '{orgname}%'",con=con)
	NUMTs["genomic_end"]=NUMTs["genomic_start"]+NUMTs["genomic_length"]
	NUMTs["gene"]=annot.GetIntragenicNUMTs(NUMTs=NUMTs,GBFFdf=GBFFdf)
	IntragenicNUMTs.append(NUMTs[["id","gene"]].dropna())

pd.concat(IntragenicNUMTs).to_csv("../scratch/IntragenicNUMTs.tsv",sep="\t",index=False)

