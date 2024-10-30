#importations
import numpy as np
import pandas as pd
from Bio import SeqIO

MtIDs=pd.read_csv("../scratch/MANUDB_MtIDs.csv")

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

df=pd.DataFrame(data=df,columns=["orgname","type","start","end","gene"])
df.to_csv("../scratch/MtAnnotations.csv",index=False)