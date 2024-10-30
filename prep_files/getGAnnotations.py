#import dependencies
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from ftplib import FTP
from subprocess import call, run

#class definition for processing DNA
class Annotation():
    def __init__(self, organism_name):
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
#create classes and apply functions
for organism in organisms:
    organism_class=Annotation(organism)#instantiation
    organism_class.GetGBFF()#download and decompress annotation_file