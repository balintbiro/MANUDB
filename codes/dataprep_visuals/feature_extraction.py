#importations
import pandas as pd
from Bio import SeqIO
from itertools import product

#function to read sequences
def seq_reader(filepath:str,label:str)->None:
    global headers,seqs,labels
    with open(filepath)as infile:
        for record in SeqIO.parse(infile,'fasta'):
            headers.append(record.name)
            seqs.append(str(record.seq))
            labels.append(label)

#et sequences
headers,seqs,labels=[],[],[]
seq_reader(filepath='../data/numt_sequences.fasta',label='numt')
seq_reader(filepath='../data/random_sequences.fasta',label='random')

#sample the full dataset
df=pd.DataFrame(data=[headers,seqs,labels]).T
df.columns=['id','seq','label']
random_state=0
df=pd.concat(
    [
        df[df['label']!='numt'].sample(10000,random_state=random_state),
        df[df['label']=='numt'].sample(10000,random_state=random_state)
    ]
)

#generate kmers
k=3
bases=list('ACGT')
kmers=[''.join(p) for p in product(bases, repeat=k)]

#extract kmers
def feature_extraction(seq:str)->None:
    kmer_counts=[]
    for kmer in kmers:
        kmer_counts.append(seq.count(kmer))
    return kmer_counts

features=df['seq'].apply(feature_extraction)

features=pd.DataFrame(data=features.tolist(),columns=kmers,index=df['id'].values)

#add labels
features['label']=df.label.values

#write df
features.to_csv('../data/NUMT_features.csv',index=False)
