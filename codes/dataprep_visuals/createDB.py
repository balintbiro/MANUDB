#importations
import sqlite3
import pandas as pd

#read in the data
numts=pd.read_csv('../data/ncbi_numts.csv')
numts.rename(columns={'order':'taxonomy_order'},inplace=True)

#add ID column for further joins
numts['id']=numts['organism_name']+'_'+numts.index.astype('str')

#connect to DB and initialize cursor
connection=sqlite3.connect('../data/MANUDB.db')
cursor=connection.cursor()

#add tables
cursor.execute("CREATE TABLE statistic(id, eg2_value, e_value)")
cursor.execute("CREATE TABLE location(id, genomic_id, genomic_start, mitochondrial_start, genomic_length, mitochondrial_length, genomic_strand, mitochondrial_strand)")
cursor.execute("CREATE TABLE sequence(id, genomic_sequence, mitochondrial_sequence)")
cursor.execute("CREATE TABLE taxonomy(id, taxonomy_order, family, genus)")

"""#sanity check->should list all the table names
result=cursor.execute("SELECT name FROM sqlite_master")
result.fetchall()"""

#function for getting elements in list of tuples
def fillDB(cols_list:list,query:str)->None:
    tuples_list=[]
    for index,row in numts[cols_list].iterrows():
        tuples_list.append(tuple(row.values))
    cursor.executemany(query, tuples_list)
    connection.commit()

#fillDB with elements
fillDB(
    cols_list=['id','eg2_value','e_value'],
    query="INSERT INTO statistic VALUES(?, ?, ?)"
)
fillDB(
    cols_list=['id','genomic_id','genomic_start','mitochondrial_start','genomic_length','mitochondrial_length','genomic_strand','mitochondrial_strand'],
    query="INSERT INTO location VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
)
fillDB(
    cols_list=['id','genomic_sequence','mitochondrial_sequence'],
    query="INSERT INTO sequence VALUES(?, ?, ?)"
)
fillDB(
    cols_list=['id','taxonomy_order','family','genus'],
    query="INSERT INTO taxonomy VALUES(?, ?, ?, ?)"
)

"""#sanity check
col1='id'
favspec='lutra_lutra'
res = cursor.execute(f"SELECT {col1},family FROM taxonomy WHERE id LIKE '%{favspec}%'")
res.fetchall()"""
