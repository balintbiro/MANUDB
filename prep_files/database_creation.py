#importations
import sqlite3
import pandas as pd

#read in the data
numts=pd.read_csv('../../../../../Downloads/numts.csv',index_col=0)
assemblies=pd.read_csv('../data/assembly_versions.csv',index_col=0)

#add ID column for further joins
numts['id']=numts['organism_name']+'_'+numts.index.astype('str')

#connect to DB and initialize cursor
connection=sqlite3.connect('../data/MANUDB.db')
cursor=connection.cursor()

#add tables
cursor.execute("CREATE TABLE statistic(id, eg2_value, e_value)")
cursor.execute("CREATE TABLE location(id, genomic_id, genomic_size, genomic_start, mitochondrial_start, genomic_length, mitochondrial_length, genomic_strand, mitochondrial_strand)")
cursor.execute("CREATE TABLE sequence(id, genomic_sequence, mitochondrial_sequence)")
cursor.execute("CREATE TABLE taxonomy(organism_name, taxonomy_order, family, genus, assembly_version)")

#sanity check->should list all the table names
result=cursor.execute("SELECT name FROM sqlite_master")
result.fetchall()

#function for getting elements in list of tuples
def fillDB(cols_list:list,query:str, input_df:pd.DataFrame)->None:
    tuples_list=[]
    for index,row in input_df[cols_list].iterrows():
        tuples_list.append(tuple(row.values))
    cursor.executemany(query, tuples_list)
    connection.commit()

#fillDB with elements
fillDB(
    cols_list=['id','eg2_value','e_value'],
    query="INSERT INTO statistic VALUES(?, ?, ?)",
    input_df=numts
)
fillDB(
    cols_list=['id','genomic_id','genomic_size','genomic_start','mitochondrial_start','genomic_length','mitochondrial_length','genomic_strand','mitochondrial_strand'],
    query="INSERT INTO location VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
    input_df=numts
)
fillDB(
    cols_list=['id','genomic_sequence','mitochondrial_sequence'],
    query="INSERT INTO sequence VALUES(?, ?, ?)",
    input_df=numts
)
fillDB(
    cols_list=['organism_name','taxonomy_order','family','genus','assembly_version'],
    query="INSERT INTO taxonomy VALUES(?, ?, ?, ?, ?)",
    input_df=assemblies
)

pd.read_sql_query("""SELECT * FROM taxonomy where organism_name LIKE '%mus_musculus%'""",connection)
