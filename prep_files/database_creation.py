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

cursor.execute(
        """CREATE TABLE statistic(
            id VARCHAR,
            eg2_value FLOAT,
            e_value FLOAT,
            seq_identity FLOAT,
            alignment_score INT
        )"""
    )
cursor.execute(
        """CREATE TABLE location(
            id VARCHAR,
            genomic_id VARCHAR,
            genomic_size INT,
            genomic_start INT,
            mitochondrial_start INT,
            genomic_length INT,
            mitochondrial_length INT,
            genomic_strand CHAR(1),
            mitochondrial_strand CHAR(1)
        )"""
    )
cursor.execute(
        """CREATE TABLE genes(
            id VARCHAR,
            gene VARCHAR,
            mt_gene VARCHAR
        )"""
    )
cursor.execute(
        """CREATE TABLE taxonomy(
            organism_name VARCHAR,
            taxonomy_order VARCHAR,
            family VARCHAR,
            genus VARCHAR,
            assembly_version VARCHAR
        )"""
    )

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
    cols_list=['id','eg2_value','e_value','seq_identity','alignment_score'],
    query="INSERT INTO statistic VALUES(?, ?, ?, ?, ?)",
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
