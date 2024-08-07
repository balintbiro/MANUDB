import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from pycirclize import Circos
import matplotlib.pyplot as plt

class MANUDB:
    def __init__(self):
        self.name='MANUDB'

    def introduction(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            There is an ongoing process in which mitochondrial sequences are
            being integrated into the nuclear genome. These sequences are called NUclear MiTochondrial sequences (NUMTs)
            (<a href="https://www.sciencedirect.com/science/article/pii/S0888754396901883?via%3Dihub">Lopez et al., 1996</a>).<br>
            The importance of NUMTs has already been revealed in cancer biology
            (<a href="https://link.springer.com/article/10.1186/s13073-017-0420-6">Srinivasainagendra et al., 2017</a>),
            forensic (<a href="https://www.sciencedirect.com/science/article/pii/S1872497321000363">Marshall & Parson, 2021</a>),
            phylogenetic studies (<a href="https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000834">Hazkani-Covo et al., 2010</a>)
            and in the evolution of the eukaryotic genetic information
            (<a href="https://www.sciencedirect.com/science/article/pii/S1055790317302609?via%3Dihub">Nacer & Raposo do Amaral, 2017</a>).<br> 
            Human and numerous model organisms’ genomes were described from the NUMTs point of view.
            Furthermore, recent studies were published on the patterns of these nuclear localised mitochondrial sequences
            in different taxa (<a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0286620">Hebert et al., 2023; 
            <a href="https://www.biorxiv.org/content/10.1101/2022.08.05.502930v2.full.pdf"> Biró et al., 2022</a>). <br>
            However, the results of the previously released studies are difficult to compare 
            due to the lack of standardised methods and/or using few numbers of genomes. To overcome this limitations,
            our group has already published a computational pipeline to mine NUMTs
            (<a href="https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9">Biró et al., 2024</a>). Therefore, our goal with MANUDB is to
            store, visualize, predict and make NUMTs accessible that were mined with our workflow.
            </div>''',
            unsafe_allow_html=True
        )
    
    def status(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            MANUDB currently contains 79 649 NUMTs derived from 153 mammalian genomes of NCBI. These 153 genomes belong to 20 taxonomical orders.
            It supports the retrieval of species specific datasets into a format based on the end user's preference. 
            During the export one can download 14 features (e value, genomic identifier, genomic start position, mitochondrial start position, genomic length,
            mitochondrial length, genomic strand, mitochondrial strand, genomic size, genomic sequence, mitochondrial sequence, genus name, family name and order name).
            Furthermore, MANUDB makes specific NUMT visualizations accessible in downloadable format.
            It is also possible with MANUDB to perform NUMT predictions on .fasta files and on sequences.
            </div>''',
            unsafe_allow_html=True
        )
    
    def contact(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            MANUDB was created and is maintained by the Model Animal Genetics Group
            (PI.: Dr. Orsolya Ivett Hoffmann), Department of Animal Biotechnology,
            Institute of Genetics and Biotechnology,
            Hungarian University of Agriculture and Life Sciences.<br>
            For tehnical queries and or bug report, please contact biro[dot]balint[at]uni-mate[dot]hu or create a pull request at
            <a href="https://github.com/balintbiro/MANUDB">MANUDB's GitHub page</a>.
            </div>''',
            unsafe_allow_html=True
        )
    
    def reference(self)->st.markdown:
        return st.markdown(
            '''<div style="text-align: justify;">
            Biró, B., Gál, Z., Fekete, Z., Klecska, E., & Hoffmann, O. I. (2024). <a href="https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-024-10201-9">
            Mitochondrial genome plasticity of mammalian species.</a> BMC genomics, 25(1), 278.
            </div>''',
            unsafe_allow_html=True
        )
    
class Export:
    def __init__(self):
        self.name='Export'

    def describe_functionality(self):
        return st.markdown(
            '''<div style="text-align: justify;">
            This functionality makes it possible to export the selection of NUMTs based on 
            the selected organism name. 
            Just click on the dropdown above and check the available options. 
            Right now MANUDB supports <a href="https://en.wikipedia.org/wiki/Comma-separated_values">
            .csv format</a> exports. During the export one can download 14 features 
            (e value, genomic identifier, genomic start position, mitochondrial start position, genomic length,
            mitochondrial length, genomic strand, mitochondrial strand, genomic size, genomic sequence, 
            mitochondrial sequence, genus name, family name and order name).
            </div>''',
            unsafe_allow_html=True
        )
    
class Predict:
    def __init__(self):
        self.name='Predict'

    def describe_functionality(self):
        return st.markdown(
            '''<div style="text-align: justify;">
            With this functionality one can predict whether a particular sequence is a NUMT or is not a NUMT.
            To use this functionality just simply paste your <a href="https://en.wikipedia.org/wiki/FASTA_format">
            .FASTA format</a> sequence(s) here. If you click on the 'Example' button above it will paste two 
            correctly formatted sequences into the text area. Then you can use the 'Predict' button to 
            calculate the probability that these two sequences are actual NUMTs. The output of this functionality 
            is downloadable in a <a href="https://en.wikipedia.org/wiki/Comma-separated_values">
            .csv format</a> file which contains the FASTA headers and the corresponding predicted 
            probabilities.<br> 
            Please bear in mind that MANUDB prediction functionality is designed for mammalian sequences. 
            And so if you run prediction on non-mammalian sequences the results will be unreliable.
            </div>''',
            unsafe_allow_html=True
        )
    
class Visualize:
    def __init__(self):
        self.name='Visualize'

    def describe_functionality(self):
        return st.markdown(
            '''<div style="text-align: justify;">
            The visualize functionality makes it possible to visualize the genetic flow from the mitochondria to the
            nuclear genome. MANUDB offers four types of chord diagrams to display NUMTs of a particular genome. 
            Unplaced and unlocalized scaffolds are plotted too in a merged form. We define raw version of visualization 
            where all genomic parts are displayed with their corresponding sizes in bp. The genomic parts are color coded 
            with randomly chosen colors (Figure 1./A). The same raw input can be visualized with proportional coloring 
            with respect to NUMT sizes. In this case the genomic parts are colored uniformly (Figure 1./C). This raw 
            version of visualization is more focused on the genomic parts of chromosomes and scaffolds rather than the 
            mitochondrion. With the help of it one can visualize the flow into individual chromosomes and scaffolds from 
            the mitochondrion. We define optimized version of visualization where all the genomic parts except the 
            mitochondrion are displayed with their corresponding sizes in Mbp. While mitochondrion is displayed in bp. 
            It is possible to color the optimized visualization version randomly or proportionally with respect to NUMT 
            sizes (Figure 1. /B and Figure 1./D respectively). The optimized visualization is focusing on the mitochondrion. 
            Contrary to the raw form of visualization this type is more appropriate if someone would like to visualize 
            the source of the NUMTs within the mitochondrion.
            </div>''',
            unsafe_allow_html=True
        )
    
    def get_dfs(self,organism_name)->tuple:
        with open('queries.json')as json_file:
            queries=json.load(json_file)
        #connect to DB and initialize cursor
        connection=sqlite3.connect('MANUDB.db')
        cursor=connection.cursor()
        numts=pd.read_sql_query(
            queries['Location'].format(organism_name=organism_name.replace(' ','_')),
            connection
        )
        with open('assemblies.json')as infile:
            assemblies=json.load(infile)
        assembly=assemblies[organism_name.replace(' ','_')]
        assembly=pd.DataFrame([
            assembly['assigned_molecule'],
            assembly['length'],
            assembly['sequence_role'],
            assembly['refseq_accn']
        ]).T
        assembly.columns=['molecule','length','role','refseq']
        assembly.loc[assembly['role'].str.contains('unlocal|scaffold'),'molecule']='scaffold'
        assembly['molecule']=assembly['molecule'].apply(lambda name: int(name) if name.isnumeric() else name)
        mapper=pd.Series(data=assembly['molecule'].values,index=assembly['refseq'].values)
        numts['molecule']=mapper[numts['genomic_id'].values].values
        return (numts,assembly)
    
    def get_sectors(self,assembly:pd.DataFrame,scaler=1)->dict:
        assembly['length']=assembly['length'].astype(int)
        sectors=assembly.groupby(by='molecule')['length'].sum().reset_index().apply(
            lambda row: int(int(row['length'])/scaler) if row['molecule']!='MT' else int(row['length']),
            axis=1
        )
        sectors.index=assembly.groupby(by='molecule')['length'].sum().index
        sectors=sectors[sectors>0]
        return sectors.to_dict()
    
    def get_links(self,numts:pd.DataFrame,assembly:pd.DataFrame,scaler=1)->list:
        mt_size=assembly[assembly['molecule']=='MT']['length'].values[0]
        fil=(numts['mitochondrial_start']+numts['mitochondrial_length'])<mt_size
        numts=numts[fil]
        links=numts[['molecule','genomic_start','mitochondrial_start','genomic_length','mitochondrial_length']].apply(
            lambda row: (
                ('MT',int(row['mitochondrial_start']),int(row['mitochondrial_start']+row['mitochondrial_length'])),
                (row['molecule'],int(row['genomic_start']/scaler),int((row['genomic_start']+row['genomic_length'])/scaler))
            ),axis=1
        ).tolist()
        return links
    
    def plotter(self,numts:pd.DataFrame,sectors:dict,links:list,proportional_coloring=False)->None:
        fig,ax=plt.subplots(1,1,figsize=(7,7),subplot_kw={'projection': 'polar'})
        circos=Circos(sectors,space=5)
        for sector in circos.sectors:
            fontsize=12
            track=sector.add_track((95,100))
            if proportional_coloring==True:
                track.axis(fc='grey')
                norm=Normalize(vmin=min(numts['mitochondrial_length']),vmax=max(numts['mitochondrial_length']))
                cmap=plt.get_cmap('Reds')
                colors=[cmap(norm(value)) for value in numts['mitochondrial_length'].tolist()]
                name2color=dict(zip(sectors.keys(), colors))
                sm=ScalarMappable(cmap=cmap,norm=norm)
                sm.set_array([])
            else:
                np.random.seed(0)
                colors=[
                    '#'+''.join(list(np.random.choice(a=list('123456789ABCDEF'), size=6))) for i in range(len(sectors.keys()))
                ]
                name2color=dict(zip(sectors.keys(), colors))
                if sector.name=='MT':
                    track.axis(fc='grey')
                else:
                    track.axis(fc=name2color[sector.name])
            if sector.name=='scaffold':
                track.text(sector.name,color='black',size=fontsize,r=120,orientation='vertical')
            elif len(str(sector.name))==2:
                track.text(sector.name,color='black',size=fontsize,r=110,orientation='vertical')
            else:
                track.text(sector.name,color='black',size=fontsize,r=110)
        for link in links:
            circos.link(link[0],link[1],color=name2color[link[1][0]])
        circos.plotfig(ax=ax)
        return fig
