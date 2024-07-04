import numpy as np
import pandas as pd
from pycirclize import Circos
import matplotlib.pyplot as plt

numts=pd.read_csv('../data/ncbi_numts.csv')
numts.rename(columns={'order':'taxonomy_order'},inplace=True)

rn_numts=numts[numts['organism_name']=='rattus_norvegicus']

report=pd.read_csv('../results/rat_report.txt',sep='\t',comment='#',header=None)
report.columns=['Sequence-Name','Sequence-Role','Assigned-Molecule','Assigned-Molecule-Location/Type','GenBank-Accn','Relationship','RefSeq-Accn','Assembly-Unit','Sequence-Length','UCSC-style-name']

report['Sequence-Name']=report['Sequence-Name'].apply(
    lambda name:
    'scaffold' if 'scaffold' in name
    else (
        name.split('_')[0] if 'unloc' in name
        else name
    )
)
report['Sequence-Name']=report['Sequence-Name'].apply(lambda name: int(name) if name.isnumeric() else name)

mt_size=report[report['Sequence-Name']=='MT']['Sequence-Length'].values[0]+1

rn_numts=rn_numts[(rn_numts['mitochondrial_start']+rn_numts['mitochondrial_length'])<mt_size]

id_dict=pd.Series(index=report['RefSeq-Accn'].values,data=report['Sequence-Name'].values)
rn_numts['sequence_name']=id_dict[rn_numts['genomic_id'].values].values

def chord_plotter(report:pd.DataFrame, numts:pd.DataFrame, cmap:bool, colors:list, ax:None, raw=True)->tuple:#sls stands for scale links and sectors
    numt_sub=numts.copy()
    scaler=1
    if raw==False:
        scaler=1_000_000
    sectors=report.groupby(by='Sequence-Name')['Sequence-Length'].sum().reset_index().apply(
        lambda row: int(row['Sequence-Length']/scaler) if row['Sequence-Name']!='MT' else row['Sequence-Length'],
        axis=1
    )
    sectors.index=report.groupby(by='Sequence-Name')['Sequence-Length'].sum().index
    numt_sub['genomic_start']=numt_sub['genomic_start']/scaler
    numt_sub['genomic_length']=numt_sub['genomic_length']/scaler
    links=numt_sub[['sequence_name','genomic_start','mitochondrial_start','genomic_length','mitochondrial_length']].apply(
        lambda row: (
            ('MT',int(row['mitochondrial_start']),int(row['mitochondrial_start']+row['mitochondrial_length'])),
            (row['sequence_name'],int(row['genomic_start']),int(row['genomic_start']+row['genomic_length']))
        ),axis=1
    ).tolist()
    sectors=sectors.to_dict()
    name2color=dict(zip(sectors.keys(), colors))
    circos=Circos(sectors,space=5)
    for sector in circos.sectors:
        fontsize=6
        track=sector.add_track((95,100))
        if cmap==False:
            track.axis(fc=name2color[sector.name])
        else:
            track.axis(fc='grey')
        if sector.name=='scaffold':
            track.text(sector.name,color='black',size=fontsize,r=120,orientation='vertical')
        elif len(str(sector.name))==2:
            track.text(sector.name,color='black',size=fontsize,r=110,orientation='vertical')
        else:
            track.text(sector.name,color='black',size=fontsize,r=110)

    for link in links:
        circos.link(link[0],link[1],color=name2color[link[1][0]])

    circos.plotfig(ax=ax)

plt.figure(figsize=(6.7, 6.7))
ax1 = plt.subplot(2,2,1,polar=True)
ax2 = plt.subplot(2,2,2,polar=True)
ax3 = plt.subplot(2,2,3,polar=True)
ax4 = plt.subplot(2,2,4,polar=True)

np.random.seed(0)
colors=['#'+''.join(list(np.random.choice(a=list('123456789ABCDEF'), size=6))) for i in range(len(sectors.keys()))]

chord_plotter(report=report,numts=rn_numts,ax=ax1,colors=colors,cmap=False)
chord_plotter(report=report,numts=rn_numts,ax=ax2,colors=colors,raw=False,cmap=False)

norm=Normalize(vmin=min(values),vmax=max(values))
cmap=plt.get_cmap('Reds')
colors=[cmap(norm(value)) for value in rn_numts['genomic_length'].tolist()]
sm=ScalarMappable(cmap=cmap,norm=norm)
sm.set_array([])

chord_plotter(report=report,numts=rn_numts,ax=ax3,colors=colors,cmap=True)
chord_plotter(report=report,numts=rn_numts,ax=ax4,colors=colors,raw=False,cmap=True)

ax1.set_title('A',fontsize=16,x=.1,y=1.05)
ax2.set_title('B',fontsize=16,x=.1,y=1.05)
ax3.set_title('C',fontsize=16,x=.1,y=1.05)
cbar=plt.colorbar(sm,ax=ax3)
cbar.set_label('NUMT size',fontsize=6)
cbar.set_ticklabels([0,200,400,600,800,1000,1200,1400],fontsize=6)

ax4.set_title('D',fontsize=16,x=.1,y=1.05)
cbar=plt.colorbar(sm,ax=ax4)
cbar.set_label('NUMT size',fontsize=6)
cbar.set_ticklabels([0,200,400,600,800,1000,1200,1400],fontsize=6)

plt.savefig('../results/sample_chords.png',dpi=800,bbox_inches='tight')
