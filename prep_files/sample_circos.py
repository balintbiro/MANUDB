import json
import joblib
import sqlite3
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pycirclize import Circos
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

visualize_func=Visualize()
organism_name="Rattus_norvegicus"
numts,assembly,alignment_scores=visualize_func.get_dfs(organism_name=organism_name)

sectors,MtScaler=visualize_func.get_sectors(assembly=assembly)
links=visualize_func.get_links(numts=numts,assembly=assembly,MtScaler=MtScaler)
size_heatmap=pd.Series(sectors.index).apply(visualize_func.heatmap,args=(numts,sectors,MtScaler,))
size_heatmap.index=sectors.index
count_heatmap=pd.Series(sectors.index).apply(visualize_func.heatmap,args=(numts,sectors,MtScaler,True,))
count_heatmap.index=sectors.index

def add_cbar(values:list,title:str,cbar_pos:tuple,cmap_name,ax)->None:
    norm=plt.Normalize(vmin=min(values),vmax=max(values))
    cmap=plt.get_cmap(cmap_name)
    colors=[cmap(norm(value)) for value in values]
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cbar_ax=plt.axes(cbar_pos)
    cbar=plt.colorbar(sm,cax=cbar_ax)
    cbar.ax.set_title(title,fontsize=8)
    cbar.ax.yaxis.label.set_verticalalignment("bottom")
    cbar.ax.yaxis.label.set_position((0.5,1.2))
    cbar.ax.yaxis.set_tick_params(labelsize=6)

def plotter(numts:pd.DataFrame,sectors:dict,links:list,organism_name:str,size_heatmap:pd.Series,count_heatmap:pd.Series,alignment_scores:pd.Series)->None:
    fig,ax=plt.subplots(1,1,figsize=(7,7),subplot_kw={'projection': 'polar'})
    circos=Circos(sectors,space=2)
    fontsize=8
    for sector in circos.sectors:
        track=sector.add_track((93,100))
        track.axis(fc='#008080')
        if sector.name=='scaffold':
            track.text(sector.name,color='black',size=fontsize,r=120,orientation='vertical')
        elif len(str(sector.name))==2:
            track.text(sector.name,color='black',size=fontsize,r=110,orientation='vertical')
        else:
            track.text(sector.name,color='black',size=fontsize,r=110)
        hms_track=sector.add_track((85,92))
        hms_track.axis(fc="none")
        hms_track.heatmap(size_heatmap[sector.name],cmap="Greens")

        hms_track=sector.add_track((77,84))
        hms_track.axis(fc="none")
        hms_track.heatmap(count_heatmap[sector.name],cmap="Greys")
    cmap=plt.cm.coolwarm
    norm=matplotlib.colors.Normalize(vmin=min(alignment_scores),vmax=max(alignment_scores))
    sm=matplotlib.cm.ScalarMappable(cmap="seismic",norm=norm)
    for index,link in enumerate(links):
        circos.link(link[0],link[1],color=cmap(norm(alignment_scores[index])))
    circos.plotfig(ax=ax)
    plt.title(f"",x=.5,y=1.1)
    add_cbar(values=alignment_scores,title="Alignment score",cbar_pos=(-.1,.7,0.015,0.1),cmap_name="coolwarm",ax=ax)
    add_cbar(values=np.concatenate(size_heatmap.values),title="NUMT size (bp)",cbar_pos=(-.1,.5,0.015,0.1),cmap_name="Greens",ax=ax)
    add_cbar(values=np.concatenate(count_heatmap.values),title="NUMT count",cbar_pos=(-.1,.3,0.015,0.1),cmap_name="Greys",ax=ax)
    return fig

plotter(numts,sectors,links,"Rattus_norvegicus",size_heatmap,count_heatmap,alignment_scores)
plt.savefig("SampleSingleSpeciesUseCase.png",dpi=400)