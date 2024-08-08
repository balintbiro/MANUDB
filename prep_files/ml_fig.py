import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

optimized=pd.read_csv('../results/param_search_result.csv',index_col=0)
selection=pd.read_csv('../../../results/model_selection_results.csv')
selection.columns=['model','AUROC','test_F1','test_Accuracy','test_Precision','test_Recall','test_Jaccard','polygon_area']
feature_selection=pd.read_csv('../../../results/feature_selection_results.csv',index_col=0)

optimization=pd.DataFrame([
    optimized.sort_values(by='polygon_area',ascending=False).reset_index(drop=True).iloc[0][optimized.columns.str.contains('mean_test')].values,
    selection[selection['model']=='XGB'].drop(columns=['model','polygon_area']).values[0]
],columns=['AUROC','F1','Accuracy','Precision','Recall','Jaccard'])
optimization['model']=['Optim','Def']

def radar_plot(categories:list,input_df:pd.DataFrame,names:list,to_be_higlighted:str,lim:tuple,ax:None,legend_pos:tuple,ncols=1)->None:
    categories=[*categories, categories[0]] #to close the radar, duplicategoriese the first column
    n_points=len(categories)
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))#basically the angles for each label
    groups={}
    for index,category in enumerate(input_df.index):
        group=np.array(input_df)[index]
        group=[*group,group[0]]#duplicate the first value to close the radar
        groups[f'group_{index+1}']=group
    colors=['#00FFFF','#FF0000','#000080','#C0C0C0','#000000','#FF0000','#FFFF00','#FF00FF']
    for index,group in enumerate(groups.values()):
        if names[index]==to_be_higlighted:
            ax.plot(label_loc,group,'o-',color='#008000',label=names[index],lw=3,markersize=1.5)
            ax.fill(label_loc, group, color='#008000', alpha=0.1)#the selected is highlighted with green
        else:
            ax.plot(label_loc,group,'o--',color=colors[index],label=names[index],lw=1.5,markersize=.75)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(label_loc), categories)
    for label, angle in zip(ax.get_xticklabels(), label_loc):
        if 0 < angle < np.pi:
            label.set_fontsize(8)
            label.set_horizontalalignment('left')
            label.set_y(.2)
            label.set_x(.2)
        else:
            label.set_fontsize(8)
            label.set_horizontalalignment('right')
            label.set_y(.2)
            label.set_x(.2)
    ax.set_ylim(lim[0],lim[1])
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(color='#AAAAAA')# Change the color of the circular gridlines.
    ax.spines['polar'].set_color('#eaeaea')# Change the color of the outermost gridline (the spine).
    #ax.set_facecolor('#FAFAFA')# Change the background color inside the circle itself.
    ax.legend(loc='upper right', bbox_to_anchor=legend_pos,prop={'size': 8},ncols=ncols)
    return ax


fig=plt.figure(figsize=(8,6.7))
'''fig.set_figheight(8)
fig.set_figwidth(6.7)'''
ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(3, 3), loc=(0, 1), colspan=1)
ax3=plt.subplot2grid(shape=(3,3),loc=(1,0),colspan=1,polar=True)
ax4=plt.subplot2grid(shape=(3,3),loc=(1,1),polar=True)
ax5 = plt.subplot2grid(shape=(3, 3), loc=(2, 0), colspan=2)

ax1.plot(feature_selection['best_auc'],label='Best AUROC\nwith std')
ax1.fill_between(
    feature_selection['n_used']-1,
    (feature_selection['best_auc']-feature_selection['std']),(feature_selection['best_auc']+feature_selection['std']),facecolor='orange',alpha=.3
)
ax1.set_ylabel('AUROC',fontsize=8)
ax1.set_xlabel('Number of tested features',fontsize=8)
ax1.set_title('A',fontsize=16,x=-.25,y=1.05)
ax1.legend(fontsize=8)

ax2.plot(feature_selection['n_used'],feature_selection['n_best'],label='Selected')
ax2.plot(feature_selection['n_used'],feature_selection['n_used'],label='Tested')
ax2.set(ylim=(0,64),xlim=(0,64),xticks=np.arange(start=1,stop=64,step=10),xticklabels=np.arange(start=1,stop=64,step=10),yticks=np.arange(start=1,stop=64,step=10),yticklabels=np.arange(start=1,stop=64,step=10))
#ax2.set_xticklabels(fontsize=8)
ax2.set_xlabel('Number of tested features',fontsize=8)
ax2.set_ylabel('Number of selected features',fontsize=8)
ax2.set_title('B',fontsize=16,x=-.25,y=1.05)
ax2.legend(fontsize=8,loc='best')

radar_plot(
    categories=selection.drop(columns=['model','polygon_area','test_Jaccard']).columns.str.replace('_',' ').str.replace('test ',''),
    input_df=selection.drop(columns=['model','polygon_area','test_Jaccard']),
    names=selection['model'].values,
    to_be_higlighted='XGB',
    lim=(.65,1),
    ax=ax3,
    legend_pos=(.75, -.01),
    ncols=2
)
ax3.set_title('C',fontsize=16,x=-.25,y=1.05)

radar_plot(
    categories=optimization.drop(columns=['model','Jaccard']).columns.str.replace('_',' ').str.replace('test ',''),
    input_df=optimization.drop(columns=['model','Jaccard']),
    names=optimization['model'].values,
    to_be_higlighted='Optim',
    lim=(.88,.98),
    ax=ax4,
    legend_pos=(.75,-.15)
)
ax4.set_title('D',fontsize=16,x=-.25,y=1.05)

params=optimized.columns[optimized.columns.str.contains('param_|polygon')]
heatmap_input=optimized.sort_values(by='polygon_area')[params]
heatmap_input=heatmap_input.apply(lambda col: (col-col.min())/(col.max()-col.min())).T
heatmap_input.index=heatmap_input.index.str.replace('param_','')
sns.heatmap(heatmap_input,cmap='coolwarm_r',ax=ax5)
ax5.set_xticklabels([])
ax5.set_yticks(np.arange(0,len(heatmap_input.index))+0.5)
ax5.set_yticklabels(labels=heatmap_input.index,fontsize=8)
cbar = ax5.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
ax5.set_title('E',fontsize=16,x=-.25,y=1.05)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)


#plt.savefig('../results/ml_plot.png',dpi=800,bbox_inches='tight')
