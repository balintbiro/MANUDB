import pandas as pd
import xgboost
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

features=pd.read_csv('../data/NUMT_features.csv')

#feature scaling
scaler=StandardScaler()
scaled_features=pd.DataFrame(data=scaler.fit_transform(features.drop(columns=['label'])),columns=features.drop(columns=['label']).columns)
y=features['label'].replace(['numt','random'],[0,1]).values
best_features=pd.read_csv('../results/best_features.csv',index_col=0)['0'].tolist()

def model_selection(model_tuple):
    global model_results
    model,name=model_tuple
    cv_res=cross_validate(
            estimator=model,
            X=scaled_features[best_features],y=y,scoring=('roc_auc','f1','accuracy','precision','recall','jaccard'),cv=10
        )
    cv_res['model']=name
    model_results.append(pd.DataFrame(cv_res).drop(columns=['fit_time','score_time']))
    print(f'model done: {name}!')

models=pd.Series(
    data=[
        (xgboost.XGBClassifier(),'XGB'),
        (SVC(),'SVM'),
        (DecisionTreeClassifier(),'DTC'),
        (RandomForestClassifier(),'RFC')
    ]
)

model_results=[]
models.apply(model_selection)

model_results_ready=(
    pd.concat(model_results)
    .groupby(by='model')
    .mean()
    .reset_index()
)
model_results_ready=model_results_ready.rename(columns={'test_roc_auc':'AUROC'})

df=model_results_ready

def radar_plot(categories:list,input_df:pd.DataFrame,names:list,to_be_higlighted:str,lim:tuple)->None:
    categories=[*categories, categories[0]] #to close the radar, duplicategoriese the first column
    n_points=len(categories)
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))#basically the angles for each label
    groups={}
    for index,category in enumerate(input_df.index):
        group=np.array(input_df)[index]
        group=[*group,group[0]]#duplicate the first value to close the radar
        groups[f'group_{index+1}']=group
    plt.figure(figsize=(6, 6), facecolor="white")
    ax=plt.subplot(polar=True)
    colors=['#00FFFF','#FF0000','#000080','#C0C0C0','#000000','#FF0000','#FFFF00','#FF00FF']
    for index,group in enumerate(groups.values()):
        if names[index]==to_be_higlighted:
            ax.plot(label_loc,group,'o-',color='#008000',label=names[index],lw=3)
            ax.fill(label_loc, group, color='#008000', alpha=0.1)#the selected is highlighted with green
        else:
            ax.plot(label_loc,group,'o--',color=colors[index],label=names[index])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(label_loc), categories)
    for label, angle in zip(ax.get_xticklabels(), label_loc):
        if 0 < angle < np.pi:
            label.set_fontsize(16)
            label.set_horizontalalignment('left')
        else:
            label.set_fontsize(16)
            label.set_horizontalalignment('right')
    ax.set_ylim(lim[0],lim[1])
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(color='#AAAAAA')# Change the color of the circular gridlines.
    ax.spines['polar'].set_color('#eaeaea')# Change the color of the outermost gridline (the spine).
    #ax.set_facecolor('#FAFAFA')# Change the background color inside the circle itself.
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.3),prop={'size': 16})
    return ax

radar_plot(
    categories=df.drop(columns=['model']).columns.str.replace('_',' ').str.replace('test ','').str.capitalize(),
    input_df=df.drop(columns=['model']),
    names=df['model'].values,
    to_be_higlighted='XGB',
    lim=(.65,1)
)
plt.tight_layout()
#plt.savefig('../results/radar_plot.png',dpi=800)

model_results_ready=(
    pd.concat(model_results)
    .groupby(by='model')
    .mean()
    .reset_index()
)
model_results_ready=model_results_ready.rename(columns={'test_roc_auc':'AUROC'})

def polygon_area(sides:list)->float:
    angle=360/len(sides)
    all_triangles=sides.copy()
    all_triangles.append(sides[0])
    area=0
    for index,element in enumerate(all_triangles[:-1]):
        area+=0.5*element*all_triangles[index+1]*np.sin(np.deg2rad(angle))
    return area

polygon_areas=model_results_ready.drop(columns=['model']).apply(lambda row: polygon_area(row.tolist()),axis=1)/polygon_area(6*[1])
polygon_areas.index=model_results_ready['model'].values
model_results_ready['polygon_area']=polygon_areas.values

model_results_ready.to_csv('../results/model_selection_results.csv',index=False)
