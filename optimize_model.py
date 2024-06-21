import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

features=pd.read_csv('../data/NUMT_features.csv')
best_features=pd.read_csv('../results/best_features.csv',index_col=0)['0'].tolist()

#feature scaling
scaler=StandardScaler()
scaled_features=pd.DataFrame(data=scaler.fit_transform(features.drop(columns=['label'])),columns=features.drop(columns=['label']).columns)

X,y=scaled_features,features['label'].replace(['numt','random'],[0,1]).values
X_train,X_test,y_train,y_test=train_test_split(X,y)

def polygon_area(sides:list)->float:
    angle=360/len(sides)
    all_triangles=sides.copy()
    all_triangles.append(sides[0])
    area=0
    for index,element in enumerate(all_triangles[:-1]):
        area+=0.5*element*all_triangles[index+1]*np.sin(np.deg2rad(angle))
    return area

def radar_plot(categories:list,input_df:pd.DataFrame,names:list,to_be_higlighted:str,lim:tuple,to_be_suppressed:str)->None:
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
        elif names[index]==to_be_suppressed:
            ax.plot(label_loc,group,'o--',color='red',label=names[index],lw=1)
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

model_selection_results=pd.read_csv('../results/model_selection_results.csv')

params={
    'min_child_weight': [0.8,0.9,1,1.1,1.2],
   'gamma': [0,0.1,0.2,0.3,0.4,0.5],
   'subsample': [0.7,0.8,0.9,1.0] ,
   'colsample_bytree': [0.5,0.6,0.7,0.8,0.9,1.0] ,
   'max_depth': [3,4,5,6,7,8,9],
   'alpha':[0,0.1,0.2,0.3,0.4,0.5],
   'lambda':[0.8,0.9,1,1.1,1.2],
   'n_estimators':[98,99,100,101,102]
}

cv_=RandomizedSearchCV(
    estimator=xgboost.XGBClassifier(),
    param_distributions=params,
    scoring=('roc_auc','f1','accuracy','precision','recall','jaccard'),cv=10,n_iter=100,
    refit=False
)
cv_.fit(X_train,y_train)
df=pd.DataFrame(cv_.cv_results_)
df['polygon_area']=df.loc[:,df.columns.str.contains('mean_test')].apply(lambda row: polygon_area(row.tolist()),axis=1)/polygon_area(6*[1])


radar_input=pd.DataFrame(
    data=[
        model_selection_results[model_selection_results['model']=='XGB'].drop(columns=['model','polygon_area']).values[0].tolist(),
        df.loc[:,df.columns.str.contains('mean_test|polygon')].sort_values(by='polygon_area',ascending=False).reset_index(drop=True).iloc[0].drop('polygon_area').values.tolist()
    ]
)
radar_input.columns=['AUROC','test_f1','test_accuracy','test_precision','test_recall','test_jaccard']
radar_input.to_csv('../results/optimized_metrics.csv',index=False)

radar_plot(
    categories=radar_input.columns.str.replace('_',' ').str.replace('test ','').str.capitalize(),
    input_df=radar_input,
    names=['default','optimized'],
    to_be_higlighted='optimized',
    to_be_suppressed='default',
    lim=(.825,.98)
)
