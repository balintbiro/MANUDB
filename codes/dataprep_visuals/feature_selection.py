#importations
import pandas as pd
import numpy as np
import xgboost
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

features=pd.read_csv('../data/NUMT_features.csv')

#feature scaling
scaler=StandardScaler()
scaled_features=pd.DataFrame(data=scaler.fit_transform(features.drop(columns=['label'])),columns=features.drop(columns=['label']).columns)

X,y=scaled_features,features['label'].replace(['numt','random'],[0,1]).values
X_train,X_test,y_train,y_test=train_test_split(X,y)

#funcion for feature selection
def select_features(best_mis:str)->tuple:
    cv_res=cross_validate(
        estimator=LogisticRegression(),
        X=scaled_features[[best_mis]],y=y,
        scoring='roc_auc'
    )
    best_auc=cv_res['test_score'].mean()
    best_std=cv_res['test_score'].std()
    metrics=[]
    best_features=[best_mis]
    used_features=[best_mis]
    np.random.seed(1)
    for i in range(0,scaled_features.shape[1]-1):
        np.random.seed(i)
        to_test=np.random.choice(
            a=list(set(scaled_features.columns)-set(used_features)),
            size=1
        )[0]
        subX=scaled_features[best_features+[to_test]]
        cv_res=cross_validate(
            estimator=LogisticRegression(),
            X=subX,y=y,scoring='roc_auc',cv=10
        )
        mean=cv_res['test_score'].mean()
        std=cv_res['test_score'].std()
        if mean>best_auc:
            metrics.append([
                best_auc,best_std,mean,std,
                len(best_features),len(used_features)
            ])
            best_features.append(to_test)
            best_auc=mean
            best_std=std
        else:
            metrics.append([
                best_auc,best_std,mean,std,
                len(best_features),len(used_features)
            ])
        used_features.append(to_test)
    df=pd.DataFrame(
        data=metrics,
        columns=['best_auc','best_std','auc','std','n_best','n_used']
    )
    return (df,best_features)

mis=mutual_info_classif(X=scaled_features,y=y)

best_mis=pd.Series(data=mis,index=scaled_features.columns).sort_values(ascending=False).index[0]

result,best_features=select_features(best_mis=best_mis)

#create figure
fig,ax1=plt.subplots(1,1,figsize=(6,6))
ln1=ax1.plot(result['best_auc'],label='Best AUROC with std',lw=3)
ax1.fill_between(result['n_used']-1,(result['best_auc']-result['std']),(result['best_auc']+result['std']),facecolor='orange',alpha=.3)

ax2=ax1.twinx()
ln2=ax2.plot((result['n_best'])/result['n_used'],'r--',label='Selected features ratio',lw=3)
ax1.set_ylim(0.65,1)
ax2.set_ylim(0.65,1)

ax1.set_ylabel('AUROC',fontsize=16)
ax1.set_xlabel('Number of tested features',fontsize=16)
ax2.set_ylabel('Features ratio',fontsize=16)
# added these three lines
lns = ln1+ln2
labs = [l.get_label() for l in lns]
#ax1.grid()
plt.legend(lns, labs, loc=0,prop={'size':14})
plt.tight_layout()
#plt.savefig('../results/feature_selection.png',dpi=800)
#result.to_csv('../results/feature_selection_results.csv')
#pd.Series(best_features).to_csv('../results/best_features.csv')
