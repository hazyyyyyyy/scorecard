# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:46:08 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scorecard_class import ScoreCard

random_seed = 7
# from sklearn import tree
# import re

#%% 
#----------------------- 一. 导入数据 -----------------------#
All_Data = pd.read_csv("data_in_.csv")


#%%
#----------------------- 二. 数据预处理 -----------------------#
# *0. Drop D0&D1 Label
All_Data = All_Data.drop( All_Data[All_Data['cust_tag']=='D0'].index, axis=0 )
All_Data = All_Data.drop( All_Data[All_Data['cust_tag']=='D1'].index, axis=0 )

# 1. Drop CUS_NUM, label
All_Data = All_Data.drop(['CUS_NUM'], axis=1)
Y = All_Data['cust_tag']
All_Data = All_Data.drop(['cust_tag'], axis=1)
Y = Y.replace('E', 1)
Y = Y.replace('F', 1)
Y = Y.replace('A', 0)

# 2. Drop Features having "nan">95%;  KEY FUNC: df.count()
FEA_moreNan = list()
non_NA_All_Data = All_Data.count()
for i in non_NA_All_Data.index:
    if ( (len(All_Data)-non_NA_All_Data[i])/len(All_Data)>0.95 ):
        FEA_moreNan.append(i)
All_Data = All_Data.drop(FEA_moreNan, axis=1)

# 3. Drop Features having (same value)>95%; KEY FUNC: df['a'].value_counts()
FEA_moreSame = list()
for i in range(0, All_Data.shape[1]):
    non_nan_counts = All_Data.iloc[:,i].count()
    if ( any(All_Data.iloc[:,i].value_counts() > ( (All_Data.shape[0]-non_nan_counts)*0.95) ) ):
        FEA_moreSame.append(i)
FEA_moreSame = All_Data.columns[FEA_moreSame]
All_Data = All_Data.drop(FEA_moreSame, axis=1)        

# 4. Transfer nan to -99 And 'blank' Respectively
All_Data_types = pd.Series(All_Data.dtypes, dtype='str')

for i in range(0, np.shape(All_Data)[1]): # loop by columns
    # Check nan existence
    if any( All_Data.iloc[:, i].isna() ):
        if All_Data_types[i]=='float64' or All_Data_types[i]=='int64':
            nan_index = np.where(All_Data.iloc[:, i].isna())[0] #np.where() returns a tuple with only 1 element, just extract it.
            All_Data.iloc[nan_index, i] = -99
        else:
            nan_index = np.where(All_Data.iloc[:, i].isna())[0]
            All_Data.iloc[nan_index, i] = 'blank'

#%%
#----------------------- 三. 划分拒绝/放贷样本->对放贷样本划分测试集训练集合 -----------------------#
# 构造放贷/拒绝样本: 设置IK=3
# 放贷样本中 1：4000个，0：3000个
# 拒绝样本中 1：4000个，0：1000个
Good = All_Data.iloc[np.where(Y==1)[0]]
Bad = All_Data.iloc[np.where(Y==0)[0]]

Accept_Good = Good.sample(n=4000,axis=0,random_state=random_seed)
Accept_Bad = Bad.sample(n=3000,axis=0,random_state=random_seed)
Accept = pd.concat([Accept_Good,Accept_Bad], axis=0)

Reject_Good_index = list(set(Good.index.tolist()) - set(Accept.index.tolist()))
Reject_Bad_index = list(set(Bad.index.tolist()) - set(Accept.index.tolist()))
Reject_Good_index = Good.filter(Reject_Good_index, axis=0)
Reject_Bad_index = Bad.filter(Reject_Bad_index, axis=0)

Reject = pd.concat([Reject_Good_index, Reject_Bad_index], axis=0)




x_train, x_test, y_train, y_test = train_test_split(All_Data, Y,
                                                    test_size=0.3, random_state=random_seed)
X = x_train
Y = y_train

#%%

# 评分卡对象
sc = ScoreCard(X,Y)
sc_test = ScoreCard(x_test, y_test)

# woe分箱
binning_return = sc.woe_tree()
print(binning_return['box_num_list'].apply(lambda x:np.sum(x)).unique())

# 用三个模型过滤特征
fea_models_return = sc.filter_feature_by_3_models(binning_return)
x_woe = fea_models_return['x']
y = fea_models_return['y']
Fea_choosed_en_name = fea_models_return['Fea_choosed_en_name']

x_test_woe = sc_test.orig_2_woe(binning_return)

# 用共线性过滤特征
corr_return = sc.filter_feature_by_correlation(x_woe,Fea_choosed_en_name,binning_return)
x = x_woe[corr_return['Fea_choosed_en_name']]

x_test_woe = x_test_woe[corr_return['Fea_choosed_en_name']]

# lr建模
Lr_return = sc.lr(x,y)

# ks
print('-------训练集-------')
auc_ks_return = sc.auc_ks(Lr_return['model'], x, y)
print('-------测试集-------')
auc_ks_return_test = sc_test.auc_ks(Lr_return['model'], x_test_woe, y_test.reset_index(drop=True))






























































